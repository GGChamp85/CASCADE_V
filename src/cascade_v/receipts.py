"""
receipts.py — Payout receipts with formal verification certificates.

The receipt is the artifact that goes to creators. It contains:
    - Per-creator payout breakdown (aggregated from per-source attribution)
    - Per-source attribution with point estimates and intervals
    - Z3 proof certificate status (PROVEN/VIOLATED for each axiom)
    - SMT-LIB proof file path and SHA-256 hash for independent verification
    - Optional `currencies` block (currencies.py): QWS-shaped monetary
      payouts with Rawlsian floor, symbolic recognition roles, lottery
      feature, and opportunity adjustments. Recorded under the same
      receipt schema so dashboards and consumers can pick it up
      additively without breaking existing fields.

This is the audit-grade artifact that an auditor or regulator can verify
independently using only z3 and the SMT-LIB file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from cascade_v.config import (
    CURRENCY_ALPHA,
    CURRENCY_BETA,
    CURRENCY_GAMMA,
    ENABLE_CURRENCIES,
    RECEIPTS_DIR,
)
from cascade_v.pipeline import CascadeVResult


def make_receipt(
    result: CascadeVResult,
    source_to_creator: dict[str, str],
    creator_names: dict[str, str],
    total_payout_usd: float = 1.0,
    catalog_embeddings: Optional[np.ndarray] = None,
    catalog_ids: Optional[list[str]] = None,
    target_embedding: Optional[np.ndarray] = None,
    enable_currencies: Optional[bool] = None,
    currency_alpha: Optional[float] = None,
    currency_beta: Optional[float] = None,
    currency_gamma: Optional[float] = None,
    contributor_history: Optional[dict] = None,
    quality_reference: Optional[np.ndarray] = None,
    seed: int = 0,
) -> dict:
    """
    Build a receipt dict from a CascadeVResult.
    """
    attribution = result.attribution

    # Per-source payouts
    per_source: list[dict] = []
    for i, sid in enumerate(attribution.source_ids):
        cid = source_to_creator.get(sid, "unknown")
        per_source.append({
            "source_id": sid,
            "creator_id": cid,
            "creator_name": creator_names.get(cid, cid),
            "weight_point": float(attribution.weights[i]),
            "weight_lower": float(attribution.intervals[i, 0]) if attribution.intervals is not None else float(attribution.weights[i]),
            "weight_upper": float(attribution.intervals[i, 1]) if attribution.intervals is not None else float(attribution.weights[i]),
            "payout_usd": float(attribution.weights[i] * total_payout_usd),
            "stage_path": attribution.metadata.get("stage_paths", [""] * len(attribution.source_ids))[i],
        })

    # Per-creator aggregation
    per_creator_totals: dict[str, dict] = {}
    for entry in per_source:
        cid = entry["creator_id"]
        if cid not in per_creator_totals:
            per_creator_totals[cid] = {
                "creator_id": cid,
                "creator_name": entry["creator_name"],
                "weight_point": 0.0,
                "weight_lower": 0.0,
                "weight_upper": 0.0,
                "payout_usd": 0.0,
                "n_sources": 0,
            }
        agg = per_creator_totals[cid]
        agg["weight_point"] += entry["weight_point"]
        agg["weight_lower"] += entry["weight_lower"]
        agg["weight_upper"] += entry["weight_upper"]
        agg["payout_usd"] += entry["payout_usd"]
        agg["n_sources"] += 1

    per_creator = sorted(per_creator_totals.values(), key=lambda d: -d["weight_point"])

    # Stage-by-stage breakdown — used by the per-output explainer in the dashboard
    triage_block = {
        "method": result.triage.method,
        "k": int(len(result.triage.top_k_ids)),
        "metadata": result.triage.metadata,
        "candidates": [
            {
                "source_id": sid,
                "creator_id": source_to_creator.get(sid, "unknown"),
                "creator_name": creator_names.get(
                    source_to_creator.get(sid, ""), "unknown"
                ),
                "score": float(result.triage.top_k_scores[i]),
                "rank": i + 1,
            }
            for i, sid in enumerate(result.triage.top_k_ids)
        ],
    }
    grouping_block = {
        "method": result.grouping.method,
        "metadata": result.grouping.metadata,
        "n_clusters": int(len(result.grouping.cluster_ids)),
        "clusters": [
            {
                "cluster_id": int(cid),
                "weight": float(result.grouping.cluster_weights[cid]),
                "raw_score": float(result.grouping.cluster_raw_scores[cid]),
                "member_local_indices": list(result.grouping.members_by_cluster[cid]),
                "member_source_ids": [
                    result.triage.top_k_ids[k]
                    for k in result.grouping.members_by_cluster[cid]
                ],
            }
            for cid in result.grouping.cluster_ids
        ],
    }
    shapley_block = {
        "per_cluster": [
            {
                "cluster_id": int(cid),
                "method": sr.method,
                "n": len(sr.source_indices),
                "metadata": sr.metadata,
                "members": [
                    {
                        "source_id": result.triage.top_k_ids[local_idx],
                        "shapley_value": float(sr.shapley_values[k]),
                        "weight_in_cluster": float(sr.weights[k]),
                        "interval_lower": float(sr.intervals[k, 0]),
                        "interval_upper": float(sr.intervals[k, 1]),
                    }
                    for k, local_idx in enumerate(sr.source_indices)
                ],
            }
            for cid, sr in result.per_cluster_shapley.items()
        ],
    }

    receipt = {
        "receipt_id": result.target_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_payout_usd": total_payout_usd,
        "method": "CASCADE-V",
        "version": "0.2.0",
        "per_creator": per_creator,
        "per_source": per_source,
        "latency_ms": dict(result.latency_ms),
        "stages": {
            "stage1_triage": triage_block,
            "stage2_grouping": grouping_block,
            "stage3_shapley": shapley_block,
        },
        "verification": {
            "overall_status": result.proof.overall_status,
            "axioms": [a.to_dict() for a in result.proof.axioms],
            "smt_lib_file": result.proof.smt_lib_file,
            "smt_lib_hash": result.proof.smt_lib_hash,
        },
        "pipeline_metadata": {
            "catalog_size": result.attribution.metadata.get("catalog_size"),
            "candidates_from_triage": result.attribution.metadata.get("candidates_from_triage"),
            "n_clusters": result.attribution.metadata.get("n_clusters"),
            "validations_passed": all(v.passed for v in result.validations),
            "validation_details": [v.to_dict() for v in result.validations],
        },
    }

    # ---- Optional triple-currency layer ----
    # The currencies block is attached when:
    #   - ENABLE_CURRENCIES is True (or caller forced it on)
    #   - catalog_embeddings + target_embedding are available
    # The currencies layer runs only over active contributors (those that
    # survived Stage-V sparsification) so the Rawlsian floor doesn't
    # re-inflate sources we've intentionally suppressed.
    use_currencies = (
        enable_currencies if enable_currencies is not None else ENABLE_CURRENCIES
    )
    if use_currencies and catalog_embeddings is not None and target_embedding is not None:
        from cascade_v.currencies import (
            CurrencyDials,
            compute_triple_currency_receipt,
        )

        dials = CurrencyDials(
            alpha=currency_alpha if currency_alpha is not None else CURRENCY_ALPHA,
            beta=currency_beta if currency_beta is not None else CURRENCY_BETA,
            gamma=currency_gamma if currency_gamma is not None else CURRENCY_GAMMA,
        )
        # Resolve candidate embeddings from attribution.source_ids — this
        # gives the correct length whether we're on the single-pass path
        # (length = triage_k) or the Demucs-aggregate path (length = union
        # of source_ids across all stems, which can exceed triage_k).
        attr_source_ids = list(attribution.source_ids)
        currency_embeddings: Optional[np.ndarray] = None
        if catalog_ids is None:
            receipt["currencies_error"] = (
                "catalog_ids not provided to make_receipt; cannot map "
                "attribution.source_ids → catalog_embeddings rows."
            )
        else:
            sid_to_row = {sid: i for i, sid in enumerate(catalog_ids)}
            try:
                candidate_indices = np.array(
                    [sid_to_row[sid] for sid in attr_source_ids], dtype=int
                )
                currency_embeddings = catalog_embeddings[candidate_indices]
            except KeyError as e:
                receipt["currencies_error"] = f"source_id not in catalog: {e}"
                currency_embeddings = None
            else:
                pass

        if "currencies_error" not in receipt and currency_embeddings is not None:
            try:
                currency_receipt = compute_triple_currency_receipt(
                    receipt_id=result.target_id,
                    cascade_v_result=result,
                    candidate_embeddings=currency_embeddings,
                    target_embedding=target_embedding,
                    contributor_ids=attr_source_ids,
                    total_payout_usd=total_payout_usd,
                    dials=dials,
                    contributor_history=contributor_history,
                    quality_reference=quality_reference,
                    seed=seed,
                    active_only=True,
                )
                receipt["currencies"] = currency_receipt.to_dict()
            except Exception as e:
                # Don't let the currencies layer break the core receipt.
                # Surface the failure in the receipt itself for triage.
                receipt["currencies_error"] = f"{type(e).__name__}: {e}"

    return receipt


def save_receipt(receipt: dict, receipts_dir: Path = RECEIPTS_DIR) -> Path:
    """Write the receipt JSON to disk and return the path."""
    receipts_dir.mkdir(parents=True, exist_ok=True)
    path = receipts_dir / f"{receipt['receipt_id']}.json"
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2)
    return path
