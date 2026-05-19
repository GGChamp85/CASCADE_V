"""
pipeline_demucs.py — Pipeline wrapper that prepends Demucs Stage 0.

For each input mix:
  1. Run Demucs to separate into 4 stems (drums/bass/other/vocals)
  2. Drop near-silent stems (RMS < threshold) to skip wasted work
  3. Run the standard pipeline on each surviving stem
  4. Aggregate per-source weights as a sum across stems, weighted by
     each stem's energy in the mix
  5. Renormalize so Σ wᵢ = 1, then reuse the proof certificate from the
     loudest-stem attribution as the audit anchor (or generate a fresh
     proof on the aggregated weights — that's what we do here)

Returns a CascadeVResult with the aggregated weights + a proof produced
on those aggregated weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cascade_v.config import PROOFS_DIR, SHAPLEY_VALUE_TEMPERATURE, TRIAGE_TOP_K
from cascade_v.embeddings import embed_audio
from cascade_v.logging_setup import event, get_logger
from cascade_v.pipeline import CascadeVResult, run_cascade_v
from cascade_v.stages.stage0_separate import separate
from cascade_v.types import AttributionResult
from cascade_v.verification.proofs import generate_proof_certificate

_log = get_logger("pipeline_demucs")


def run_cascade_v_with_demucs(
    target_id: str,
    target_audio: np.ndarray,
    sample_rate: int,
    encoder,
    catalog_embeddings: np.ndarray,
    catalog_ids: list[str],
    triage_k: int = TRIAGE_TOP_K,
    shapley_temperature: float = SHAPLEY_VALUE_TEMPERATURE,
    proofs_dir: Path = PROOFS_DIR,
    raise_on_validation_failure: bool = True,
    seed: int = 0,
    on_event=None,
    silence_rms_threshold: float = 0.005,
) -> CascadeVResult:
    """Stage 0 + standard pipeline per stem + aggregate."""
    event(_log, "demucs.start", target_id=target_id)

    # Stage 0: separate
    sep = separate(target_audio, sample_rate=sample_rate)
    energies = {name: float(np.sqrt(np.mean(stem**2))) for name, stem in sep.stems.items()}
    total_energy = sum(energies.values()) + 1e-9
    if on_event:
        on_event("stage0.done", {"stems": list(sep.stems.keys()), "energies": energies})
    event(_log, "demucs.done", energies=energies)

    # Run pipeline per stem (skip silent stems)
    per_stem_results: dict[str, CascadeVResult] = {}
    for name, stem in sep.stems.items():
        if energies[name] < silence_rms_threshold:
            event(_log, "demucs.stem_skipped", stem=name, rms=energies[name])
            continue
        stem_emb = embed_audio(stem, encoder)
        if not np.all(np.isfinite(stem_emb)) or float(np.linalg.norm(stem_emb)) < 1e-9:
            event(_log, "demucs.stem_skipped",
                  stem=name, rms=energies[name], reason="degenerate_embedding")
            continue
        stem_result = run_cascade_v(
            target_id=f"{target_id}__{name}",
            target_embedding=stem_emb,
            catalog_embeddings=catalog_embeddings,
            catalog_ids=catalog_ids,
            triage_k=triage_k,
            shapley_temperature=shapley_temperature,
            proofs_dir=proofs_dir,
            raise_on_validation_failure=raise_on_validation_failure,
            seed=seed,
        )
        per_stem_results[name] = stem_result

    if not per_stem_results:
        # All stems were silent — fall back to running on the raw mix.
        target_emb = embed_audio(target_audio, encoder)
        return run_cascade_v(
            target_id=target_id,
            target_embedding=target_emb,
            catalog_embeddings=catalog_embeddings,
            catalog_ids=catalog_ids,
            triage_k=triage_k,
            shapley_temperature=shapley_temperature,
            proofs_dir=proofs_dir,
            raise_on_validation_failure=raise_on_validation_failure,
            seed=seed,
        )

    # Aggregate: union of all per-stem candidates, weights = Σ_stem (stem_energy_share × per_stem_weight)
    candidate_set: dict[str, dict] = {}  # source_id → {weight, lower, upper, raw}
    for stem_name, res in per_stem_results.items():
        share = energies[stem_name] / total_energy
        for k, sid in enumerate(res.attribution.source_ids):
            entry = candidate_set.setdefault(
                sid, {"weight": 0.0, "lower": 0.0, "upper": 0.0, "raw": 0.0,
                      "stems": []}
            )
            entry["weight"] += share * float(res.attribution.weights[k])
            entry["lower"] += share * float(res.attribution.intervals[k, 0])
            entry["upper"] += share * float(res.attribution.intervals[k, 1])
            entry["raw"] += share * float(res.attribution.raw_scores[k])
            entry["stems"].append(stem_name)

    # Materialize aggregated arrays in canonical order
    agg_ids = sorted(candidate_set.keys())
    agg_w = np.array([candidate_set[sid]["weight"] for sid in agg_ids])
    agg_lo = np.array([candidate_set[sid]["lower"] for sid in agg_ids])
    agg_hi = np.array([candidate_set[sid]["upper"] for sid in agg_ids])
    agg_raw = np.array([candidate_set[sid]["raw"] for sid in agg_ids])

    # Renormalize so Σ w = 1 (efficiency axiom)
    s = agg_w.sum()
    if s > 1e-12:
        agg_w = agg_w / s
        agg_lo = agg_lo / s
        agg_hi = agg_hi / s

    # Build a fresh proof on the aggregated weights so Z3 verifies the final
    # audit-grade artifact.
    sid_to_global = {sid: i for i, sid in enumerate(catalog_ids)}
    agg_global_idx = np.array([sid_to_global[sid] for sid in agg_ids])
    agg_emb = catalog_embeddings[agg_global_idx]
    proof = generate_proof_certificate(
        receipt_id=target_id,
        weights=agg_w,
        embeddings=agg_emb,
        raw_scores=agg_raw,
        target_sum=1.0,
        proofs_dir=proofs_dir,
    )

    # Stitch together a CascadeVResult-like object using the most-energetic stem
    # as the "main" pipeline result for stage details + latency.
    main_stem = max(energies, key=lambda n: energies[n] if n in per_stem_results else -1)
    main = per_stem_results[main_stem]

    attribution = AttributionResult(
        method="cascade_v_demucs",
        source_ids=agg_ids,
        weights=agg_w,
        raw_scores=agg_raw,
        intervals=np.column_stack([agg_lo, agg_hi]),
        stage_path="demucs->cascade_v_per_stem->aggregate",
        metadata={
            "n_clusters": main.attribution.metadata.get("n_clusters"),
            "candidates_from_triage": len(agg_ids),
            "catalog_size": len(catalog_ids),
            "stage_paths": [
                f"demucs[{','.join(candidate_set[sid]['stems'])}]->stage1..V"
                for sid in agg_ids
            ],
            "demucs_energies": energies,
            "demucs_total_energy": total_energy,
            "main_stem": main_stem,
            "n_stems_used": len(per_stem_results),
        },
    )

    # Total latency = sum of per-stem totals + Demucs overhead (we don't time the
    # separate step here; user-visible latency is dominated by the per-stem totals)
    latency_ms = dict(main.latency_ms)
    latency_ms["total"] = sum(
        r.latency_ms.get("total", 0.0) for r in per_stem_results.values()
    )

    return CascadeVResult(
        target_id=target_id,
        attribution=attribution,
        proof=proof,
        triage=main.triage,
        grouping=main.grouping,
        per_cluster_shapley=main.per_cluster_shapley,
        validations=main.validations,
        latency_ms=latency_ms,
    )
