"""
pipeline.py — CASCADE-V orchestrator.

Pipeline:
    target_audio
        |  encode
        v
    [Stage 1: triage]      -> top-K candidates       [validate]
    [Stage 2: grouping]    -> clusters + weights     [validate]
    [Stage 3: shapley]     -> per-instance weights   [validate]
    [compose]              -> final per-source weights with intervals
    [Stage V: proof]       -> Z3 fairness certificate

Each stage is timed; latencies surface in the receipt JSON. Validation
failures abort by default (strict mode); pass `raise_on_validation_failure=False`
to record-and-continue.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cascade_v.config import (
    MIN_FINAL_WEIGHT,
    OWEN_OR_DECOMP,
    PROOFS_DIR,
    SHAPLEY_VALUE_TEMPERATURE,
    TRIAGE_TOP_K,
)
from cascade_v.logging_setup import event, get_logger
from cascade_v.stages.stage1_triage import (
    TriageResult,
    triage,
    validate_triage_invariants,
)
from cascade_v.stages.stage2_grouping import (
    GroupingResult,
    cluster_and_attribute,
    validate_grouping_invariants,
)
from cascade_v.stages.stage3_shapley import (
    ShapleyResult,
    shapley_attribute,
    validate_shapley_invariants,
)
from cascade_v.stages.stage3_owen import (
    OwenResult,
    owen_attribute,
    validate_owen_invariants,
)
from cascade_v.types import AttributionResult
from cascade_v.verification.intervals import Interval
from cascade_v.verification.proofs import (
    ProofCertificate,
    generate_proof_certificate,
)
from cascade_v.verification.validators import StageValidation, assert_invariants


_log = get_logger("pipeline")


@dataclass
class CascadeVResult:
    target_id: str
    attribution: AttributionResult
    proof: ProofCertificate
    triage: TriageResult
    grouping: GroupingResult
    per_cluster_shapley: dict[int, ShapleyResult]
    validations: list[StageValidation] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "attribution": self.attribution.to_dict(),
            "proof": self.proof.to_dict(),
            "latency_ms": self.latency_ms,
            "triage": {
                "top_k_ids": self.triage.top_k_ids,
                "top_k_scores": self.triage.top_k_scores.tolist(),
                "metadata": self.triage.metadata,
            },
            "grouping": {
                "n_clusters": len(self.grouping.cluster_ids),
                "cluster_weights": self.grouping.cluster_weights,
                "members_by_cluster": {
                    int(k): v for k, v in self.grouping.members_by_cluster.items()
                },
                "metadata": self.grouping.metadata,
            },
            "validations": [v.to_dict() for v in self.validations],
        }


@contextmanager
def _timed(latency_ms: dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        latency_ms[name] = round((time.perf_counter() - t0) * 1000.0, 3)


def run_cascade_v(
    target_id: str,
    target_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_ids: list[str],
    triage_k: int = TRIAGE_TOP_K,
    shapley_temperature: float = SHAPLEY_VALUE_TEMPERATURE,
    proofs_dir: Path = PROOFS_DIR,
    raise_on_validation_failure: bool = True,
    seed: int = 0,
    on_event: callable = None,
    target_bpm: float | None = None,
    target_key: str | None = None,
    catalog_bpms: np.ndarray | None = None,
    catalog_keys: list[str] | None = None,
) -> CascadeVResult:
    """Run the full CASCADE-V pipeline on a single target.

    on_event(event_name: str, payload: dict) — optional callback fired at every
    stage boundary; used by the FastAPI SSE endpoint to stream progress.
    """
    validations: list[StageValidation] = []
    latency_ms: dict[str, float] = {}

    def emit(name: str, **payload):
        event(_log, name, **payload)
        if on_event is not None:
            try:
                on_event(name, payload)
            except Exception:
                pass

    emit("pipeline.start", target_id=target_id, catalog_size=len(catalog_ids), triage_k=triage_k)

    pipeline_t0 = time.perf_counter()

    # ---------------- Stage 1: Triage ----------------
    with _timed(latency_ms, "stage1_triage"):
        tri = triage(
            target_embedding, catalog_embeddings, catalog_ids, k=triage_k,
            target_bpm=target_bpm, target_key=target_key,
            catalog_bpms=catalog_bpms, catalog_keys=catalog_keys,
        )
        inv1 = validate_triage_invariants(tri)
        validations.append(assert_invariants(
            "stage1_triage", inv1,
            context={"k": triage_k, "catalog_size": len(catalog_ids)},
            raise_on_fail=raise_on_validation_failure,
        ))
    emit("stage1.done", latency_ms=latency_ms["stage1_triage"], top_k=len(tri.top_k_ids),
         max_score=float(tri.top_k_scores[0]))

    candidate_indices = tri.top_k_indices
    candidate_embeddings = catalog_embeddings[candidate_indices]
    candidate_ids = tri.top_k_ids

    # ---------------- Stage 2: Grouping ----------------
    with _timed(latency_ms, "stage2_grouping"):
        grp = cluster_and_attribute(
            target_embedding, candidate_embeddings,
            temperature=shapley_temperature,
        )
        inv2 = validate_grouping_invariants(grp)
        validations.append(assert_invariants(
            "stage2_grouping", inv2,
            context={"n_candidates": len(candidate_indices)},
            raise_on_fail=raise_on_validation_failure,
        ))
    emit("stage2.done", latency_ms=latency_ms["stage2_grouping"],
         n_clusters=len(grp.cluster_ids))

    # ---------------- Stage 3: Owen value or per-cluster Shapley ----------------
    n_candidates = len(candidate_indices)
    per_cluster_shapley: dict[int, ShapleyResult] = {}
    owen_result: OwenResult | None = None

    use_owen = OWEN_OR_DECOMP == "owen"

    with _timed(latency_ms, "stage3_shapley_total"):
        if use_owen:
            # Owen cost is O(2^m · 2^max_cluster_size). Stage 2 sparsification
            # already zeroed out clusters below MIN_CLUSTER_WEIGHT — those
            # clusters are predicted not to contribute, so we drop them
            # entirely from the Owen partition. This typically takes
            # m: 12 → 3-5, making Owen feasible (32 × 32 ≈ 1024 coalitions).
            active_cids = [
                cid for cid, w in grp.cluster_weights.items() if w > 1e-9
            ]
            active_members: list[int] = []
            cluster_of_active: dict[int, int] = {}  # local index → cluster id
            for cid in active_cids:
                for k in grp.members_by_cluster[cid]:
                    cluster_of_active[k] = cid
                    active_members.append(k)
            active_members.sort()
            local_to_global = {i: g for i, g in enumerate(active_members)}
            global_to_local = {g: i for i, g in enumerate(active_members)}
            cluster_of = np.array(
                [cluster_of_active[active_members[i]] for i in range(len(active_members))],
                dtype=int,
            )
            owen_subset_emb = candidate_embeddings[active_members]
            owen_result = owen_attribute(
                cluster_assignment=cluster_of,
                target_embedding=target_embedding,
                embeddings=owen_subset_emb,
                temperature=shapley_temperature,
            )
            inv3 = validate_owen_invariants(owen_result)
            validations.append(assert_invariants(
                "stage3_owen", inv3,
                context={
                    "n_active": len(active_members),
                    "n_active_clusters": len(active_cids),
                    "n_total_clusters": len(grp.cluster_ids),
                },
                raise_on_fail=raise_on_validation_failure,
            ))
            emit("stage3.cluster_done", cluster_id=-1,
                 n=len(active_members), method=owen_result.method)
        else:
            for cid, member_indices in grp.members_by_cluster.items():
                sr = shapley_attribute(
                    source_indices=member_indices,
                    target_embedding=target_embedding,
                    embeddings=candidate_embeddings,
                    temperature=shapley_temperature,
                    seed=seed + cid,
                )
                per_cluster_shapley[cid] = sr
                inv3 = validate_shapley_invariants(sr)
                validations.append(assert_invariants(
                    f"stage3_shapley_cluster_{cid}", inv3,
                    context={"cluster_id": cid, "members": member_indices},
                    raise_on_fail=raise_on_validation_failure,
                ))
                emit("stage3.cluster_done", cluster_id=cid,
                     n=len(member_indices), method=sr.method)
    emit("stage3.done", latency_ms=latency_ms["stage3_shapley_total"])

    # ---------------- Compose: build final per-source weights ----------------
    with _timed(latency_ms, "compose"):
        final_weights = np.zeros(n_candidates)
        final_intervals = np.zeros((n_candidates, 2))
        final_raw = np.zeros(n_candidates)
        stage_paths = [""] * n_candidates

        if use_owen:
            # Owen value gave us weights for the ACTIVE subset only.
            # Map back to the full candidate vector; non-active candidates get 0.
            for local_i, global_idx in local_to_global.items():
                final_weights[global_idx] = owen_result.weights[local_i]
                final_intervals[global_idx] = [
                    owen_result.intervals[local_i, 0],
                    owen_result.intervals[local_i, 1],
                ]
                final_raw[global_idx] = owen_result.owen_values[local_i]
                cid = int(owen_result.cluster_assignment[local_i])
                stage_paths[global_idx] = f"stage1->stage2_cluster_{cid}->stage3_owen"
            for k in range(n_candidates):
                if k not in global_to_local:
                    stage_paths[k] = "stage1->stage2_dropped"
        else:
            for cid, member_indices in grp.members_by_cluster.items():
                cw = grp.cluster_weights[cid]
                cw_interval = Interval.point_value(cw)
                sr = per_cluster_shapley[cid]
                instance_intervals_in_cluster = [
                    Interval(point=sr.weights[k], lower=sr.intervals[k, 0], upper=sr.intervals[k, 1])
                    for k in range(len(member_indices))
                ]
                composed = [cw_interval * iv for iv in instance_intervals_in_cluster]
                for local_k, global_idx in enumerate(member_indices):
                    final_weights[global_idx] = composed[local_k].point
                    final_intervals[global_idx] = [composed[local_k].lower, composed[local_k].upper]
                    final_raw[global_idx] = cw * sr.shapley_values[local_k]
                    stage_paths[global_idx] = f"stage1->stage2_cluster_{cid}->stage3_shapley"

        s = final_weights.sum()
        if s > 1e-12:
            final_weights = final_weights / s
            final_intervals = final_intervals / s

        # Sparsify final weights: drop sources whose final weight falls below
        # MIN_FINAL_WEIGHT (false-positive control), then renormalize so the
        # efficiency axiom (Σ w = 1) still holds.
        below = final_weights < MIN_FINAL_WEIGHT
        if below.any() and (~below).any():
            final_weights[below] = 0.0
            final_intervals[below] = 0.0
            s2 = final_weights.sum()
            if s2 > 1e-12:
                final_weights = final_weights / s2
                final_intervals = final_intervals / s2

    # ---------------- Stage V: Proof certificate ----------------
    with _timed(latency_ms, "stageV_proof"):
        proof = generate_proof_certificate(
            receipt_id=target_id,
            weights=final_weights,
            embeddings=candidate_embeddings,
            raw_scores=final_raw,
            target_sum=1.0,
            proofs_dir=proofs_dir,
        )
    emit("stageV.done", latency_ms=latency_ms["stageV_proof"], status=proof.overall_status)

    latency_ms["total"] = round((time.perf_counter() - pipeline_t0) * 1000.0, 3)

    attribution = AttributionResult(
        method="cascade_v",
        source_ids=candidate_ids,
        weights=final_weights,
        raw_scores=final_raw,
        intervals=final_intervals,
        stage_path="cascade_v_3_stage",
        metadata={
            "n_clusters": len(grp.cluster_ids),
            "candidates_from_triage": len(candidate_indices),
            "catalog_size": len(catalog_ids),
            "stage_paths": stage_paths,
            "latency_ms": dict(latency_ms),
        },
    )

    emit("pipeline.end", target_id=target_id, total_ms=latency_ms["total"],
         status=proof.overall_status,
         all_validations_passed=all(v.passed for v in validations))

    return CascadeVResult(
        target_id=target_id,
        attribution=attribution,
        proof=proof,
        triage=tri,
        grouping=grp,
        per_cluster_shapley=per_cluster_shapley,
        validations=validations,
        latency_ms=latency_ms,
    )
