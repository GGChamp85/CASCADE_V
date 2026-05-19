"""
evaluate.py — Evaluation harness.

Compares CASCADE-V against TRAK-alone, LOO-alone, and Shapley-alone on:

    1. Recovery error: mean absolute error between attributed weights
       and ground-truth weights (instance level).
    2. Creator-DNA consistency: attributed weight at the *creator* level
       compared to ground-truth creator weight. Tests group-wise behavior.
    3. Axiom satisfaction rate: how often the four fairness axioms hold.
    4. Top-1 hit rate: how often the highest-weighted source is one of
       the true contributing sources.

Outputs a CSV and a comparison plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cascade_v.generate import GroundTruthRecord
from cascade_v.types import AttributionResult


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------

@dataclass
class PerOutputMetrics:
    output_id: str
    method: str
    instance_mae: float          # mean absolute error on per-source weights
    creator_mae: float           # mean absolute error on per-creator weights (aggregated)
    top1_hit: bool               # is top-1 attributed source actually a contributor?
    coverage_at_k: float         # fraction of true contributors in attributed top-K
    precision_at_k: float = 0.0  # |attr_topK ∩ truth| / |attr_topK|  (false-positive rate, inverted)
    recall_at_k: float = 0.0     # |attr_topK ∩ truth| / |truth|       (= coverage_at_k; kept for clarity)
    creator_precision_at_k: float = 0.0  # same, aggregated to creator level
    creator_recall_at_k: float = 0.0     # same, aggregated to creator level
    axioms_proven: int = 0       # number of axioms with status PROVEN (CASCADE-V only)
    axioms_total: int = 0        # number of axioms checked (CASCADE-V only)
    is_dna_case: bool = False    # was this a creator-DNA test case?

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AggregateMetrics:
    method: str
    n_outputs: int
    instance_mae_mean: float
    instance_mae_std: float
    creator_mae_mean: float
    creator_mae_std: float
    top1_hit_rate: float
    coverage_at_k_mean: float
    axiom_satisfaction_rate: float
    precision_at_k_mean: float = 0.0
    recall_at_k_mean: float = 0.0
    creator_precision_at_k_mean: float = 0.0
    creator_recall_at_k_mean: float = 0.0
    # Subgroup metrics for DNA cases
    dna_creator_mae_mean: float = 0.0
    dna_n: int = 0


# ---------------------------------------------------------------------------
# Per-output metrics
# ---------------------------------------------------------------------------

def compute_instance_mae(
    gt_record: GroundTruthRecord,
    attribution: AttributionResult,
) -> float:
    """
    MAE between ground-truth source weights and attributed weights.

    Sources in ground truth get their true weight; sources in attribution
    that aren't in ground truth count as having ground-truth weight 0;
    sources in ground truth missing from attribution count as having
    attribution weight 0.
    """
    gt_weights: dict[str, float] = dict(zip(gt_record.source_ids, gt_record.weights))
    attr_weights: dict[str, float] = dict(zip(attribution.source_ids, attribution.weights.tolist()))

    all_ids = set(gt_weights.keys()) | set(attr_weights.keys())
    if not all_ids:
        return 0.0
    diffs = [
        abs(gt_weights.get(sid, 0.0) - attr_weights.get(sid, 0.0))
        for sid in all_ids
    ]
    return float(np.mean(diffs))


def compute_creator_mae(
    gt_record: GroundTruthRecord,
    attribution: AttributionResult,
    source_to_creator: dict[str, str],
) -> float:
    """
    MAE at the creator level (after aggregating both ground truth and
    attribution by creator).
    """
    gt_creators = gt_record.creator_weights

    attr_creators: dict[str, float] = {}
    for sid, w in zip(attribution.source_ids, attribution.weights.tolist()):
        cid = source_to_creator.get(sid, "unknown")
        attr_creators[cid] = attr_creators.get(cid, 0.0) + float(w)

    all_creators = set(gt_creators.keys()) | set(attr_creators.keys())
    if not all_creators:
        return 0.0
    diffs = [
        abs(gt_creators.get(cid, 0.0) - attr_creators.get(cid, 0.0))
        for cid in all_creators
    ]
    return float(np.mean(diffs))


def compute_top1_hit(
    gt_record: GroundTruthRecord,
    attribution: AttributionResult,
) -> bool:
    """True if the top-1 attributed source is in the ground-truth contributors."""
    if len(attribution.weights) == 0:
        return False
    top_idx = int(np.argmax(attribution.weights))
    top_id = attribution.source_ids[top_idx]
    return top_id in gt_record.source_ids


def compute_coverage_at_k(
    gt_record: GroundTruthRecord,
    attribution: AttributionResult,
    k: int | None = None,
) -> float:
    """
    Fraction of true contributors that appear in the top-k attributed
    sources. If k is None, use len(gt_record.source_ids).
    """
    if k is None:
        k = len(gt_record.source_ids)
    if k == 0:
        return 0.0

    # Top-k attributed source IDs
    weights = attribution.weights
    if len(weights) == 0:
        return 0.0
    top_k_idx = np.argsort(-weights)[:k]
    top_k_ids = {attribution.source_ids[i] for i in top_k_idx}

    true_set = set(gt_record.source_ids)
    if not true_set:
        return 0.0
    return len(top_k_ids & true_set) / len(true_set)


def compute_precision_recall_at_k(
    gt_record: GroundTruthRecord,
    attribution: AttributionResult,
    k: int | None = None,
) -> tuple[float, float]:
    """
    Precision@K and recall@K at the source level.

    K defaults to the number of true contributors. Recall@K equals
    coverage_at_k; precision@K complements it by penalizing receipts that
    pad their top-K with false positives. A receipt with 13 creators when
    only 3 are real should have low precision even if recall is high.
    """
    if k is None:
        k = len(gt_record.source_ids)
    if k == 0:
        return 0.0, 0.0
    weights = attribution.weights
    if len(weights) == 0:
        return 0.0, 0.0
    top_k_idx = np.argsort(-weights)[:k]
    top_k_ids = {attribution.source_ids[i] for i in top_k_idx}
    true_set = set(gt_record.source_ids)
    if not true_set:
        return 0.0, 0.0
    intersection = len(top_k_ids & true_set)
    precision = intersection / max(len(top_k_ids), 1)
    recall = intersection / len(true_set)
    return float(precision), float(recall)


def compute_creator_precision_recall_at_k(
    gt_record: GroundTruthRecord,
    attribution: AttributionResult,
    source_to_creator: dict[str, str],
    k: int | None = None,
) -> tuple[float, float]:
    """
    Precision@K / recall@K aggregated to the creator level — answers "how
    many of the top-K *creators* in the receipt are real contributors?"
    K defaults to the number of true contributing creators.
    """
    true_creator_set = set(gt_record.creator_weights.keys())
    if not true_creator_set:
        return 0.0, 0.0
    if k is None:
        k = len(true_creator_set)
    if k == 0:
        return 0.0, 0.0

    creator_weight: dict[str, float] = {}
    for sid, w in zip(attribution.source_ids, attribution.weights.tolist()):
        cid = source_to_creator.get(sid, "unknown")
        creator_weight[cid] = creator_weight.get(cid, 0.0) + float(w)
    if not creator_weight:
        return 0.0, 0.0

    ranked = sorted(creator_weight.items(), key=lambda kv: -kv[1])[:k]
    top_k_creators = {cid for cid, _ in ranked}
    intersection = len(top_k_creators & true_creator_set)
    precision = intersection / max(len(top_k_creators), 1)
    recall = intersection / len(true_creator_set)
    return float(precision), float(recall)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_metrics(per_output: list[PerOutputMetrics]) -> AggregateMetrics:
    """Aggregate per-output metrics by method."""
    if not per_output:
        return AggregateMetrics(
            method="empty", n_outputs=0,
            instance_mae_mean=0, instance_mae_std=0,
            creator_mae_mean=0, creator_mae_std=0,
            top1_hit_rate=0, coverage_at_k_mean=0,
            axiom_satisfaction_rate=0,
        )

    method = per_output[0].method
    inst = np.array([m.instance_mae for m in per_output])
    cre = np.array([m.creator_mae for m in per_output])
    top1 = np.array([float(m.top1_hit) for m in per_output])
    cov = np.array([m.coverage_at_k for m in per_output])

    axioms = [(m.axioms_proven, m.axioms_total) for m in per_output if m.axioms_total > 0]
    if axioms:
        axiom_rate = sum(p for p, t in axioms) / sum(t for p, t in axioms)
    else:
        axiom_rate = float("nan")

    dna_metrics = [m for m in per_output if m.is_dna_case]
    dna_mae = float(np.mean([m.creator_mae for m in dna_metrics])) if dna_metrics else 0.0

    prec = np.array([m.precision_at_k for m in per_output])
    rec = np.array([m.recall_at_k for m in per_output])
    cprec = np.array([m.creator_precision_at_k for m in per_output])
    crec = np.array([m.creator_recall_at_k for m in per_output])

    return AggregateMetrics(
        method=method,
        n_outputs=len(per_output),
        instance_mae_mean=float(inst.mean()),
        instance_mae_std=float(inst.std()),
        creator_mae_mean=float(cre.mean()),
        creator_mae_std=float(cre.std()),
        top1_hit_rate=float(top1.mean()),
        coverage_at_k_mean=float(cov.mean()),
        precision_at_k_mean=float(prec.mean()),
        recall_at_k_mean=float(rec.mean()),
        creator_precision_at_k_mean=float(cprec.mean()),
        creator_recall_at_k_mean=float(crec.mean()),
        axiom_satisfaction_rate=axiom_rate,
        dna_creator_mae_mean=dna_mae,
        dna_n=len(dna_metrics),
    )
