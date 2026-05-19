"""
stage2_grouping.py — Stage 2: cluster candidates, run group-wise attribution.

Algorithm:
  1. Ward-linkage hierarchical clustering on the candidate embeddings.
  2. For each cluster C, compute a group counterfactual: raw[C] = v(N) - v(N\\C).
  3. Normalize to per-cluster weights summing to 1 (clip negatives to 0).

Implementation provenance:
  - Clustering uses scipy.cluster.hierarchy (Ward 1963, off-the-shelf).
  - The group counterfactual itself is the obvious construction —
    "drop a cluster, measure the value-function drop" — three lines built
    on the value function from cascade_v.types.
  - This file is NOT an implementation of a specific named paper. The
    broader literature on group-wise data attribution (group influence
    functions, group-wise unlearning, correlated-training-point
    counterfactuals) inspired the approach, but we don't claim to
    reproduce any specific paper's algorithm.

Why group-level: when the catalog contains highly similar items (e.g.,
multiple stems by the same producer), pure instance-level Shapley splits
credit between near-duplicates, which is correct at the source level but
wrong at the creator level. Group-level counterfactuals assign one weight
to the cluster, then Stage 3 splits within.

Output: a GroupingResult with per-cluster weights and the cluster
assignment for each candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from cascade_v.config import (
    CLUSTER_DISTANCE_THRESHOLD,
    CLUSTERING_METHOD,
    MIN_CLUSTER_WEIGHT,
    SHAPLEY_VALUE_TEMPERATURE,
)
from cascade_v.types import make_embedding_value_function


def _ward_labels(candidate_embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    distances = pdist(candidate_embeddings, metric="euclidean")
    Z = linkage(distances, method="ward")
    return fcluster(Z, t=distance_threshold, criterion="distance")  # 1-indexed


def _hdbscan_labels(candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    Density-based clustering. HDBSCAN labels noise points as -1; we promote
    every noise point to its own singleton cluster so the partition stays
    complete (every candidate must be in exactly one cluster — see
    validate_grouping_invariants I2).

    Imported lazily so HDBSCAN is an optional dependency.
    """
    import hdbscan  # type: ignore[import-not-found]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    raw = clusterer.fit_predict(candidate_embeddings)
    # Promote noise (-1) to singleton clusters with fresh IDs above the max
    next_id = int(raw.max(initial=-1)) + 1
    labels = raw.copy()
    for i, c in enumerate(raw):
        if c == -1:
            labels[i] = next_id
            next_id += 1
    # Make labels 1-indexed to match Ward's fcluster convention
    return labels + 1


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GroupingResult:
    cluster_labels: np.ndarray          # (n_candidates,) cluster ID per candidate
    cluster_ids: list[int]              # unique cluster IDs
    cluster_weights: dict[int, float]   # cluster_id -> normalized weight (sums to 1)
    cluster_raw_scores: dict[int, float]  # cluster_id -> raw counterfactual score
    members_by_cluster: dict[int, list[int]]  # cluster_id -> indices into candidates
    method: str = "guda_ward"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Group-wise attribution
# ---------------------------------------------------------------------------

def cluster_and_attribute(
    target_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
    temperature: float = SHAPLEY_VALUE_TEMPERATURE,
) -> GroupingResult:
    """
    Cluster the candidates and compute group-wise counterfactual attribution.

    Args:
        target_embedding: (D,) target embedding
        candidate_embeddings: (n, D) candidate embeddings (top-K from triage)
        distance_threshold: Ward-linkage cut threshold
        temperature: temperature for the value function

    Returns:
        GroupingResult with cluster labels, weights, and member indices.
    """
    n = len(candidate_embeddings)

    # Edge case: only one candidate -> single cluster
    if n == 1:
        return GroupingResult(
            cluster_labels=np.array([1]),
            cluster_ids=[1],
            cluster_weights={1: 1.0},
            cluster_raw_scores={1: 1.0},
            members_by_cluster={1: [0]},
            metadata={"n_clusters": 1, "n_candidates": 1, "trivial": True},
        )

    # 1. Cluster the candidates. Ward (default) is SMT-stable and well-tuned
    # by CLUSTER_DISTANCE_THRESHOLD; HDBSCAN is density-adaptive and removes
    # the threshold knob (better for variable cluster sizes / outliers).
    # Embeddings are L2-normalized, so Euclidean dist = sqrt(2 - 2·cos(θ)).
    method_choice = (CLUSTERING_METHOD or "ward").lower()
    if method_choice == "hdbscan":
        try:
            labels = _hdbscan_labels(candidate_embeddings)
            method_used = "hdbscan"
        except ImportError:
            # Graceful fallback when hdbscan isn't installed
            labels = _ward_labels(candidate_embeddings, distance_threshold)
            method_used = "ward_fallback_no_hdbscan"
    else:
        labels = _ward_labels(candidate_embeddings, distance_threshold)
        method_used = "ward"

    cluster_ids = sorted(set(labels.tolist()))
    members_by_cluster: dict[int, list[int]] = {cid: [] for cid in cluster_ids}
    for i, c in enumerate(labels):
        members_by_cluster[int(c)].append(i)

    # 2. Build value function on the candidates
    v = make_embedding_value_function(target_embedding, candidate_embeddings, temperature=temperature)

    full_coalition = tuple(range(n))
    v_full = v(full_coalition)

    # 3. Group-wise counterfactual: drop in v(all) when each cluster is removed
    raw_scores: dict[int, float] = {}
    for cid in cluster_ids:
        members = members_by_cluster[cid]
        without_cluster = tuple(i for i in range(n) if i not in members)
        v_without = v(without_cluster)
        raw_scores[cid] = float(v_full - v_without)

    # 4. Normalize to weights summing to 1 (clip negatives, then sparsify)
    raw_array = np.array([raw_scores[cid] for cid in cluster_ids])
    clipped = np.clip(raw_array, 0.0, None)
    if clipped.sum() > 1e-12:
        normalized = clipped / clipped.sum()
    else:
        normalized = np.ones(len(cluster_ids)) / len(cluster_ids)

    # Sparsify: drop clusters below MIN_CLUSTER_WEIGHT, then renormalize.
    # This kills the long tail of weak counterfactuals — clusters that look
    # like contributors only because the encoder ranked one of their members
    # near the target by accident. The efficiency axiom is preserved by the
    # subsequent renormalization.
    n_before = int((normalized > 0).sum())
    sparsified = np.where(normalized >= MIN_CLUSTER_WEIGHT, normalized, 0.0)
    if sparsified.sum() > 1e-12:
        sparsified = sparsified / sparsified.sum()
    else:
        # Fallback: keep the single largest cluster
        sparsified = np.zeros_like(normalized)
        sparsified[int(np.argmax(normalized))] = 1.0
    n_after = int((sparsified > 0).sum())

    cluster_weights = {cid: float(w) for cid, w in zip(cluster_ids, sparsified)}

    return GroupingResult(
        cluster_labels=labels,
        cluster_ids=cluster_ids,
        cluster_weights=cluster_weights,
        cluster_raw_scores=raw_scores,
        members_by_cluster=members_by_cluster,
        metadata={
            "n_clusters": len(cluster_ids),
            "n_candidates": n,
            "v_full": float(v_full),
            "distance_threshold": distance_threshold,
            "clustering_method": method_used,
            "max_cluster_size": max(len(m) for m in members_by_cluster.values()),
            "min_cluster_size": min(len(m) for m in members_by_cluster.values()),
            "n_active_clusters_before_sparsify": n_before,
            "n_active_clusters_after_sparsify": n_after,
            "min_cluster_weight_threshold": float(MIN_CLUSTER_WEIGHT),
        },
    )


# ---------------------------------------------------------------------------
# Validation invariants
# ---------------------------------------------------------------------------

def validate_grouping_invariants(result: GroupingResult, tolerance: float = 1e-3) -> dict[str, bool]:
    """Check invariants for a grouping result."""
    checks: dict[str, bool] = {}

    # I1: cluster weights sum to 1
    total = sum(result.cluster_weights.values())
    checks["weights_sum_to_one"] = bool(abs(total - 1.0) < tolerance)

    # I2: every candidate is in exactly one cluster
    n_candidates = len(result.cluster_labels)
    member_count = sum(len(m) for m in result.members_by_cluster.values())
    checks["partition_complete"] = (member_count == n_candidates)

    # I3: cluster IDs in members map matches cluster_ids
    checks["cluster_ids_consistent"] = (
        set(result.members_by_cluster.keys()) == set(result.cluster_ids)
    )

    # I4: weights are non-negative
    checks["weights_nonneg"] = all(w >= -tolerance for w in result.cluster_weights.values())

    # I5: members are unique within each cluster
    all_members = []
    for m in result.members_by_cluster.values():
        all_members += m
    checks["members_unique"] = (len(set(all_members)) == len(all_members))

    return checks
