"""
stage3_owen.py — Owen values for cooperative games with coalition structure.

When agents are organized into a partition M = {C_1, ..., C_m} (our clusters
from Stage 2), the Shapley value is no longer the right cooperative-game
solution because it assumes a flat coalition. The correct extension is the
**Owen value** (Owen 1977, "Values of games with a priori unions"):

    φ_i^Owen = Σ_{T⊆M\\{C(i)}} Σ_{S⊆C(i)\\{i}} w(T,S) · [v(T'∪S∪{i}) − v(T'∪S)]

where T' = ⋃_{C∈T} C is the union of all clusters in T, and the weight is
the product of Shapley-style weights at both levels:

    w(T, S) = (|T|! · (m−|T|−1)! / m!) · (|S|! · (n_C − |S| − 1)! / n_C!)

The Owen value preserves all four Shapley axioms within each level AND adds
a new "intra-coalition fairness" axiom: agents in the same cluster are
treated symmetrically with respect to each other.

How this differs from the previous CASCADE-V composition:
  Previous: cluster_weight × within_cluster_shapley   (heuristic decomposition)
  Owen:     formal cooperative-game-theoretic extension that satisfies
            cluster-symmetry by construction.

For partitions where clusters are roughly independent (which is our case
after Ward linkage cuts at distance threshold 0.6), the two converge —
but the Owen value is the formally correct answer, and it eliminates a
class of edge-case violations of within-cluster symmetry that the
heuristic decomposition can produce.

Cost: O((2^m) · (2^max_n_C)) where m = number of clusters and max_n_C is
the largest cluster size. With our typical m ≤ 12, max_n_C ≤ 6, this is
~1024 × 64 = 65k coalition evaluations — still well under a second.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from cascade_v.config import SHAPLEY_VALUE_TEMPERATURE
from cascade_v.types import make_embedding_value_function


@dataclass
class OwenResult:
    """Per-source Owen values + composed final weights for the full coalition."""
    source_indices: list[int]            # local indices into candidates (length n)
    owen_values: np.ndarray              # raw Owen values, length n
    weights: np.ndarray                  # normalized non-negative weights summing to 1
    intervals: np.ndarray                # (n, 2) lower/upper bounds (= weights for exact)
    cluster_assignment: np.ndarray       # (n,) cluster index for each source
    method: str = "owen_exact"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def owen_attribute(
    cluster_assignment: list[int] | np.ndarray,
    target_embedding: np.ndarray,
    embeddings: np.ndarray,
    temperature: float = SHAPLEY_VALUE_TEMPERATURE,
) -> OwenResult:
    """Compute Owen values for the candidates given a cluster partition.

    Args:
        cluster_assignment: length-n integer cluster id per candidate
        target_embedding: (D,) target embedding
        embeddings: (n, D) candidate embeddings (already triaged)
        temperature: value-function temperature

    Returns:
        OwenResult with per-source Owen values + normalized weights.
    """
    n = len(embeddings)
    if n == 0:
        return OwenResult(
            source_indices=[],
            owen_values=np.array([]),
            weights=np.array([]),
            intervals=np.zeros((0, 2)),
            cluster_assignment=np.array([], dtype=int),
            metadata={"n": 0, "n_evaluations": 0},
        )

    cluster_assignment = np.asarray(cluster_assignment, dtype=int)
    cluster_ids = sorted(set(cluster_assignment.tolist()))
    m = len(cluster_ids)
    members_by_cluster: dict[int, list[int]] = {
        cid: [int(i) for i in np.where(cluster_assignment == cid)[0]]
        for cid in cluster_ids
    }

    # value function on this candidate subset
    v = make_embedding_value_function(target_embedding, embeddings, temperature=temperature)

    phi = np.zeros(n, dtype=np.float64)
    n_evals = 0

    # For each player i in cluster C(i):
    #   φ_i = Σ_{T ⊆ other_clusters} Σ_{S ⊆ C(i)\{i}}
    #         (|T|! (m-|T|-1)! / m!) · (|S|! (n_C-|S|-1)! / n_C!)
    #         · [v(T' ∪ S ∪ {i}) − v(T' ∪ S)]
    other_clusters_by_player = {
        cid: [oc for oc in cluster_ids if oc != cid] for cid in cluster_ids
    }

    for cid in cluster_ids:
        members = members_by_cluster[cid]
        n_C = len(members)
        others_clusters = other_clusters_by_player[cid]

        # All subsets T of other clusters
        T_subsets: list[tuple[int, ...]] = []
        for size in range(len(others_clusters) + 1):
            T_subsets.extend(itertools.combinations(others_clusters, size))

        for T in T_subsets:
            # T' = union of clusters in T
            T_prime = []
            for cc in T:
                T_prime.extend(members_by_cluster[cc])
            T_prime_tuple = tuple(sorted(T_prime))
            t = len(T)
            cluster_weight = (
                math.factorial(t)
                * math.factorial(m - t - 1)
                / math.factorial(m)
            )

            for i in members:
                others_in_cluster = [j for j in members if j != i]
                # All subsets S of cluster\{i}
                for s_size in range(len(others_in_cluster) + 1):
                    intra_weight = (
                        math.factorial(s_size)
                        * math.factorial(n_C - s_size - 1)
                        / math.factorial(n_C)
                    )
                    weight = cluster_weight * intra_weight
                    for S in itertools.combinations(others_in_cluster, s_size):
                        coalition_without_i = T_prime_tuple + tuple(sorted(S))
                        coalition_with_i = tuple(sorted(coalition_without_i + (i,)))
                        v_with = v(coalition_with_i)
                        v_without = v(coalition_without_i) if coalition_without_i else 0.0
                        phi[i] += weight * (v_with - v_without)
                        n_evals += 2

    # Normalize: clip negatives, sum to 1
    clipped = np.clip(phi, 0.0, None)
    if clipped.sum() > 1e-12:
        weights = clipped / clipped.sum()
    else:
        weights = np.ones(n) / n

    # Exact Owen — intervals are degenerate (point estimates)
    intervals = np.column_stack([weights, weights])

    return OwenResult(
        source_indices=list(range(n)),
        owen_values=phi,
        weights=weights,
        intervals=intervals,
        cluster_assignment=cluster_assignment,
        metadata={
            "n": n,
            "n_clusters": m,
            "n_evaluations": n_evals,
            "v_full": float(v(tuple(range(n)))),
            "temperature": temperature,
            "max_cluster_size": max(len(ms) for ms in members_by_cluster.values()),
        },
    )


def validate_owen_invariants(result: OwenResult, tolerance: float = 1e-3) -> dict[str, bool]:
    n = len(result.source_indices)
    checks: dict[str, bool] = {}
    checks["shapes_aligned"] = (
        len(result.owen_values) == n
        and len(result.weights) == n
        and result.intervals.shape == (n, 2)
    )
    if n == 0:
        return checks
    checks["weights_sum_to_one"] = bool(abs(result.weights.sum() - 1.0) < tolerance)
    checks["weights_nonneg"] = bool((result.weights >= -tolerance).all())
    checks["intervals_contain_weights"] = bool(
        ((result.intervals[:, 0] - tolerance) <= result.weights).all()
        and (result.weights <= (result.intervals[:, 1] + tolerance)).all()
    )
    return checks
