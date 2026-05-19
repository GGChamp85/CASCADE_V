"""
types.py — Shared types for attribution results and value functions.

The `AttributionResult` is the unified return type from every stage and
every method. It carries weights, raw scores, intervals, and stage path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Coalition value function
# ---------------------------------------------------------------------------

# A value function takes a coalition (tuple of source indices) and returns
# v(S) ∈ [0, 1]. It captures "how well do these sources together explain
# the target output?"
ValueFunction = Callable[[tuple[int, ...]], float]


def make_embedding_value_function(
    target_embedding: np.ndarray,
    source_embeddings: np.ndarray,
    temperature: float = 8.0,
) -> ValueFunction:
    """
    Build a value function v(S) based on embedding-space reconstruction.

    v(S) = sharpness( max(0, cosine(mean(source_embeddings[i] for i in S), target)) )
    where sharpness(x) = x ** (temperature / 4.0).

    Properties:
    - v(empty) = 0
    - v(S) ∈ [0, 1] (cosine clamped to [0, 1] since negative similarity means
      the coalition mean points the opposite way from the target — no contribution)
    - Higher `temperature` ⇒ sharper concentration on coalitions whose mean
      sits very close to the target (suppresses the long tail of weak
      contributors that was inflating receipts to 13+ creators).

    Why this form (not exp(-||mean - target||/T)):
    - Plain Euclidean on L2-normalized embeddings was non-monotone in
      coalition size (mean can drift toward the target by removing a
      misaligned source); cosine-of-mean is more directly aligned with
      "do these sources together point toward the target?"
    - Bounded in [0, 1] without dependence on the embedding norm.
    - Behaves well under the sparsification thresholds in pipeline.py.
    """
    target = target_embedding.astype(np.float64)
    target_norm = float(np.linalg.norm(target))
    if target_norm > 1e-12:
        target = target / target_norm
    sources = source_embeddings.astype(np.float64)
    sharpness = float(temperature) / 4.0

    def v(coalition: tuple[int, ...]) -> float:
        if len(coalition) == 0:
            return 0.0
        combined = sources[list(coalition)].mean(axis=0)
        norm = float(np.linalg.norm(combined))
        if norm < 1e-12:
            return 0.0
        cos = float(np.dot(combined, target) / norm)
        cos = max(0.0, min(1.0, cos))
        return cos ** sharpness

    return v


# ---------------------------------------------------------------------------
# Attribution result
# ---------------------------------------------------------------------------

@dataclass
class AttributionResult:
    """
    Result of an attribution computation over a set of sources.

    Fields:
        method: name of the attribution method (e.g. "shapley_exact", "trak", "cascade_v")
        source_ids: list of source IDs in the order corresponding to weights/scores
        weights: normalized payout weights, length n, sum to 1.0 (post clipping)
        raw_scores: unnormalized contribution scores, length n
        intervals: optional (n, 2) array of [lower, upper] uncertainty bounds for each weight
        stage_path: optional human-readable path through the pipeline ("stage1->stage2_cluster_3->stage3_shapley")
        metadata: free-form metadata (n_evaluations, hyperparameters, etc.)
    """
    method: str
    source_ids: list[str]
    weights: np.ndarray
    raw_scores: np.ndarray
    intervals: Optional[np.ndarray] = None
    stage_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.raw_scores = np.asarray(self.raw_scores, dtype=np.float64)
        if self.intervals is not None:
            self.intervals = np.asarray(self.intervals, dtype=np.float64)

    def as_payout(self, total: float = 1.0) -> dict[str, float]:
        """Convert weights to a dict of source_id -> payout amount."""
        return {sid: float(w * total) for sid, w in zip(self.source_ids, self.weights)}

    def as_creator_payout(self, source_to_creator: dict[str, str], total: float = 1.0) -> dict[str, float]:
        """Aggregate weights to creator level."""
        creator_weights: dict[str, float] = {}
        for sid, w in zip(self.source_ids, self.weights):
            cid = source_to_creator[sid]
            creator_weights[cid] = creator_weights.get(cid, 0.0) + float(w * total)
        return creator_weights

    def to_dict(self) -> dict:
        d = asdict(self)
        d["weights"] = self.weights.tolist()
        d["raw_scores"] = self.raw_scores.tolist()
        if self.intervals is not None:
            d["intervals"] = self.intervals.tolist()
        return d


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def normalize_to_payout(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize raw scores to non-negative weights summing to 1.

    Negative scores are clipped to 0 (they represent sources that hurt
    explanation rather than helping; we don't penalize them, just zero them).
    If all scores are non-positive, return uniform weights.
    """
    clipped = np.clip(scores, 0.0, None)
    total = clipped.sum()
    if total < eps:
        return np.ones(len(scores)) / len(scores)
    return clipped / total
