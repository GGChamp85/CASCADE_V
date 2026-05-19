"""
baselines.py — Baseline attribution methods for comparison against CASCADE-V.

Each baseline runs over the *full catalog* (not the triaged subset) where
applicable, to make the comparison fair against the literature where these
methods are presented as standalone solutions.

Methods:
    - trak_alone:    pure TRAK-style cosine + calibration, no clustering or Shapley
    - loo_alone:     leave-one-out on the triaged candidates (Shapley with no game theory)
    - shapley_alone: Monte Carlo Shapley on the triaged candidates (no clustering)
"""

from __future__ import annotations

import numpy as np

from cascade_v.config import (
    SHAPLEY_MC_PERMUTATIONS,
    SHAPLEY_VALUE_TEMPERATURE,
    TRIAGE_TOP_K,
)
from cascade_v.stages.stage1_triage import triage
from cascade_v.stages.stage3_shapley import shapley_monte_carlo
from cascade_v.types import (
    AttributionResult,
    make_embedding_value_function,
    normalize_to_payout,
)


# ---------------------------------------------------------------------------
# TRAK alone (Stage 1 only - no Shapley, no grouping)
# ---------------------------------------------------------------------------

def attribute_trak_alone(
    target_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_ids: list[str],
    k: int = TRIAGE_TOP_K,
) -> AttributionResult:
    """
    Pure TRAK-style attribution: top-K candidates with influence scores
    normalized to weights. Skips grouping and Shapley — this is what the
    TRAK paper alone delivers.
    """
    tri = triage(target_embedding, catalog_embeddings, catalog_ids, k=k)
    weights = normalize_to_payout(tri.top_k_scores)
    return AttributionResult(
        method="trak_alone",
        source_ids=tri.top_k_ids,
        weights=weights,
        raw_scores=tri.top_k_scores,
        intervals=None,
        stage_path="trak_only",
        metadata={"k": k, "calibrated": tri.metadata.get("calibrated", False)},
    )


# ---------------------------------------------------------------------------
# Leave-one-out alone (no grouping)
# ---------------------------------------------------------------------------

def attribute_loo_alone(
    target_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_ids: list[str],
    k: int = TRIAGE_TOP_K,
    temperature: float = SHAPLEY_VALUE_TEMPERATURE,
) -> AttributionResult:
    """
    Leave-one-out attribution on the top-K candidates. No grouping, no Shapley.
    """
    tri = triage(target_embedding, catalog_embeddings, catalog_ids, k=k)
    cand = catalog_embeddings[tri.top_k_indices]
    n = len(cand)
    v = make_embedding_value_function(target_embedding, cand, temperature=temperature)
    full = tuple(range(n))
    v_full = v(full)

    loo = np.zeros(n)
    for i in range(n):
        without = tuple(j for j in range(n) if j != i)
        loo[i] = v_full - v(without)

    weights = normalize_to_payout(loo)
    return AttributionResult(
        method="loo_alone",
        source_ids=tri.top_k_ids,
        weights=weights,
        raw_scores=loo,
        intervals=None,
        stage_path="trak_then_loo",
        metadata={"k": k, "v_full": float(v_full)},
    )


# ---------------------------------------------------------------------------
# Shapley alone (Monte Carlo, no grouping)
# ---------------------------------------------------------------------------

def attribute_shapley_alone(
    target_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_ids: list[str],
    k: int = TRIAGE_TOP_K,
    n_permutations: int = SHAPLEY_MC_PERMUTATIONS,
    temperature: float = SHAPLEY_VALUE_TEMPERATURE,
    seed: int = 0,
) -> AttributionResult:
    """
    Monte Carlo Shapley on the top-K candidates as a flat coalition.
    No clustering — this is what pure Shapley TDA papers deliver.
    """
    tri = triage(target_embedding, catalog_embeddings, catalog_ids, k=k)
    sr = shapley_monte_carlo(
        source_indices=list(range(len(tri.top_k_indices))),
        target_embedding=target_embedding,
        embeddings=catalog_embeddings[tri.top_k_indices],
        n_permutations=n_permutations,
        temperature=temperature,
        seed=seed,
    )

    return AttributionResult(
        method="shapley_alone",
        source_ids=tri.top_k_ids,
        weights=sr.weights,
        raw_scores=sr.shapley_values,
        intervals=sr.intervals,
        stage_path="trak_then_shapley_flat",
        metadata={"k": k, "n_permutations": n_permutations, **sr.metadata},
    )
