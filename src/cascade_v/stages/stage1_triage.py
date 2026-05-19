"""
stage1_triage.py — Stage 1: candidate filtering for the catalog.

Given a target output and a large catalog, identify the top-K sources that
plausibly contributed. This is what makes the pipeline scalable: Stages 2
and 3 operate only on the K candidates, not the full catalog.

Algorithm: cosine similarity in the encoder's L2-normalized embedding
space, with a TRAK-inspired centering step.

NOTE on relationship to TRAK (Park et al. 2023, MadryLab/trak):
This is NOT the full TRAK library. TRAK computes per-sample gradient
features via JVP through a target model, random-projects them, and
applies EKFAC preconditioning. We use only the spiritual successor of
TRAK's preconditioning step (subtracting a random-subset baseline) on
top of plain cosine retrieval in our learned encoder space.

Why not real TRAK here: TRAK answers "which training point most
influenced this prediction?" — a counterfactual on model parameters.
Audio attribution answers "which source stems most contributed to this
mix?" — a counterfactual on content. The right Stage-1 is fast retrieval
in a learned embedding space; the formal gradient-influence machinery is
not the right tool for content attribution.

For larger production catalogs, swap this implementation for FAISS HNSW
on the same embedding space (~15 lines in embeddings.py).

Output: a TriageResult with:
    - top_k_indices: indices into the catalog
    - top_k_scores: calibrated cosine scores
    - validation flags (preconditions satisfied)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from cascade_v.config import BPM_FILTER_TOLERANCE, TRIAGE_TOP_K
from cascade_v.utils.audio_meta import keys_compatible


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    top_k_indices: np.ndarray       # (k,) catalog indices
    top_k_scores: np.ndarray        # (k,) similarity scores, descending
    top_k_ids: list[str]            # (k,) source IDs aligned with indices
    method: str = "trak_cosine"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Triage algorithm
# ---------------------------------------------------------------------------

def triage(
    target_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_ids: list[str],
    k: int = TRIAGE_TOP_K,
    use_baseline_calibration: bool = True,
    n_residual_passes: int = 2,
    residual_strength: float = 0.55,
    use_nnls_rerank: bool = True,
    nnls_shortlist_size: int = 80,
    nnls_active_skip_residual: int = 5,
    target_bpm: float | None = None,
    target_key: str | None = None,
    catalog_bpms: np.ndarray | None = None,
    catalog_keys: list[str] | None = None,
    bpm_tolerance: float = BPM_FILTER_TOLERANCE,
) -> TriageResult:
    """
    Run Stage 1 triage and return the top-K candidate source indices.

    Standard cosine retrieval suffers from "dominant-source masking": when one
    contributor dominates the mix (e.g. weight 0.5), the target embedding
    sits near that dominant source and the secondary contributors get pushed
    out of the top-K. We mitigate this with a matching-pursuit-style
    residual pass: after each pass, subtract a scaled version of the top
    candidates' contribution from the target and re-rank to expose the
    next layer of contributors. Final top-K is the union of all passes,
    re-scored on the original (un-residualized) target.

    Args:
        target_embedding: (D,) L2-normalized embedding of the target output
        catalog_embeddings: (N, D) L2-normalized catalog embeddings
        catalog_ids: list of source IDs aligned with catalog_embeddings
        k: number of candidates to return
        use_baseline_calibration: if True, subtract the mean similarity to
            a random subsample (TRAK-inspired centering step).
        n_residual_passes: how many residual passes to run. 0 = vanilla
            cosine retrieval; 2 typically catches the second contributor
            in a 2-source dominant mix; 3+ rarely helps and adds noise.
        residual_strength: how much to subtract per pass. 0.5–0.6 is a
            good range — too high and you over-suppress legitimate matches.

    Returns:
        TriageResult with top-K indices and scores.
    """
    n = len(catalog_embeddings)
    k = min(k, n)

    base_sims = catalog_embeddings @ target_embedding  # (N,)

    # Optional BPM/key pre-filter. We don't drop entries from the catalog
    # arrays (that would re-index everything downstream); instead we build
    # a boolean eligibility mask and zero-out base_sims for ineligible
    # entries so they're effectively unreachable in retrieval. The cost is
    # negligible — one vector compare and a multiply.
    eligible = np.ones(n, dtype=bool)
    n_filtered_bpm = 0
    n_filtered_key = 0
    if target_bpm is not None and catalog_bpms is not None and len(catalog_bpms) == n:
        # Treat catalog_bpms == 0 as "unknown" and don't filter on it.
        bpm_known = catalog_bpms > 0
        bpm_close = np.abs(catalog_bpms - float(target_bpm)) <= float(bpm_tolerance)
        bpm_ok = (~bpm_known) | bpm_close
        n_filtered_bpm = int((~bpm_ok).sum())
        eligible &= bpm_ok
    if target_key and catalog_keys is not None and len(catalog_keys) == n:
        key_ok = np.array(
            [keys_compatible(target_key, k_) for k_ in catalog_keys], dtype=bool
        )
        n_filtered_key = int((~key_ok).sum())
        eligible &= key_ok

    if eligible.sum() < max(k, 1):
        # If gating leaves fewer than k candidates, drop the gate entirely
        # rather than return a degenerate top-K. Better to surface false
        # positives than to miss real contributors.
        eligible = np.ones(n, dtype=bool)
        n_filtered_bpm = 0
        n_filtered_key = 0

    # Force ineligible entries below the floor so they never enter the top-K.
    base_sims = np.where(eligible, base_sims, -1e6)
    if use_baseline_calibration and n > 16:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n, size=min(32, n), replace=False)
        baseline = float(base_sims[sample_idx].mean())
    else:
        baseline = 0.0

    # Pre-NNLS probe: do a cheap NNLS solve on the global top-N by base
    # similarity to see if it already finds enough active contributors.
    # If it does, we skip the residual-pursuit passes entirely. The
    # residual passes were added to fight dominant-source masking, but
    # they also suppress legitimate same-creator near-neighbors — so we
    # only pay that cost when NNLS is genuinely struggling.
    effective_residual_passes = n_residual_passes
    pre_active = -1
    if use_nnls_rerank and nnls_active_skip_residual > 0 and n > 1:
        probe_size = min(nnls_shortlist_size, n)
        probe_idx = np.argpartition(-base_sims, probe_size - 1)[:probe_size]
        try:
            A_probe = catalog_embeddings[probe_idx].T.astype(np.float64)
            w_probe, _ = nnls(A_probe, target_embedding.astype(np.float64), maxiter=100)
            pre_active = int((w_probe > 1e-6).sum())
            if pre_active >= nnls_active_skip_residual:
                effective_residual_passes = 0
        except Exception:
            pass

    # First pass — vanilla cosine + centering
    pass_per_pass = []
    seen: set[int] = set()
    candidates_ordered: list[int] = []

    residual = target_embedding.astype(np.float64).copy()

    for pass_i in range(effective_residual_passes + 1):
        sims = catalog_embeddings @ residual
        scores = sims - baseline
        if k >= n:
            top_idx = np.argsort(-scores).tolist()
        else:
            partition_idx = np.argpartition(-scores, k)[:k]
            top_idx = partition_idx[np.argsort(-scores[partition_idx])].tolist()

        pass_per_pass.append(top_idx)
        for i in top_idx:
            if i not in seen:
                seen.add(int(i))
                candidates_ordered.append(int(i))

        if pass_i == effective_residual_passes:
            break

        # Build residual: subtract scaled contribution of top picks from this pass
        # Use top-3 of this pass weighted by softmax-of-score
        top3 = top_idx[: min(3, len(top_idx))]
        top3_scores = scores[top3]
        weights = np.exp(top3_scores - top3_scores.max())
        weights = weights / weights.sum()
        contribution = (catalog_embeddings[top3] * weights[:, None]).sum(axis=0)
        contribution_norm = float(np.linalg.norm(contribution)) + 1e-9
        contribution = contribution / contribution_norm
        residual = residual - residual_strength * contribution
        rnorm = float(np.linalg.norm(residual)) + 1e-9
        residual = residual / rnorm

    # ------------------------------------------------------------------
    # NNLS re-ranking: solve target ≈ Σ wᵢ · catalog_embeddings[shortlist]
    # subject to wᵢ ≥ 0. Sources with non-zero weight are predicted
    # contributors. Aligned with the encoder's mix-consistency property.
    # ------------------------------------------------------------------
    nnls_meta: dict = {}
    if use_nnls_rerank:
        # Build a shortlist big enough to contain real contributors but
        # small enough for NNLS to solve cheaply. We seed with the union
        # of all residual-pass picks plus the global top-N by base sim.
        shortlist_set = set(candidates_ordered)
        if len(shortlist_set) < nnls_shortlist_size:
            extra = np.argpartition(-base_sims, min(nnls_shortlist_size, n - 1))[
                :nnls_shortlist_size
            ]
            for i in extra:
                shortlist_set.add(int(i))
        shortlist = np.array(sorted(shortlist_set), dtype=int)
        A = catalog_embeddings[shortlist].T.astype(np.float64)  # (D, M)
        b = target_embedding.astype(np.float64)
        try:
            w, residual_norm = nnls(A, b, maxiter=200)
        except Exception:
            w = np.zeros(len(shortlist))
            residual_norm = float("nan")

        nnls_meta = {
            "use_nnls_rerank": True,
            "nnls_shortlist_size": int(len(shortlist)),
            "nnls_residual_norm": float(residual_norm),
            "nnls_n_active": int((w > 1e-6).sum()),
            "nnls_active_mass": float(w.sum()),
        }

        # NNLS-first ranking: NNLS weight is the dominant signal, cosine is
        # a pure tie-breaker (was 0.05 — that gave cosine effective weight
        # comparable to NNLS for any source where NNLS was zero, which is
        # most of them, leading to a long tail of false positives ranked
        # by cosine alone). NNLS is naturally sparse and matches the
        # "few real contributors" prior the pipeline assumes.
        if w.sum() > 1e-9:
            w_norm = w / w.sum()
        else:
            w_norm = w
        ranking_score = w_norm + 1e-4 * (base_sims[shortlist] - baseline)
        order_in_shortlist = np.argsort(-ranking_score)
        final_idx = shortlist[order_in_shortlist][:k]
        # Report the actual ranking score (monotone-decreasing by construction)
        final_scores = ranking_score[order_in_shortlist][:k]
        final_ids = [catalog_ids[i] for i in final_idx]
    else:
        # Fallback: cosine-only reranking (the pre-NNLS path)
        nnls_meta = {"use_nnls_rerank": False}
        cand = np.array(candidates_ordered, dtype=int)
        cand_scores = base_sims[cand] - baseline
        order = np.argsort(-cand_scores)
        final_idx = cand[order][:k]
        final_scores = cand_scores[order][:k]
        final_ids = [catalog_ids[i] for i in final_idx]

    return TriageResult(
        top_k_indices=np.asarray(final_idx),
        top_k_scores=np.asarray(final_scores),
        top_k_ids=final_ids,
        metadata={
            "catalog_size": n,
            "k": k,
            "calibrated": bool(use_baseline_calibration),
            "n_residual_passes_requested": int(n_residual_passes),
            "n_residual_passes_used": int(effective_residual_passes),
            "pre_nnls_active": int(pre_active),
            "residual_strength": float(residual_strength),
            "n_unique_candidates_pre_rerank": int(len(candidates_ordered)),
            "n_filtered_bpm": n_filtered_bpm,
            "n_filtered_key": n_filtered_key,
            "n_eligible_after_filter": int(eligible.sum()),
            "raw_max_similarity": float(base_sims.max()),
            "raw_min_similarity": float(base_sims.min()),
            "raw_mean_similarity": float(base_sims.mean()),
            **nnls_meta,
        },
    )


# ---------------------------------------------------------------------------
# Validation invariants (called by the verification layer)
# ---------------------------------------------------------------------------

def validate_triage_invariants(result: TriageResult) -> dict[str, bool]:
    """
    Check the invariants that should hold for any valid triage result.

    Returns a dict of invariant name -> True/False.
    """
    checks: dict[str, bool] = {}

    # I1: top_k_scores are sorted descending
    diffs = np.diff(result.top_k_scores)
    checks["scores_monotone_decreasing"] = bool((diffs <= 1e-9).all())

    # I2: shapes align
    checks["shapes_aligned"] = (
        len(result.top_k_indices) == len(result.top_k_scores)
        and len(result.top_k_indices) == len(result.top_k_ids)
    )

    # I3: indices are unique
    checks["indices_unique"] = len(set(result.top_k_indices.tolist())) == len(result.top_k_indices)

    # I4: top score is not pathologically low (would indicate broken encoder)
    checks["top_score_nontrivial"] = bool(result.top_k_scores[0] > -0.5)

    return checks
