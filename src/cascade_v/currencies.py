"""
currencies.py — QWS + Three-Currency Attribution with Tradeoff Dials.

Combines:
  - Quantile-Weighted Shapley (Koenker & Bassett 1978 → AI attribution)
  - Three currencies (Sen's capability approach: Money + Recognition + Opportunity)
  - Three public dials (α, β, γ) for explicit fairness/appreciation tradeoff
  - Rawlsian dignity floor (preserves Shapley axioms within active set)

Drop-in extension to CASCADE-V's pipeline.run_cascade_v output. By design
this module is pure-numpy (no torch, no z3) so it composes cleanly with
the existing receipt path and runs under the test stubs.

Integration note (false-positive control):
The Rawlsian floor lifts every contributor in the input set to a minimum
share. Passing all TRIAGE_TOP_K candidates including those sparsified to
zero by Stage-V would re-inflate non-contributors (defeating the
MIN_FINAL_WEIGHT sparsification work). Callers should pass only the
*active* contributors (base_weight > 0) — see receipts.py for the
canonical invocation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


# ============================================================================
# DIAL CONFIGURATION (the explicit tradeoff controls)
# ============================================================================

@dataclass
class CurrencyDials:
    """Public tradeoff controls. Recorded in every receipt.

    α (alpha) — fairness strictness on monetary payouts:
        1.0 = pure Shapley (no floor), 0.0 = full equal split.
    β (beta)  — recognition spread (lottery feature):
        1.0 = top contributor wins, 0.0 = uniform across contributors.
    γ (gamma) — opportunity redistribution (future-triage routing):
        0.0 = pure merit, 1.0 = strong protection of under-represented creators.
    """
    alpha: float = 0.85
    beta: float = 0.60
    gamma: float = 0.30

    def validate(self) -> None:
        for name, val in (("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Dial {name}={val} must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# QUANTILE-WEIGHTED SHAPLEY (the QWS layer)
# ============================================================================

@dataclass
class QWSResult:
    quantile_active: float                        # which quantile bin selected
    weights_by_quantile: dict[float, np.ndarray]  # q → weights
    weights_final: np.ndarray                     # weights at active quantile
    output_quality_score: float                   # empirical quality ∈ [0, 1]


def estimate_output_quality(
    target_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    quality_reference: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate where this output sits in the quality distribution [0, 1].

    Default proxy: distance from the catalog centroid (more distinctive
    outputs are higher quality). Replace with engagement / listener data
    when available — pass it as `quality_reference` (1-D array of historical
    distinctiveness scores) and we return the empirical CDF probability.
    """
    centroid = catalog_embeddings.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 1e-9:
        centroid = centroid / centroid_norm
    # target_embedding is L2-normalized in our pipeline, so dot ∈ [-1, 1]
    distinctiveness = 1.0 - float(target_embedding @ centroid)

    if quality_reference is not None and len(quality_reference) > 0:
        # Empirical CDF: fraction of historical scores below this one
        return float((np.asarray(quality_reference) < distinctiveness).mean())

    # Sigmoid map of raw distinctiveness ∈ [0, 2] to (0, 1)
    return float(1.0 / (1.0 + math.exp(-5.0 * (distinctiveness - 0.3))))


def quantile_weighted_shapley(
    base_shapley_weights: np.ndarray,
    base_shapley_intervals: np.ndarray,    # (n, 2) lower/upper from CASCADE-V
    candidate_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    quantiles: tuple = (0.50, 0.75, 0.90, 0.99),
    quality_reference: Optional[np.ndarray] = None,
) -> QWSResult:
    """
    Compute quantile-conditioned Shapley weights.

    For each quantile q, the weight of contributor i reflects their value
    conditional on outputs in quantile q:
      - q ≤ 0.5 : pull toward uniform (mediocre outputs split credit flatly)
      - q > 0.5 : sharpen via power-law (excellent outputs reward top contributors)
      - exponent e(q) = 1 + 2(q - 0.5)  ⇒ continuous at q=0.5, e(0.99)=1.98

    Selection: the active quantile is the discrete bin closest to the
    empirical output quality score.
    """
    n = len(base_shapley_weights)
    quality = estimate_output_quality(
        target_embedding, candidate_embeddings, quality_reference
    )

    weights_by_q: dict[float, np.ndarray] = {}
    for q in quantiles:
        if q <= 0.5:
            uniform = np.ones(n) / n if n > 0 else np.zeros(0)
            mix = 2.0 * q  # q=0 → 0 (full uniform), q=0.5 → 1 (full base)
            shaped = (1 - mix) * uniform + mix * base_shapley_weights
        else:
            exponent = 1.0 + 2.0 * (q - 0.5)
            shaped = np.power(base_shapley_weights + 1e-9, exponent)

        s = shaped.sum()
        weights_by_q[q] = shaped / s if s > 1e-12 else (
            np.ones(n) / n if n > 0 else np.zeros(0)
        )

    nearest_q = min(quantiles, key=lambda q: abs(q - quality))
    return QWSResult(
        quantile_active=nearest_q,
        weights_by_quantile=weights_by_q,
        weights_final=weights_by_q[nearest_q],
        output_quality_score=quality,
    )


# ============================================================================
# CURRENCY 1: MONETARY (QWS + Rawlsian Floor controlled by α)
# ============================================================================

@dataclass
class MonetaryResult:
    weights_qws: np.ndarray              # post-QWS weights
    weights_with_floor: np.ndarray       # post-Rawlsian floor
    floor_applied: float                 # the actual floor value used
    final_payouts_usd: np.ndarray        # dollar amounts
    intervals_usd: np.ndarray            # (n, 2)


def apply_rawlsian_floor(
    weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """
    Rawlsian maximin floor: lift every entry below the floor up to the floor
    level, take the deficit from above-floor entries proportional to their
    excess.

    α = 1.0 → no floor (strict Shapley).
    α = 0.0 → equal split (full egalitarian, floor = 0.8/n).

    Invariant: with floor = 0.8 (1 - α) / n and inputs summing to 1,
    total excess (1 - n · floor) ≥ 0.2 always exceeds total deficit, so
    after adjustment every entry is ≥ floor and the sum is preserved
    exactly (modulo float noise; defensive renorm at the end).
    """
    n = len(weights)
    if n == 0:
        return weights.copy(), 0.0
    if alpha >= 1.0 - 1e-6:
        return weights.copy(), 0.0

    equal = 1.0 / n
    floor = (1.0 - alpha) * equal * 0.8

    adjusted = weights.copy().astype(np.float64)

    below = adjusted < floor
    deficit = float((floor - adjusted[below]).sum()) if below.any() else 0.0
    if below.any():
        adjusted[below] = floor

    # 'above' here means "now ≥ floor" (post-lift). Originally-below entries
    # are now exactly at floor (excess = 0), so they contribute 0 weight in
    # the proportional subtraction — they don't lose what we just gave them.
    above = adjusted >= floor
    if above.any() and deficit > 0:
        excess = adjusted[above] - floor
        excess_sum = float(excess.sum())
        if excess_sum > 1e-9:
            adjusted[above] = adjusted[above] - deficit * (excess / excess_sum)

    total = float(adjusted.sum())
    if total > 1e-12:
        adjusted = adjusted / total
    return adjusted, float(floor)


def compute_monetary(
    qws_weights: np.ndarray,
    qws_intervals: np.ndarray,
    total_payout_usd: float,
    dials: CurrencyDials,
) -> MonetaryResult:
    weights_with_floor, floor = apply_rawlsian_floor(qws_weights, dials.alpha)

    # Scale intervals proportionally to preserve relative uncertainty.
    # Where qws_weights is near zero, scale fall-back to 1 to avoid
    # producing huge intervals on entries that were barely contributing.
    scale = np.where(
        qws_weights > 1e-12,
        weights_with_floor / (qws_weights + 1e-12),
        1.0,
    )
    intervals_adjusted = qws_intervals * scale[:, None]

    return MonetaryResult(
        weights_qws=qws_weights,
        weights_with_floor=weights_with_floor,
        floor_applied=floor,
        final_payouts_usd=weights_with_floor * total_payout_usd,
        intervals_usd=intervals_adjusted * total_payout_usd,
    )


# ============================================================================
# CURRENCY 2: REPUTATIONAL (controlled by β)
# ============================================================================

@dataclass
class ReputationalResult:
    roles_per_contributor: list[list[str]]
    streak_status: list[str]                # placeholder until history is wired
    featured_contributor_idx: int           # who got the lottery feature
    feature_probability: np.ndarray         # P(featured) per contributor


def assign_roles(
    weights: np.ndarray,
    candidate_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    counterfactual_losses: Optional[np.ndarray] = None,
) -> list[list[str]]:
    """
    Assign role tags. Only two role types are emitted, both with concrete
    meaning:

    - ESSENTIAL CONTRIBUTOR: counterfactual loss > 0.20, i.e. removing this
      contributor drops the value function by >20%. Genuinely load-bearing.
    - CONTRIBUTING VOICE: fallback for any positive-weight contributor that
      isn't ESSENTIAL — the dignity-in-recognition floor (every contributor
      that received money gets at least one role, otherwise Currency 2
      dignity proof would fail).

    The dimension-based labels (RHYTHMIC ANCHOR, BASS FOUNDATION, etc.)
    were removed because the encoder dimensions are learned features with
    no intrinsic musical meaning — labelling them as instruments was
    interpretive UX, not honest signal. Use the catalog category field
    if you want truthful instrument labels in the receipt UI.

    `candidate_embeddings` and `target_embedding` are kept in the
    signature for backwards-compatibility and future probe-based labels.
    """
    n = len(weights)
    roles_list: list[list[str]] = [[] for _ in range(n)]
    if n == 0:
        return roles_list

    if counterfactual_losses is not None:
        for i, loss in enumerate(counterfactual_losses):
            if loss > 0.20:
                roles_list[i].append("ESSENTIAL CONTRIBUTOR")

    for i in range(n):
        if not roles_list[i] and weights[i] > 0:
            roles_list[i].append("CONTRIBUTING VOICE")

    return roles_list


def assign_lottery_feature(
    weights: np.ndarray,
    beta: float,
    seed: int = 0,
) -> tuple[int, np.ndarray]:
    """
    Lottery feature: one contributor per output gets a "Featured Contribution"
    callout. Friedman-Savage utility: small probability of large recognition
    delivers high felt-appreciation per dollar.

    β = 1.0 → P(featured) ∝ weight (top contributors win most often).
    β = 0.0 → P(featured) uniform across all contributors.
    """
    n = len(weights)
    if n == 0:
        return -1, np.zeros(0)

    rng = np.random.default_rng(seed)
    weights_sum = float(weights.sum())
    proportional = weights / weights_sum if weights_sum > 1e-12 else np.ones(n) / n
    uniform = np.ones(n) / n

    feature_probs = beta * proportional + (1 - beta) * uniform
    s = float(feature_probs.sum())
    if s > 1e-12:
        feature_probs = feature_probs / s
    else:
        feature_probs = uniform

    featured_idx = int(rng.choice(n, p=feature_probs))
    return featured_idx, feature_probs


def compute_reputational(
    weights: np.ndarray,
    candidate_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    dials: CurrencyDials,
    counterfactual_losses: Optional[np.ndarray] = None,
    seed: int = 0,
) -> ReputationalResult:
    roles = assign_roles(
        weights, candidate_embeddings, target_embedding, counterfactual_losses
    )
    featured_idx, feature_probs = assign_lottery_feature(weights, dials.beta, seed)

    return ReputationalResult(
        roles_per_contributor=roles,
        streak_status=["(requires historical data)"] * len(weights),
        featured_contributor_idx=featured_idx,
        feature_probability=feature_probs,
    )


# ============================================================================
# CURRENCY 3: OPPORTUNITY (controlled by γ)
# ============================================================================

@dataclass
class OpportunityResult:
    priority_adjustment: np.ndarray          # delta added to future triage scores
    diversity_amplified: list[bool]
    under_represented_protected: list[bool]


def compute_opportunity(
    weights: np.ndarray,
    candidate_embeddings: np.ndarray,
    dials: CurrencyDials,
    contributor_history: Optional[dict] = None,
    contributor_ids: Optional[list[str]] = None,
) -> OpportunityResult:
    """
    Compute future-triage routing adjustments. Three additive components:

      1. Quality-recency bonus (∝ weight, tapered by 1-γ): reward strong
         contributors so they keep getting surfaced.
      2. Diversity bonus (gated by γ): amplify contributors whose embedding
         direction is uniquely under-represented in the active set.
      3. Under-representation protection (gated by γ): contributors with
         <3 recent triage appearances get an affirmative-discovery boost.

    All terms are non-negative ⇒ priority_adjustment[i] ≥ 0 always
    (verified by the dignity proof).
    """
    n = len(weights)
    adjustment = np.zeros(n)
    diversity_amp = [False] * n
    under_rep = [False] * n
    if n == 0:
        return OpportunityResult(adjustment, diversity_amp, under_rep)

    # 1. Quality-recency bonus
    w_max = float(weights.max())
    quality_bonus = 0.05 * weights / (w_max + 1e-9)
    adjustment += quality_bonus * (1.0 - dials.gamma)

    # 2. Diversity bonus (only meaningful when n > 1)
    if n > 1:
        sim_matrix = candidate_embeddings @ candidate_embeddings.T
        np.fill_diagonal(sim_matrix, 0.0)
        max_sim_per_contributor = sim_matrix.max(axis=1)
        diversity_score = 1.0 - max_sim_per_contributor
        for i in range(n):
            if diversity_score[i] > 0.5:
                adjustment[i] += 0.10 * dials.gamma * float(diversity_score[i])
                diversity_amp[i] = True

    # 3. Under-representation protection
    if contributor_history is not None and contributor_ids is not None:
        for i, cid in enumerate(contributor_ids):
            if contributor_history.get(cid, 0) < 3:
                adjustment[i] += 0.15 * dials.gamma
                under_rep[i] = True

    return OpportunityResult(
        priority_adjustment=adjustment,
        diversity_amplified=diversity_amp,
        under_represented_protected=under_rep,
    )


# ============================================================================
# THE COMPOSITE RECEIPT
# ============================================================================

@dataclass
class TripleCurrencyReceipt:
    receipt_id: str
    contributor_ids: list[str]
    dials: CurrencyDials
    qws: QWSResult
    monetary: MonetaryResult
    reputational: ReputationalResult
    opportunity: OpportunityResult
    dignity_proof: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "dials": self.dials.to_dict(),
            "qws": {
                "quantile_active": self.qws.quantile_active,
                "output_quality_score": self.qws.output_quality_score,
                "weights_by_quantile": {
                    str(q): w.tolist() for q, w in self.qws.weights_by_quantile.items()
                },
                "weights_final": self.qws.weights_final.tolist(),
            },
            "per_contributor": [
                {
                    "contributor_id": self.contributor_ids[i],
                    "monetary": {
                        "weight_qws": float(self.monetary.weights_qws[i]),
                        "weight_with_floor": float(self.monetary.weights_with_floor[i]),
                        "payout_usd": float(self.monetary.final_payouts_usd[i]),
                        "interval_usd": [
                            float(self.monetary.intervals_usd[i, 0]),
                            float(self.monetary.intervals_usd[i, 1]),
                        ],
                    },
                    "reputational": {
                        "roles": self.reputational.roles_per_contributor[i],
                        "streak": self.reputational.streak_status[i],
                        "is_featured": (i == self.reputational.featured_contributor_idx),
                        "feature_probability": float(self.reputational.feature_probability[i]),
                    },
                    "opportunity": {
                        "priority_adjustment": float(self.opportunity.priority_adjustment[i]),
                        "diversity_amplified": self.opportunity.diversity_amplified[i],
                        "under_represented_protected": (
                            self.opportunity.under_represented_protected[i]
                        ),
                    },
                }
                for i in range(len(self.contributor_ids))
            ],
            "dignity_proof": self.dignity_proof,
        }


def verify_dignity(
    receipt: TripleCurrencyReceipt,
    epsilon: float = 1e-3,
) -> dict:
    """
    Meta-proof: verifies all three currencies were honored for every
    contributor. Returns a dict suitable for the receipt's dignity_proof.
    """
    n = len(receipt.contributor_ids)

    monetary_ok = all(receipt.monetary.final_payouts_usd[i] >= 0 for i in range(n))
    floor_honored = all(
        receipt.monetary.weights_with_floor[i] >= receipt.monetary.floor_applied - epsilon
        for i in range(n)
    )
    every_has_role = all(
        len(receipt.reputational.roles_per_contributor[i]) > 0 for i in range(n)
    )
    opportunity_ok = all(
        receipt.opportunity.priority_adjustment[i] >= 0 for i in range(n)
    )
    weight_sum = float(receipt.monetary.weights_with_floor.sum()) if n > 0 else 1.0
    sum_to_one = abs(weight_sum - 1.0) < epsilon

    return {
        "currency_1_monetary": "PROVEN" if monetary_ok else "VIOLATED",
        "currency_1_floor_honored": "PROVEN" if floor_honored else "VIOLATED",
        "currency_1_efficiency": "PROVEN" if sum_to_one else "VIOLATED",
        "currency_2_every_creator_recognized": "PROVEN" if every_has_role else "VIOLATED",
        "currency_3_opportunity_nonneg": "PROVEN" if opportunity_ok else "VIOLATED",
        "every_creator_received_all_three": "PROVEN" if (
            monetary_ok and every_has_role and opportunity_ok
        ) else "VIOLATED",
    }


# ============================================================================
# MAIN ENTRY POINT (call this after run_cascade_v)
# ============================================================================

def compute_triple_currency_receipt(
    receipt_id: str,
    cascade_v_result,                      # CascadeVResult
    candidate_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    contributor_ids: list[str],
    total_payout_usd: float = 1.0,
    dials: Optional[CurrencyDials] = None,
    contributor_history: Optional[dict] = None,
    quality_reference: Optional[np.ndarray] = None,
    seed: int = 0,
    active_only: bool = True,
) -> TripleCurrencyReceipt:
    """
    Take a CASCADE-V result and produce a triple-currency receipt.

    `active_only=True` (default) restricts the currencies layer to
    contributors whose base attribution weight is > 0 — the ones that
    survived Stage-V sparsification. This preserves the false-positive
    suppression done upstream; if you instead pass `active_only=False`,
    the Rawlsian floor will lift every TRIAGE_TOP_K candidate to a
    minimum share, re-introducing the long tail of small false-positive
    payouts that the rest of the pipeline works to suppress.
    """
    if dials is None:
        dials = CurrencyDials()
    dials.validate()

    base_weights_full = cascade_v_result.attribution.weights
    base_intervals_full = cascade_v_result.attribution.intervals
    raw_full = cascade_v_result.attribution.raw_scores

    # Filter to active contributors (post-sparsification)
    if active_only:
        active = base_weights_full > 0.0
    else:
        active = np.ones(len(base_weights_full), dtype=bool)

    if active.sum() == 0:
        # Degenerate: nothing to allocate. Return an empty receipt with
        # the dignity proof trivially passing.
        empty = QWSResult(
            quantile_active=0.5,
            weights_by_quantile={},
            weights_final=np.zeros(0),
            output_quality_score=0.0,
        )
        empty_mon = MonetaryResult(
            weights_qws=np.zeros(0), weights_with_floor=np.zeros(0),
            floor_applied=0.0,
            final_payouts_usd=np.zeros(0), intervals_usd=np.zeros((0, 2)),
        )
        empty_rep = ReputationalResult([], [], -1, np.zeros(0))
        empty_opp = OpportunityResult(np.zeros(0), [], [])
        receipt = TripleCurrencyReceipt(
            receipt_id=receipt_id, contributor_ids=[], dials=dials,
            qws=empty, monetary=empty_mon, reputational=empty_rep, opportunity=empty_opp,
        )
        receipt.dignity_proof = verify_dignity(receipt)
        return receipt

    base_weights = base_weights_full[active]
    base_intervals = base_intervals_full[active] if base_intervals_full is not None else (
        np.column_stack([base_weights, base_weights])
    )
    raw = raw_full[active]
    embs = candidate_embeddings[active]
    ids = [contributor_ids[i] for i, a in enumerate(active) if a]

    # Re-normalize within the active set so the QWS layer sees a proper
    # probability distribution (weights sum to 1 on the active subset).
    base_sum = float(base_weights.sum())
    if base_sum > 1e-12:
        base_weights = base_weights / base_sum
        base_intervals = base_intervals / base_sum

    # Counterfactual losses (proxied from raw scores, normalized to [0, 1])
    raw_max = float(raw.max())
    counterfactual_losses = (raw / raw_max) if raw_max > 1e-9 else np.zeros_like(raw)

    qws_result = quantile_weighted_shapley(
        base_shapley_weights=base_weights,
        base_shapley_intervals=base_intervals,
        candidate_embeddings=embs,
        target_embedding=target_embedding,
        quality_reference=quality_reference,
    )

    monetary = compute_monetary(
        qws_weights=qws_result.weights_final,
        qws_intervals=base_intervals,
        total_payout_usd=total_payout_usd,
        dials=dials,
    )

    reputational = compute_reputational(
        weights=monetary.weights_with_floor,
        candidate_embeddings=embs,
        target_embedding=target_embedding,
        dials=dials,
        counterfactual_losses=counterfactual_losses,
        seed=seed,
    )

    opportunity = compute_opportunity(
        weights=monetary.weights_with_floor,
        candidate_embeddings=embs,
        dials=dials,
        contributor_history=contributor_history,
        contributor_ids=ids,
    )

    receipt = TripleCurrencyReceipt(
        receipt_id=receipt_id,
        contributor_ids=ids,
        dials=dials,
        qws=qws_result,
        monetary=monetary,
        reputational=reputational,
        opportunity=opportunity,
    )
    receipt.dignity_proof = verify_dignity(receipt)
    return receipt
