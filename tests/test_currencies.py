"""
Smoke test for the QWS + three-currency layer (cascade_v.currencies).

Pure-numpy module — no torch/z3 stubs required, but we install them anyway
so this test can run alongside the rest of the math-only suite.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Minimal torch stub so any transitive imports survive
_torch = types.ModuleType("torch")
_torch_backends = types.ModuleType("torch.backends")
_torch_mps = types.ModuleType("torch.backends.mps")
_torch_mps.is_available = lambda: False
_torch_cuda = types.ModuleType("torch.cuda")
_torch_cuda.is_available = lambda: False


class _Device:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"device('{self.name}')"


_torch.device = _Device
_torch.backends = _torch_backends
_torch_backends.mps = _torch_mps
_torch.cuda = _torch_cuda
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.backends", _torch_backends)
sys.modules.setdefault("torch.backends.mps", _torch_mps)
sys.modules.setdefault("torch.cuda", _torch_cuda)


from cascade_v.currencies import (  # noqa: E402
    CurrencyDials,
    apply_rawlsian_floor,
    assign_lottery_feature,
    assign_roles,
    compute_monetary,
    compute_opportunity,
    compute_reputational,
    compute_triple_currency_receipt,
    estimate_output_quality,
    quantile_weighted_shapley,
    verify_dignity,
)


def _make_unit_vec(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _make_normalized_embeddings(rng, n, d):
    return np.stack([_make_unit_vec(rng, d) for _ in range(n)])


# ---------------------------------------------------------------------------
# Test 1: Dial validation
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 1: Dial validation")
print("=" * 70)
ok = CurrencyDials(0.5, 0.5, 0.5)
ok.validate()
try:
    CurrencyDials(1.5, 0.5, 0.5).validate()
    raise AssertionError("expected validation failure")
except ValueError as e:
    print(f"  validation correctly rejected α=1.5: {e}")
print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: Rawlsian floor — invariants
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 2: Rawlsian floor invariants")
print("=" * 70)

cases = [
    np.array([0.7, 0.2, 0.05, 0.05]),
    np.array([0.0, 0.0, 0.0, 1.0]),
    np.array([0.1, 0.1, 0.1, 0.1, 0.6]),
    np.array([0.25] * 4),  # already uniform
]
for w in cases:
    for alpha in (1.0, 0.85, 0.5, 0.0):
        adj, floor = apply_rawlsian_floor(w, alpha)
        assert abs(adj.sum() - 1.0) < 1e-6, f"sum drift: {adj.sum()}"
        assert (adj >= floor - 1e-9).all(), f"floor violated: {adj} < {floor}"
        if alpha >= 1.0 - 1e-6:
            assert np.allclose(adj, w), "alpha=1 must be a no-op"
        print(f"  α={alpha:.2f}, floor={floor:.4f}, weights={adj.round(3)}  OK")
print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: QWS — sharpening at high q, flattening at low q
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 3: QWS quantile shaping")
print("=" * 70)
rng = np.random.default_rng(0)
n = 6
base = np.array([0.4, 0.25, 0.15, 0.10, 0.07, 0.03])
embs = _make_normalized_embeddings(rng, n, 32)
target = _make_unit_vec(rng, 32)

qws = quantile_weighted_shapley(
    base, np.column_stack([base, base]), embs, target,
    quantiles=(0.10, 0.50, 0.90, 0.99),
)
w99 = qws.weights_by_quantile[0.99]
w50 = qws.weights_by_quantile[0.50]
w10 = qws.weights_by_quantile[0.10]
# At q=0.10 (low quality output), weights should be flatter than base
flat_metric_low = float(w10.std())
flat_metric_base = float(base.std())
flat_metric_high = float(w99.std())
print(f"  std(w@q=0.10)={flat_metric_low:.4f}")
print(f"  std(base)    ={flat_metric_base:.4f}")
print(f"  std(w@q=0.99)={flat_metric_high:.4f}")
assert flat_metric_low < flat_metric_base, "q=0.1 should flatten weights"
assert flat_metric_high >= flat_metric_base * 0.95, "q=0.99 should not flatten"
for q, w in qws.weights_by_quantile.items():
    assert abs(w.sum() - 1.0) < 1e-6, f"weights at q={q} don't sum to 1"
assert qws.quantile_active in (0.10, 0.50, 0.90, 0.99)
print("  PASS")


# ---------------------------------------------------------------------------
# Test 4: Quality estimator stays in [0, 1]
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 4: Quality estimator")
print("=" * 70)
catalog_embs = _make_normalized_embeddings(rng, 80, 32)
for _ in range(20):
    t = _make_unit_vec(rng, 32)
    q = estimate_output_quality(t, catalog_embs)
    assert 0.0 <= q <= 1.0, f"quality out of range: {q}"
ref = rng.uniform(0, 2, size=200)
q_emp = estimate_output_quality(_make_unit_vec(rng, 32), catalog_embs, quality_reference=ref)
assert 0.0 <= q_emp <= 1.0
print(f"  range OK; empirical-CDF branch produced q={q_emp:.3f}")
print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: Roles + lottery feature + dignity proof
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 5: Reputational currency")
print("=" * 70)
weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
embs5 = _make_normalized_embeddings(rng, 5, 32)
tgt5 = _make_unit_vec(rng, 32)
counterfactuals = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
roles = assign_roles(weights, embs5, tgt5, counterfactual_losses=counterfactuals)
for i, r in enumerate(roles):
    print(f"  contributor {i}: {r}")
assert all(len(r) > 0 for r in roles), "every weighted contributor must have ≥1 role"
assert "ESSENTIAL CONTRIBUTOR" in roles[0], "high-counterfactual contributor missing badge"
# After Option 1 cleanup: only ESSENTIAL CONTRIBUTOR + CONTRIBUTING VOICE
# should appear; the dimension-based symbolic labels were removed.
allowed = {"ESSENTIAL CONTRIBUTOR", "CONTRIBUTING VOICE"}
unexpected = {r for tags in roles for r in tags} - allowed
assert not unexpected, f"unexpected role tags leaked through: {unexpected}"

idx, probs = assign_lottery_feature(weights, beta=0.6, seed=42)
assert 0 <= idx < 5
assert abs(probs.sum() - 1.0) < 1e-6
print(f"  featured idx={idx}, probs={probs.round(3)}")
print("  PASS")


# ---------------------------------------------------------------------------
# Test 6: Opportunity adjustment is non-negative
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 6: Opportunity currency")
print("=" * 70)
opp = compute_opportunity(weights, embs5, CurrencyDials(0.85, 0.6, 0.30))
assert (opp.priority_adjustment >= -1e-9).all(), "opportunity must be non-negative"
print(f"  adjustments={opp.priority_adjustment.round(4)}")

opp2 = compute_opportunity(
    weights, embs5, CurrencyDials(0.85, 0.6, 1.0),
    contributor_history={"a": 0, "b": 5, "c": 0, "d": 5, "e": 0},
    contributor_ids=["a", "b", "c", "d", "e"],
)
assert opp2.under_represented_protected == [True, False, True, False, True]
print(f"  under-rep protection (γ=1.0): {opp2.under_represented_protected}")
print("  PASS")


# ---------------------------------------------------------------------------
# Test 7: Composite receipt + dignity proof end-to-end
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 7: Triple-currency receipt end-to-end")
print("=" * 70)


# Stub a CascadeVResult-like object (we only use .attribution.weights/intervals/raw_scores
# and .triage.top_k_ids — the currency layer doesn't touch the proof or grouping).
class _StubAttr:
    def __init__(self, weights, raw, intervals):
        self.weights = weights
        self.raw_scores = raw
        self.intervals = intervals
        self.source_ids = [f"src_{i:03d}" for i in range(len(weights))]
        self.metadata = {}


class _StubTriage:
    def __init__(self, n):
        self.top_k_indices = np.arange(n)
        self.top_k_ids = [f"src_{i:03d}" for i in range(n)]


class _StubResult:
    def __init__(self, weights, raw, intervals):
        self.attribution = _StubAttr(weights, raw, intervals)
        self.triage = _StubTriage(len(weights))
        self.target_id = "test_currency_001"


# 5 active contributors + 3 sparsified-to-zero (active_only must filter these)
weights7 = np.array([0.40, 0.25, 0.15, 0.12, 0.08, 0.0, 0.0, 0.0])
raw7 = weights7.copy()
intervals7 = np.column_stack([weights7 * 0.95, weights7 * 1.05])
embs7 = _make_normalized_embeddings(rng, 8, 32)
tgt7 = _make_unit_vec(rng, 32)

result_stub = _StubResult(weights7, raw7, intervals7)
receipt = compute_triple_currency_receipt(
    receipt_id="test_currency_001",
    cascade_v_result=result_stub,
    candidate_embeddings=embs7,
    target_embedding=tgt7,
    contributor_ids=[f"src_{i:03d}" for i in range(8)],
    total_payout_usd=1.0,
    dials=CurrencyDials(0.85, 0.6, 0.30),
)

assert len(receipt.contributor_ids) == 5, (
    f"active_only must filter to 5 contributors, got {len(receipt.contributor_ids)}"
)
assert abs(receipt.monetary.weights_with_floor.sum() - 1.0) < 1e-6
assert (receipt.monetary.final_payouts_usd >= 0).all()
proof = verify_dignity(receipt)
print(f"  dignity_proof: {proof}")
for k, v in proof.items():
    assert v == "PROVEN", f"{k} = {v}"
rd = receipt.to_dict()
assert rd["dials"]["alpha"] == 0.85
assert len(rd["per_contributor"]) == 5
print(f"  receipt has {len(rd['per_contributor'])} active contributors (filtered from 8)")
print(f"  qws active quantile: {rd['qws']['quantile_active']}")
print("  PASS")


# ---------------------------------------------------------------------------
# Test 8: alpha=1.0 leaves QWS weights untouched (no floor)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 8: α=1.0 ⇒ no floor (strict QWS-Shapley)")
print("=" * 70)
mon = compute_monetary(
    qws_weights=np.array([0.7, 0.2, 0.07, 0.03]),
    qws_intervals=np.column_stack([
        [0.65, 0.18, 0.05, 0.02], [0.75, 0.22, 0.09, 0.04],
    ]),
    total_payout_usd=10.0,
    dials=CurrencyDials(1.0, 0.6, 0.3),
)
assert mon.floor_applied == 0.0
assert np.allclose(mon.weights_with_floor, mon.weights_qws)
print(f"  payouts (α=1.0): {mon.final_payouts_usd}")
print("  PASS")


# ---------------------------------------------------------------------------
print("=" * 70)
print("ALL CURRENCY TESTS PASS")
print("=" * 70)
