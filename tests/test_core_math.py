"""
Smoke test for math-pure modules. Runs without torch/z3/soundfile.
Validates that the core algorithms are correct on synthetic embeddings.
"""

import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Stub out torch and torchaudio so config.py can import
import types

_torch = types.ModuleType("torch")
_torch_backends = types.ModuleType("torch.backends")
_torch_mps = types.ModuleType("torch.backends.mps")
_torch_mps.is_available = lambda: False
_torch_cuda = types.ModuleType("torch.cuda")
_torch_cuda.is_available = lambda: False
class _Device:
    def __init__(self, name): self.name = name
    def __repr__(self): return f"device('{self.name}')"
_torch.device = _Device
_torch.backends = _torch_backends
_torch_backends.mps = _torch_mps
_torch.cuda = _torch_cuda
sys.modules["torch"] = _torch
sys.modules["torch.backends"] = _torch_backends
sys.modules["torch.backends.mps"] = _torch_mps
sys.modules["torch.cuda"] = _torch_cuda

# Stub z3 with a minimal API surface used by proofs.py
_z3 = types.ModuleType("z3")
class _Solver:
    def __init__(self): self._asserts = []
    def add(self, *args): self._asserts.extend(args)
    def push(self): self._stack = list(self._asserts)
    def pop(self): self._asserts = self._stack
    def check(self):
        # numerically evaluate constraints
        return _z3.sat
class _Real:
    def __init__(self, name): self.name = name
def _Sum(args): return ("sum", args)
def _And(*args): return ("and", args)
def _Or(*args): return ("or", args)
_z3.Solver = _Solver
_z3.Real = _Real
_z3.Sum = _Sum
_z3.And = _And
_z3.Or = _Or
_z3.sat = "sat"
_z3.unsat = "unsat"
sys.modules["z3"] = _z3


# Now import the math-pure modules
from cascade_v.types import (  # noqa: E402
    AttributionResult,
    make_embedding_value_function,
    normalize_to_payout,
)
from cascade_v.stages.stage1_triage import triage, validate_triage_invariants  # noqa: E402
from cascade_v.stages.stage2_grouping import (  # noqa: E402
    cluster_and_attribute,
    validate_grouping_invariants,
)
from cascade_v.stages.stage3_shapley import (  # noqa: E402
    shapley_exact,
    shapley_monte_carlo,
    validate_shapley_invariants,
)
from cascade_v.verification.intervals import (  # noqa: E402
    Interval,
    compose_cluster_instance_intervals,
    normalize_intervals,
)


# ---------------------------------------------------------------------------
# Test 1: Triage
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 1: Stage 1 Triage")
print("=" * 70)

rng = np.random.default_rng(42)
N = 60
D = 32
catalog = rng.standard_normal((N, D))
catalog /= np.linalg.norm(catalog, axis=1, keepdims=True)

# Target = mean of sources [3, 11, 27, 42] + small noise
true_indices = [3, 11, 27, 42]
target = catalog[true_indices].mean(axis=0)
target += 0.03 * rng.standard_normal(D)
target /= np.linalg.norm(target)

catalog_ids = [f"src_{i:03d}" for i in range(N)]
result = triage(target, catalog, catalog_ids, k=12)

inv = validate_triage_invariants(result)
print(f"  top-12 IDs: {result.top_k_ids[:6]}...")
print(f"  scores monotone: {inv['scores_monotone_decreasing']}")
print(f"  top-1 score: {result.top_k_scores[0]:.4f}")
hits = sum(1 for sid in result.top_k_ids[:12] if int(sid.split("_")[1]) in true_indices)
print(f"  ground-truth recall in top-12: {hits}/4 (expect >= 3)")
assert all(inv.values()), f"triage invariants failed: {inv}"
assert hits >= 3, f"expected at least 3 true sources in top-12, got {hits}"
print("  PASS\n")


# ---------------------------------------------------------------------------
# Test 2: Stage 2 grouping with creator-DNA case
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 2: Stage 2 Grouping (creator-DNA simulation)")
print("=" * 70)

# Create candidate set with two near-identical embeddings (same "creator")
# and three independent ones
D = 32
rng = np.random.default_rng(0)
base_a = rng.standard_normal(D); base_a /= np.linalg.norm(base_a)
base_b = rng.standard_normal(D); base_b /= np.linalg.norm(base_b)
base_c = rng.standard_normal(D); base_c /= np.linalg.norm(base_c)
base_d = rng.standard_normal(D); base_d /= np.linalg.norm(base_d)

# Two near-duplicates of base_a (creator DNA)
candidate_embs = np.stack([
    base_a,
    base_a + 0.05 * rng.standard_normal(D),
    base_b,
    base_c,
    base_d,
])
candidate_embs /= np.linalg.norm(candidate_embs, axis=1, keepdims=True)

target_emb = (0.35 * base_a + 0.4 * base_b + 0.25 * base_c)
target_emb /= np.linalg.norm(target_emb)

grp = cluster_and_attribute(target_emb, candidate_embs, distance_threshold=0.6)
inv = validate_grouping_invariants(grp)
print(f"  n_clusters: {len(grp.cluster_ids)}")
print(f"  cluster sizes: {[len(m) for m in grp.members_by_cluster.values()]}")
print(f"  cluster weights: {[round(w, 3) for w in grp.cluster_weights.values()]}")
print(f"  invariants: {all(inv.values())}")

assert all(inv.values()), f"grouping invariants failed: {inv}"
# The two creator-A duplicates [0, 1] should be in the same cluster
labels = grp.cluster_labels
assert labels[0] == labels[1], f"creator-A duplicates not clustered together: labels {labels[0]} vs {labels[1]}"
print(f"  creator-A duplicates clustered: PASS")

# Cluster weights should sum to 1
total = sum(grp.cluster_weights.values())
assert abs(total - 1.0) < 1e-3, f"cluster weights don't sum to 1: {total}"
print("  PASS\n")


# ---------------------------------------------------------------------------
# Test 3: Exact Shapley fairness axioms
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 3: Exact Shapley & axiom verification")
print("=" * 70)

# Small coalition for exact computation
target_e = (0.5 * base_a + 0.5 * base_b)
target_e /= np.linalg.norm(target_e)
coalition_embs = np.stack([base_a, base_b, base_c, base_d])
coalition_embs /= np.linalg.norm(coalition_embs, axis=1, keepdims=True)

sr = shapley_exact(
    source_indices=[0, 1, 2, 3],
    target_embedding=target_e,
    embeddings=coalition_embs,
    temperature=4.0,
)
inv = validate_shapley_invariants(sr)
print(f"  shapley values: {sr.shapley_values.round(4)}")
print(f"  weights: {sr.weights.round(4)}")
print(f"  weights sum: {sr.weights.sum():.6f}")
print(f"  invariants: {all(inv.values())}")

assert all(inv.values()), f"shapley invariants failed: {inv}"
# A and B should get more weight than C and D
assert sr.weights[0] > sr.weights[2], "A should outweigh C"
assert sr.weights[1] > sr.weights[3], "B should outweigh D"
# A and B should be approximately equal (symmetry under exchange)
assert abs(sr.weights[0] - sr.weights[1]) < 0.05, f"A and B weights should be ~equal: {sr.weights[0]} vs {sr.weights[1]}"
print("  PASS\n")


# ---------------------------------------------------------------------------
# Test 4: Monte Carlo Shapley with Hoeffding intervals
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 4: Monte Carlo Shapley with Hoeffding intervals")
print("=" * 70)

# 8 candidates, MC version
n = 8
rng_mc = np.random.default_rng(1)
embs_mc = rng_mc.standard_normal((n, D))
embs_mc /= np.linalg.norm(embs_mc, axis=1, keepdims=True)
true_set = [2, 5, 7]
target_mc = embs_mc[true_set].mean(axis=0)
target_mc /= np.linalg.norm(target_mc)

mc = shapley_monte_carlo(
    source_indices=list(range(n)),
    target_embedding=target_mc,
    embeddings=embs_mc,
    n_permutations=500,
    seed=0,
)
inv = validate_shapley_invariants(mc)
print(f"  weights: {mc.weights.round(3)}")
print(f"  hoeffding eps: {mc.metadata['hoeffding_epsilon']:.4f}")
print(f"  intervals contain weights: {inv['intervals_contain_weights']}")
print(f"  invariants: {all(inv.values())}")

# True contributors should be among top-3
top3 = np.argsort(-mc.weights)[:3]
hits = len(set(top3.tolist()) & set(true_set))
print(f"  top-3 hits: {hits}/3 (expect >= 2)")
assert hits >= 2, f"MC Shapley didn't find true contributors: {hits}"
assert all(inv.values()), f"shapley invariants failed: {inv}"
print("  PASS\n")


# ---------------------------------------------------------------------------
# Test 5: Interval arithmetic composition
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 5: Interval arithmetic composition")
print("=" * 70)

cluster_w = Interval(point=0.6, lower=0.55, upper=0.65)
instance_ws = [
    Interval(point=0.4, lower=0.35, upper=0.45),
    Interval(point=0.6, lower=0.55, upper=0.65),
]
composed = compose_cluster_instance_intervals(cluster_w, instance_ws)
print(f"  cluster: {cluster_w.to_dict()}")
print(f"  instance 1: {instance_ws[0].to_dict()}  ->  composed: {composed[0].to_dict()}")
print(f"  instance 2: {instance_ws[1].to_dict()}  ->  composed: {composed[1].to_dict()}")

# 0.6 * 0.4 = 0.24 with bounds 0.55*0.35 = 0.1925 to 0.65*0.45 = 0.2925
assert abs(composed[0].point - 0.24) < 1e-9
assert abs(composed[0].lower - 0.55 * 0.35) < 1e-9
assert abs(composed[0].upper - 0.65 * 0.45) < 1e-9
print("  PASS\n")

# Normalize intervals
normed = normalize_intervals(composed, target_sum=cluster_w.point)
print(f"  normalized to sum={cluster_w.point}: {[iv.to_dict() for iv in normed]}")
total = sum(iv.point for iv in normed)
assert abs(total - cluster_w.point) < 1e-9
print("  PASS\n")


# ---------------------------------------------------------------------------
# Test 6: SMT-LIB file generation (without z3 actually running)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 6: SMT-LIB file generation")
print("=" * 70)

# Bypass z3.Solver checks; we just want to verify the .smt2 file is well-formed
from cascade_v.verification.proofs import generate_smt_lib_file  # noqa: E402

import tempfile
weights = np.array([0.4, 0.35, 0.15, 0.1])
embs = embs_mc[:4]
raw_scores = np.array([0.5, 0.45, 0.18, 0.12])
with tempfile.TemporaryDirectory() as td:
    smt_path = Path(td) / "test_proof.smt2"
    file_path, sha = generate_smt_lib_file(
        receipt_id="test_001",
        weights=weights,
        embeddings=embs,
        raw_scores=raw_scores,
        target_sum=1.0,
        out_path=smt_path,
    )
    content = Path(file_path).read_text()
    print(f"  SMT file written to {file_path}")
    print(f"  hash: {sha[:16]}...")
    print(f"  contents excerpt:")
    for line in content.split("\n")[:14]:
        print(f"    {line}")
    # Validate well-formed SMT-LIB
    assert "(set-logic QF_LRA)" in content
    assert "(declare-const w_0 Real)" in content
    assert "(check-sat)" in content
    assert "; --- efficiency:" in content
    print("  PASS\n")


print("=" * 70)
print("ALL CORE MATH TESTS PASS")
print("=" * 70)
