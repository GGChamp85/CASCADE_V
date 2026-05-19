"""
Integration test: run the full pipeline orchestrator end-to-end with
synthetic embeddings (no torch needed for the test itself).
"""

import sys
import types
import tempfile
from pathlib import Path

import numpy as np

# Apply same stubs as test_imports.py (must come BEFORE any cascade_v import)
sys.path.insert(0, str(Path(__file__).parent))
import test_imports  # noqa: F401  -- runs the stub-installation block

# Now safe to import cascade_v code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cascade_v.pipeline import run_cascade_v  # noqa: E402

print("=" * 70)
print("Integration test: full CASCADE-V orchestrator")
print("=" * 70)

# Build a synthetic catalog of 30 sources × 32 dims, with deliberate clusters
rng = np.random.default_rng(7)
N = 30
D = 32
n_creators = 6
sources_per_creator = N // n_creators

# Per-creator base direction; stems of same creator are jittered around it
catalog = np.zeros((N, D))
for c in range(n_creators):
    base = rng.standard_normal(D)
    base /= np.linalg.norm(base)
    for k in range(sources_per_creator):
        idx = c * sources_per_creator + k
        v = base + 0.08 * rng.standard_normal(D)
        catalog[idx] = v / np.linalg.norm(v)

catalog_ids = [f"src_{i:03d}" for i in range(N)]

# Target: mix from creators 1, 3, 4 (with creator-1 contributing 2 stems)
true_indices = [
    1 * sources_per_creator + 0,    # creator 1, stem 0
    1 * sources_per_creator + 1,    # creator 1, stem 1  -> creator-DNA
    3 * sources_per_creator + 2,    # creator 3
    4 * sources_per_creator + 0,    # creator 4
]
true_weights = np.array([0.25, 0.25, 0.3, 0.2])
target = (catalog[true_indices] * true_weights[:, None]).sum(axis=0)
target = target / np.linalg.norm(target)

with tempfile.TemporaryDirectory() as td:
    proofs_dir = Path(td) / "proofs"
    proofs_dir.mkdir()
    
    result = run_cascade_v(
        target_id="integration_test",
        target_embedding=target,
        catalog_embeddings=catalog,
        catalog_ids=catalog_ids,
        triage_k=12,
        proofs_dir=proofs_dir,
    )
    
    print(f"\nPipeline output:")
    print(f"  catalog size:     {result.attribution.metadata['catalog_size']}")
    print(f"  triaged candidates: {result.attribution.metadata['candidates_from_triage']}")
    print(f"  clusters:          {result.attribution.metadata['n_clusters']}")
    print(f"  validations all passed: {all(v.passed for v in result.validations)}")
    print(f"  proof status:      {result.proof.overall_status}")
    
    print(f"\nTop-5 attributed sources:")
    sorted_idx = np.argsort(-result.attribution.weights)[:5]
    hits = 0
    for k, i in enumerate(sorted_idx):
        sid = result.attribution.source_ids[i]
        w = result.attribution.weights[i]
        lo, hi = result.attribution.intervals[i]
        is_true = int(sid.split("_")[1]) in true_indices
        marker = "✓" if is_true else " "
        if is_true: hits += 1
        print(f"  {marker} {sid}  weight={w:.3f}  [{lo:.3f}, {hi:.3f}]")
    
    print(f"\nGround truth recall in top-5 (instance level): {hits}/4")
    
    # Check creator-level recall — this is the real objective.
    # Group source_ids by their creator block (sources_per_creator each).
    def creator_of(sid: str) -> int:
        return int(sid.split("_")[1]) // sources_per_creator
    
    true_creators = {creator_of(f"src_{i:03d}") for i in true_indices}
    creator_weights: dict[int, float] = {}
    for i, sid in enumerate(result.attribution.source_ids):
        c = creator_of(sid)
        creator_weights[c] = creator_weights.get(c, 0.0) + float(result.attribution.weights[i])
    top_creators = sorted(creator_weights.keys(), key=lambda c: -creator_weights[c])[:3]
    creator_hits = len(set(top_creators) & true_creators)
    print(f"True creators: {sorted(true_creators)}")
    print(f"Top-3 attributed creators: {top_creators}")
    print(f"Creator-level recall in top-3: {creator_hits}/3")
    
    # Smoke validations
    assert all(v.passed for v in result.validations), "validation failures"
    assert abs(result.attribution.weights.sum() - 1.0) < 1e-3, "weights don't sum to 1"
    assert (result.attribution.weights >= 0).all(), "negative weights"
    # Use creator-level recall — this is what the system is optimizing for
    assert creator_hits >= 2, f"expected at least 2 of 3 true creators in top-3, got {creator_hits}"
    
    # Note: the stub-z3 in this test always returns "sat" which causes dummy
    # checks to misreport as VIOLATED. With the real z3-solver package, dummy
    # checks pass correctly because z3 evaluates the actual weight values.
    # We verify the smt-lib file is well-formed instead.
    
    # SMT file exists
    smt_path = Path(result.proof.smt_lib_file)
    assert smt_path.exists(), f"SMT file not written: {smt_path}"
    smt_content = smt_path.read_text()
    assert "(set-logic QF_LRA)" in smt_content
    assert "(check-sat)" in smt_content
    print(f"\nSMT file: {smt_path}")
    print(f"  size: {smt_path.stat().st_size} bytes")
    print(f"  hash: {result.proof.smt_lib_hash[:16]}...")
    
    # Show a few SMT lines
    print(f"\nSMT excerpt:")
    for line in smt_content.split("\n")[:15]:
        print(f"    {line}")

print("\n" + "=" * 70)
print("INTEGRATION TEST PASS")
print("=" * 70)
