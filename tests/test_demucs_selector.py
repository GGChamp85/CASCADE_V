"""
Smoke test for the catalog-driven Demucs model selector.
Pure-python — no torch/demucs required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cascade_v.stages.demucs_selector import (  # noqa: E402
    MODEL_REGISTRY,
    coverage_score,
    select_demucs_model,
)


print("=" * 70)
print("Test 1: Drum-and-bass-heavy catalog → 4-stem htdemucs is enough")
print("=" * 70)
cats = ["kick", "snare", "hat", "bass", "sub_bass", "vocal_chop"]
chosen, decision = select_demucs_model(cats)
print(f"  chosen={chosen}  coverage={decision['coverage']:.2%}")
print(f"  routing: {decision['routing']}")
assert decision["coverage"] == 1.0
# Ties broken by smaller model: 4-stem htdemucs preferred over 6-stem.
assert chosen == "htdemucs", f"expected htdemucs (smaller), got {chosen}"
print("  PASS")


print("=" * 70)
print("Test 2: Mixed catalog with leads/pads → 6-stem htdemucs_6s wins")
print("=" * 70)
cats = [
    "kick", "snare", "hat", "bass", "sub_bass",
    "lead", "pluck", "arp",        # only covered by 6-stem
    "pad", "ambient",              # only covered by 6-stem
    "vocal_chop", "fx",
]
chosen, decision = select_demucs_model(cats)
print(f"  chosen={chosen}  coverage={decision['coverage']:.2%}")
for c in decision["considered"]:
    print(f"    {c['model']}: {c['coverage']:.2%} ({c['n_covered']}/{decision['n_total_categories']}) · {c['n_stems']} stems")
assert chosen == "htdemucs_6s", f"expected htdemucs_6s, got {chosen}"
# 11 of 12 categories covered (fx → other still)
assert decision["coverage"] >= 11/12 - 1e-9
print("  PASS")


print("=" * 70)
print("Test 3: Empty catalog → fallback to smallest")
print("=" * 70)
chosen, decision = select_demucs_model([])
print(f"  chosen={chosen}  decision={decision}")
assert chosen == "htdemucs"
assert decision.get("fallback") is True
print("  PASS")


print("=" * 70)
print("Test 4: Coverage scores are monotone in known mappings")
print("=" * 70)
# Drum categories must be covered by every model in the registry
for name in MODEL_REGISTRY:
    cov, n_cov, n_total, _ = coverage_score(["kick", "snare"], name)
    assert cov == 1.0, f"{name} should cover drum categories"
    print(f"  {name} drums coverage: {cov:.0%}")
# A 4-stem model can never cover guitar/piano-only categories
cov4, _, _, _ = coverage_score(["lead", "pad"], "htdemucs")
cov6, _, _, _ = coverage_score(["lead", "pad"], "htdemucs_6s")
assert cov4 == 0.0 and cov6 == 1.0
print(f"  htdemucs lead+pad coverage: {cov4:.0%}")
print(f"  htdemucs_6s lead+pad coverage: {cov6:.0%}")
print("  PASS")


print("=" * 70)
print("ALL DEMUCS SELECTOR TESTS PASS")
print("=" * 70)
