"""Render an SMT-LIB proof as a human-readable, annotated proof.

The annotated form preserves all numeric content but adds:
  - creator name + source ID per `w_i` declaration
  - plain-English headers per axiom group
  - cluster-similarity context for symmetry pairs
  - the actual numbers (in percentages) so a non-SMT reader can audit it

Output is plain text so it renders in a <pre> block alongside the raw SMT-LIB.
"""

from __future__ import annotations

import re
from typing import Any


def annotate_proof(receipt: dict, smt_text: str) -> str:
    sources = receipt.get("per_source", [])
    verification = receipt.get("verification", {})
    axioms = verification.get("axioms", [])
    overall = verification.get("overall_status", "?")
    lat = receipt.get("latency_ms", {})

    def src(i: int) -> dict:
        if 0 <= i < len(sources):
            return sources[i]
        return {"source_id": f"?{i}", "creator_name": "?", "weight_point": 0.0}

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("CASCADE-V — Interpretable Fairness Proof")
    lines.append("=" * 78)
    lines.append(f"Receipt:     {receipt.get('receipt_id', '?')}")
    lines.append(f"Issued:      {receipt.get('created_at_utc', '?')}")
    lines.append(f"Method:      {receipt.get('method', '?')} v{receipt.get('version', '?')}")
    lines.append(f"Verdict:     {overall}")
    if lat.get("total") is not None:
        lines.append(f"Wall time:   {lat['total']:.1f} ms")
    lines.append("")

    lines.append("─" * 78)
    lines.append("VARIABLES — one per triaged source")
    lines.append("─" * 78)
    lines.append("Each w_i is the proven payout share for one source.")
    lines.append("All values are checked simultaneously by the SMT solver.")
    lines.append("")
    for i, s in enumerate(sources):
        w = s.get("weight_point", 0.0) * 100
        lo = s.get("weight_lower", 0.0) * 100
        hi = s.get("weight_upper", 0.0) * 100
        lines.append(
            f"  w_{i:<2} = {s.get('source_id', '?'):<10} · "
            f"{s.get('creator_name', '?'):<18} · "
            f"share = {w:6.2f}%  [{lo:5.2f}, {hi:5.2f}]"
        )
    lines.append("")

    # --- Efficiency ---
    eff = next((a for a in axioms if a.get("name") == "efficiency"), None)
    lines.append("─" * 78)
    lines.append(f"AXIOM 1 — EFFICIENCY    {eff['status'] if eff else '?'}")
    lines.append("─" * 78)
    lines.append("Plain English:  every dollar of payout is accounted for —")
    lines.append("                Σ wᵢ must equal 1.0 (within tolerance 0.001).")
    lines.append("")
    actual_sum = sum(s.get("weight_point", 0.0) for s in sources)
    lines.append(f"Computed sum:   Σ wᵢ = {actual_sum:.6f}")
    lines.append(f"Target:         1.000000   (Δ = {actual_sum - 1.0:+.6f})")
    if eff and eff.get("detail"):
        lines.append(f"Z3 verdict:     {eff['detail']}")
    lines.append("")

    # --- Symmetry ---
    sym = next((a for a in axioms if a.get("name") == "symmetry"), None)
    lines.append("─" * 78)
    lines.append(f"AXIOM 2 — SYMMETRY      {sym['status'] if sym else '?'}")
    lines.append("─" * 78)
    lines.append("Plain English:  if two sources are near-identical (cosine ≥ 0.99 in")
    lines.append("                embedding space), they must receive the same payout.")
    lines.append("                Pairs are extracted from the SMT proof:")
    lines.append("")

    sym_pairs = _extract_symmetry_pairs(smt_text)
    if not sym_pairs:
        lines.append("  (no near-identical pairs found in this attribution)")
    else:
        for i, j in sym_pairs:
            si, sj = src(i), src(j)
            wi = si.get("weight_point", 0.0) * 100
            wj = sj.get("weight_point", 0.0) * 100
            same_creator = (
                si.get("creator_id", "i") == sj.get("creator_id", "j")
            )
            tag = "(same creator)" if same_creator else "(different creators)"
            lines.append(
                f"  w_{i:<2} ≈ w_{j:<2}   "
                f"{si.get('source_id', '?')} ↔ {sj.get('source_id', '?')}   "
                f"{si.get('creator_name', '?')} ↔ {sj.get('creator_name', '?')} {tag}"
            )
            lines.append(
                f"           |Δ| = |{wi:.2f}% − {wj:.2f}%| = "
                f"{abs(wi - wj):.3f}%   (must be ≤ 0.1%)"
            )
    if sym and sym.get("detail"):
        lines.append("")
        lines.append(f"Z3 verdict:     {sym['detail']}")
    lines.append("")

    # --- Null player (formerly "dummy") ---
    null_ax = next((a for a in axioms if a.get("name") == "dummy"), None)
    lines.append("─" * 78)
    lines.append(f"AXIOM 3 — NULL PLAYER   {null_ax['status'] if null_ax else '?'}")
    lines.append("─" * 78)
    lines.append("Plain English:  sources whose marginal contribution is essentially")
    lines.append("                zero (raw Shapley value < 0.001) must receive 0%.")
    lines.append("                No credit for free riders.")
    lines.append("")
    null_indices = _extract_dummy_indices(smt_text)
    if not null_indices:
        lines.append("  (no null sources in this attribution)")
    else:
        lines.append("Null sources flagged by the proof:")
        for i in null_indices:
            s = src(i)
            w = s.get("weight_point", 0.0) * 100
            lines.append(
                f"  w_{i:<2}    {s.get('source_id', '?'):<10} · "
                f"{s.get('creator_name', '?'):<18} · "
                f"share = {w:6.3f}%   (must be ≤ 0.1%)"
            )
    if null_ax and null_ax.get("detail"):
        lines.append("")
        lines.append(f"Z3 verdict:     {null_ax['detail']}")
    lines.append("")

    # --- Verifier ---
    lines.append("─" * 78)
    lines.append("HOW TO VERIFY THIS LOCALLY")
    lines.append("─" * 78)
    lines.append("  $ brew install z3        # one-time")
    lines.append(f"  $ z3 {receipt.get('receipt_id', '?')}.smt2")
    lines.append("  sat                      # ← Z3 confirms the constraints hold")
    lines.append("")
    lines.append(f"SMT-LIB hash (sha256): {verification.get('smt_lib_hash', '?')}")
    lines.append("Anyone can re-hash the .smt2 file to confirm they have the same")
    lines.append("proof; anyone can run z3 on it to confirm the math.")
    lines.append("")
    return "\n".join(lines)


_SYM_RE = re.compile(r"\(<=\s*\(-\s*w_(\d+)\s+w_(\d+)\)\s*[\d.]+\)")


def _extract_symmetry_pairs(smt_text: str) -> list[tuple[int, int]]:
    """Pull (i, j) pairs from the symmetry-section of the SMT text.

    We avoid the matching `>= -tol` line by deduping pairs.
    """
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    in_section = False
    for line in smt_text.splitlines():
        if "; --- symmetry" in line:
            in_section = True
            continue
        if line.startswith("; ---") and in_section:
            break
        if not in_section:
            continue
        m = _SYM_RE.search(line)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs.append((i, j))
    return pairs


_DUM_RE = re.compile(r"\(<=\s*w_(\d+)\s+[\d.]+\)")


def _extract_dummy_indices(smt_text: str) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    in_section = False
    for line in smt_text.splitlines():
        if "; --- dummy" in line:
            in_section = True
            continue
        if line.startswith("; ---") and in_section:
            break
        if not in_section:
            continue
        m = _DUM_RE.search(line)
        if m:
            idx = int(m.group(1))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices
