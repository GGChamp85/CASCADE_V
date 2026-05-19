"""
proofs.py — Z3 SMT proof certificates for attribution fairness axioms.

This is the patent contribution made concrete. For each per-output
attribution result, we construct an SMT-LIB proof that the result
satisfies the four Shapley fairness axioms:

    1. Efficiency:    sum_i phi_i = v(N)
    2. Symmetry:      v(S U {i}) = v(S U {j}) for all S => phi_i = phi_j
    3. Dummy:         v(S U {i}) = v(S) for all S => phi_i = 0
    4. (Linearity is structural; we omit it in per-instance proofs)

The Z3 solver checks these as numerical constraints over the actual
attribution values for the specific coalition. The result is a
serializable SMT-LIB file that any auditor with z3 can independently
verify — that's what makes the payout receipt formally auditable.

Note: For the prototype we encode the axioms over the actual numeric
values (an "instance proof"). For a stronger meta-proof — proving the
algorithm always satisfies the axioms — we'd encode the full Shapley
formula as an SMT theory. The prototype targets per-receipt proofs
because that's what regulators and creators actually need.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import z3

from cascade_v.config import (
    DUMMY_TOLERANCE,
    EFFICIENCY_TOLERANCE,
    SYMMETRY_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Axiom check result
# ---------------------------------------------------------------------------

@dataclass
class AxiomCheck:
    name: str
    status: str                  # "PROVEN", "VIOLATED", "NA"
    detail: str
    smt_constraints: list[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class ProofCertificate:
    receipt_id: str
    axioms: list[AxiomCheck]
    smt_lib_file: str            # path to the .smt2 file
    smt_lib_hash: str            # sha256 of the file content
    overall_status: str          # "PROVEN" if all axioms pass, else "VIOLATED"

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "axioms": [a.to_dict() for a in self.axioms],
            "smt_lib_file": self.smt_lib_file,
            "smt_lib_hash": self.smt_lib_hash,
            "overall_status": self.overall_status,
        }


# ---------------------------------------------------------------------------
# Axiom checks
# ---------------------------------------------------------------------------

def check_efficiency(
    weights: np.ndarray,
    target_sum: float = 1.0,
    tolerance: float = EFFICIENCY_TOLERANCE,
) -> AxiomCheck:
    """
    Efficiency axiom: sum of weights = target (e.g. v(N), or 1.0 for normalized).

    SMT encoding:
        (declare-const w_i Real)  for each i
        (assert (= w_i <value>))   for each i
        (assert (< (abs (- (+ w_0 w_1 ... w_n) target)) tol))
        check-sat
    """
    s = z3.Solver()
    w_vars = [z3.Real(f"w_{i}") for i in range(len(weights))]
    constraints = []
    for i, (var, val) in enumerate(zip(w_vars, weights)):
        c = var == float(val)
        s.add(c)
        constraints.append(f"(= w_{i} {float(val):.10f})")

    sum_expr = z3.Sum(w_vars) if len(w_vars) > 1 else w_vars[0]
    diff = sum_expr - target_sum
    bound = z3.And(diff < tolerance, diff > -tolerance)
    s.add(bound)
    constraints.append(f"(<= (abs (- (+ {' '.join(f'w_{i}' for i in range(len(weights)))}) {target_sum})) {tolerance})")

    actual_sum = float(np.sum(weights))
    if s.check() == z3.sat:
        return AxiomCheck(
            name="efficiency",
            status="PROVEN",
            detail=f"sum(weights) = {actual_sum:.6f} ≈ {target_sum:.6f} (tol={tolerance})",
            smt_constraints=constraints,
        )
    return AxiomCheck(
        name="efficiency",
        status="VIOLATED",
        detail=f"sum(weights) = {actual_sum:.6f} ≠ {target_sum:.6f} (tol={tolerance})",
        smt_constraints=constraints,
    )


def check_symmetry(
    weights: np.ndarray,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.985,
    weight_tolerance: float = SYMMETRY_TOLERANCE,
) -> AxiomCheck:
    """
    Symmetry axiom: equivalent contributors get equal weight.

    "Equivalent" = embedding cosine similarity > similarity_threshold
    (acting as proxy for "equal marginal contribution everywhere").
    Then we check |w_i - w_j| < tolerance for all such pairs.

    SMT encoding: assert all weights, assert pairwise equalities for
    similar pairs, check satisfiability.
    """
    n = len(weights)
    if n < 2:
        return AxiomCheck(
            name="symmetry",
            status="NA",
            detail=f"only {n} source(s); symmetry trivially holds",
            smt_constraints=[],
        )

    # Find equivalent pairs by cosine similarity
    sims = embeddings @ embeddings.T
    equiv_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= similarity_threshold:
                equiv_pairs.append((i, j))

    if not equiv_pairs:
        return AxiomCheck(
            name="symmetry",
            status="NA",
            detail="no equivalent source pairs found (none above similarity threshold)",
            smt_constraints=[],
        )

    s = z3.Solver()
    w_vars = [z3.Real(f"w_{i}") for i in range(n)]
    constraints = []
    for i, val in enumerate(weights):
        s.add(w_vars[i] == float(val))
        constraints.append(f"(= w_{i} {float(val):.10f})")

    violations = []
    for (i, j) in equiv_pairs:
        diff = w_vars[i] - w_vars[j]
        s.push()
        s.add(z3.Or(diff > weight_tolerance, diff < -weight_tolerance))
        if s.check() == z3.sat:
            violations.append((i, j, abs(weights[i] - weights[j])))
        s.pop()
        constraints.append(
            f"(<= (abs (- w_{i} w_{j})) {weight_tolerance}) ; pair similar (cos>{similarity_threshold})"
        )

    if violations:
        return AxiomCheck(
            name="symmetry",
            status="VIOLATED",
            detail=f"{len(violations)} equivalent pair(s) have unequal weights: {violations[:3]}",
            smt_constraints=constraints,
        )
    return AxiomCheck(
        name="symmetry",
        status="PROVEN",
        detail=f"{len(equiv_pairs)} equivalent pair(s) have equal weights (within tol={weight_tolerance})",
        smt_constraints=constraints,
    )


def check_dummy(
    weights: np.ndarray,
    raw_scores: np.ndarray,
    score_threshold: float = 1e-3,
    weight_tolerance: float = DUMMY_TOLERANCE,
) -> AxiomCheck:
    """
    Dummy axiom: zero-marginal-contribution sources get zero weight.

    "Dummy" = raw Shapley value below score_threshold.
    Check that those sources' final weights are also below weight_tolerance.

    SMT encoding: assert all weights, assert dummies have weight near zero.
    """
    dummy_indices = np.where(np.abs(raw_scores) < score_threshold)[0]

    if len(dummy_indices) == 0:
        return AxiomCheck(
            name="dummy",
            status="NA",
            detail="no dummy sources found (all have nonzero marginal contribution)",
            smt_constraints=[],
        )

    s = z3.Solver()
    w_vars = [z3.Real(f"w_{i}") for i in range(len(weights))]
    constraints = []
    for i, val in enumerate(weights):
        s.add(w_vars[i] == float(val))
        constraints.append(f"(= w_{i} {float(val):.10f})")

    violations = []
    for idx in dummy_indices:
        s.push()
        s.add(z3.Or(w_vars[idx] > weight_tolerance, w_vars[idx] < -weight_tolerance))
        if s.check() == z3.sat:
            violations.append((int(idx), float(weights[idx])))
        s.pop()
        constraints.append(f"(<= (abs w_{idx}) {weight_tolerance}) ; dummy")

    if violations:
        return AxiomCheck(
            name="dummy",
            status="VIOLATED",
            detail=f"{len(violations)} dummy source(s) have nonzero weight: {violations[:3]}",
            smt_constraints=constraints,
        )
    return AxiomCheck(
        name="dummy",
        status="PROVEN",
        detail=f"{len(dummy_indices)} dummy source(s) all have zero weight (within tol={weight_tolerance})",
        smt_constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

def generate_smt_lib_file(
    receipt_id: str,
    weights: np.ndarray,
    embeddings: np.ndarray,
    raw_scores: np.ndarray,
    target_sum: float,
    out_path: Path,
) -> tuple[str, str]:
    """
    Write an SMT-LIB file encoding the fairness-axiom constraints over the
    actual attribution values. Returns (file_path, sha256_hash).
    """
    n = len(weights)
    lines: list[str] = []
    lines.append(f"; CASCADE-V attribution fairness proof certificate")
    lines.append(f"; receipt_id: {receipt_id}")
    lines.append(f"; n_sources: {n}")
    lines.append(f"")
    lines.append(f"(set-logic QF_LRA)")
    lines.append(f"")

    # Variable declarations
    for i in range(n):
        lines.append(f"(declare-const w_{i} Real)")
    lines.append(f"")

    # Weight value assertions
    lines.append(f"; --- weight values ---")
    for i, w in enumerate(weights):
        lines.append(f"(assert (= w_{i} {float(w):.10f}))")
    lines.append(f"")

    # Efficiency
    lines.append(f"; --- efficiency: sum(weights) ≈ {target_sum:.6f} ---")
    sum_expr = " ".join(f"w_{i}" for i in range(n))
    if n > 1:
        lines.append(f"(assert (<= (- (+ {sum_expr}) {target_sum:.10f}) {EFFICIENCY_TOLERANCE}))")
        lines.append(f"(assert (>= (- (+ {sum_expr}) {target_sum:.10f}) (- {EFFICIENCY_TOLERANCE})))")
    else:
        lines.append(f"(assert (<= (- w_0 {target_sum:.10f}) {EFFICIENCY_TOLERANCE}))")
        lines.append(f"(assert (>= (- w_0 {target_sum:.10f}) (- {EFFICIENCY_TOLERANCE})))")
    lines.append(f"")

    # Symmetry: pairwise constraints for similar embeddings.
    # Threshold 0.985 (was 0.99): catches near-duplicate same-creator stems
    # that legitimately should share weight but were silently skipped at
    # 0.99. Symmetry violations on close-but-not-identical pairs indicate
    # the upstream split (Stage 2 or Stage 3) is treating them asymmetrically.
    sims = embeddings @ embeddings.T
    sym_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= 0.985:
                sym_pairs.append((i, j))
    if sym_pairs:
        lines.append(f"; --- symmetry: similar sources have equal weights ---")
        for (i, j) in sym_pairs:
            lines.append(f"(assert (<= (- w_{i} w_{j}) {SYMMETRY_TOLERANCE}))")
            lines.append(f"(assert (>= (- w_{i} w_{j}) (- {SYMMETRY_TOLERANCE})))")
        lines.append(f"")

    # Dummy: zero-score sources have zero weight
    dummy_indices = np.where(np.abs(raw_scores) < 1e-3)[0]
    if len(dummy_indices) > 0:
        lines.append(f"; --- dummy: zero-contribution sources have zero weight ---")
        for idx in dummy_indices:
            lines.append(f"(assert (<= w_{idx} {DUMMY_TOLERANCE}))")
            lines.append(f"(assert (>= w_{idx} (- {DUMMY_TOLERANCE})))")
        lines.append(f"")

    lines.append(f"(check-sat)")
    lines.append(f"(get-model)")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return str(out_path), sha


def generate_proof_certificate(
    receipt_id: str,
    weights: np.ndarray,
    embeddings: np.ndarray,
    raw_scores: np.ndarray,
    target_sum: float,
    proofs_dir: Path,
) -> ProofCertificate:
    """
    Run all axiom checks via z3 and write the SMT-LIB file.

    Returns a ProofCertificate with status per axiom.
    """
    axiom_checks = [
        check_efficiency(weights, target_sum=target_sum),
        check_symmetry(weights, embeddings),
        check_dummy(weights, raw_scores),
    ]

    smt_path = proofs_dir / f"{receipt_id}.smt2"
    file_path, sha = generate_smt_lib_file(
        receipt_id, weights, embeddings, raw_scores, target_sum, smt_path,
    )

    statuses = [a.status for a in axiom_checks if a.status != "NA"]
    overall = "PROVEN" if all(s == "PROVEN" for s in statuses) else "VIOLATED"

    return ProofCertificate(
        receipt_id=receipt_id,
        axioms=axiom_checks,
        smt_lib_file=file_path,
        smt_lib_hash=sha,
        overall_status=overall,
    )
