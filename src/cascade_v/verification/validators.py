"""
validators.py — Cascade Validation Protocol.

Implements pre- and post-condition validation at each pipeline stage.
This is the CASCADE-V analog of the Cascade Validation Protocol described
in US 12,536,365 B1: pre-emptively validate the dependency chain before
committing to outputs, with topologically-ordered checks at every stage
boundary.

Each stage has a set of invariants. If any invariant fails, the validator
raises a structured `ValidationError` with diagnostics — no payout commits
with broken math.
"""

from __future__ import annotations

from dataclasses import dataclass


class ValidationError(Exception):
    """Raised when a pipeline stage produces output violating its invariants."""
    def __init__(self, stage: str, failed_invariants: dict[str, bool], context: dict | None = None):
        self.stage = stage
        self.failed_invariants = failed_invariants
        self.context = context or {}
        message = (
            f"[{stage}] validation failed: "
            + ", ".join(name for name, passed in failed_invariants.items() if not passed)
        )
        super().__init__(message)


@dataclass
class StageValidation:
    stage: str
    invariants: dict[str, bool]
    passed: bool
    context: dict | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "invariants": self.invariants,
            "passed": self.passed,
            "context": self.context or {},
        }


def assert_invariants(
    stage: str,
    invariants: dict[str, bool],
    context: dict | None = None,
    raise_on_fail: bool = True,
) -> StageValidation:
    """
    Check a dict of invariants. Optionally raise on failure.

    Returns a StageValidation record (always), and raises ValidationError
    if any invariant failed and raise_on_fail is True.
    """
    failed = {name: passed for name, passed in invariants.items() if not passed}
    passed_all = len(failed) == 0
    record = StageValidation(stage=stage, invariants=invariants, passed=passed_all, context=context)
    if not passed_all and raise_on_fail:
        raise ValidationError(stage=stage, failed_invariants=invariants, context=context)
    return record
