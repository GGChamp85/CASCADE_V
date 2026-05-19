"""Pipeline smoke test using real torch+z3 (small synthetic catalog)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
def test_pipeline_runs_end_to_end_synthetic():
    from cascade_v.pipeline import run_cascade_v

    rng = np.random.default_rng(7)
    N, D = 30, 32
    n_creators = 6
    catalog = np.zeros((N, D))
    for c in range(n_creators):
        base = rng.standard_normal(D)
        base /= np.linalg.norm(base)
        for k in range(N // n_creators):
            v = base + 0.08 * rng.standard_normal(D)
            catalog[c * (N // n_creators) + k] = v / np.linalg.norm(v)

    target = catalog[[1, 6, 12, 18]].mean(axis=0)
    target /= np.linalg.norm(target)

    with tempfile.TemporaryDirectory() as td:
        result = run_cascade_v(
            target_id="t",
            target_embedding=target,
            catalog_embeddings=catalog,
            catalog_ids=[f"src_{i:03d}" for i in range(N)],
            triage_k=12,
            proofs_dir=Path(td),
            raise_on_validation_failure=True,
        )

    assert result.proof.overall_status in ("PROVEN", "VIOLATED")
    assert "total" in result.latency_ms
    assert all(v.passed for v in result.validations)
    np.testing.assert_allclose(result.attribution.weights.sum(), 1.0, atol=1e-3)


@pytest.mark.integration
def test_pipeline_strict_raises_on_invalid_target():
    """A non-normalized, NaN-containing target should not silently produce a result."""
    from cascade_v.pipeline import run_cascade_v
    from cascade_v.verification.validators import ValidationError

    rng = np.random.default_rng(0)
    catalog = rng.standard_normal((20, 16))
    catalog /= np.linalg.norm(catalog, axis=1, keepdims=True)

    bad_target = np.full(16, np.nan)

    with tempfile.TemporaryDirectory() as td:
        with pytest.raises((ValidationError, ValueError, AssertionError, FloatingPointError)):
            run_cascade_v(
                target_id="bad",
                target_embedding=bad_target,
                catalog_embeddings=catalog,
                catalog_ids=[f"src_{i:03d}" for i in range(20)],
                triage_k=8,
                proofs_dir=Path(td),
                raise_on_validation_failure=True,
            )
