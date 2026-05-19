"""Determinism: same seed -> identical attribution + identical SMT proof hash."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
def test_pipeline_is_deterministic_under_fixed_seed():
    from cascade_v.pipeline import run_cascade_v
    from cascade_v.utils.determinism import set_global_seeds

    rng = np.random.default_rng(123)
    N, D = 24, 32
    catalog = rng.standard_normal((N, D))
    catalog /= np.linalg.norm(catalog, axis=1, keepdims=True)
    target = catalog[[2, 8, 14]].mean(axis=0)
    target /= np.linalg.norm(target)
    ids = [f"src_{i:03d}" for i in range(N)]

    def run_once():
        set_global_seeds(99)
        with tempfile.TemporaryDirectory() as td:
            r = run_cascade_v(
                target_id="d",
                target_embedding=target,
                catalog_embeddings=catalog,
                catalog_ids=ids,
                triage_k=10,
                proofs_dir=Path(td),
                seed=99,
            )
            return r.attribution.weights.copy(), r.proof.smt_lib_hash

    w1, h1 = run_once()
    w2, h2 = run_once()
    np.testing.assert_allclose(w1, w2, atol=0)
    assert h1 == h2
