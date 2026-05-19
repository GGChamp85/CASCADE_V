"""Encoder forward pass shape + param count check."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_encoder_forward_and_param_count():
    import torch

    from cascade_v.config import EMBEDDING_DIM, N_MELS
    from cascade_v.encoder import AudioEncoder

    model = AudioEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    # ~1.4M params at 4-block residual config; allow 0.5M..3M
    assert 5e5 < n_params < 3e6, f"unexpected param count: {n_params}"

    # Forward pass on dummy mel
    x = torch.randn(2, N_MELS, 173)
    out = model(x)
    assert out.shape == (2, EMBEDDING_DIM)
    # L2-normalized
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
