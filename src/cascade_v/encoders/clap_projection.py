"""
clap_projection.py — Frozen CLAP backbone + small trainable MLP projection.

Why this exists:
  Training a 1.4M-param ResCNN from scratch on a 5k catalog with a 2-hour
  compute budget doesn't converge — `align` stays > 0.2 and `mix_cosine` < 0.8
  (see logs/training.json from the 2-epoch run). The encoder couldn't learn
  creator-DNA features in that time.

Solution:
  Use LAION-CLAP (large pretrained audio-text model) as a frozen feature
  extractor. Add a small trainable MLP that projects from CLAP's 512-D
  embedding space into our EMBEDDING_DIM space, fine-tuned with a
  *creator-conditioned* contrastive loss on top. ~150k trainable params,
  trains in <30 min on M-series.

  This is the standard "linear / MLP probe" pattern from foundation-model
  practice (Radford et al. 2021; Park et al. 2023). The projection head
  is what makes the embedding space discriminate by *creator* rather than
  just by category — which is what receipts need.

Interface contract:
  Mirrors AudioEncoder + ClapAudioEncoder so embeddings.embed_audio() routes
  to it transparently:
    - .is_clap = True (so embed_audio routes through encode_audio_np)
    - .encode_audio_np(audio_np_or_list, sample_rate, device, batch_size)
    - .train()/.eval()/.to() — only the projection head is trainable;
      backbone is permanently frozen
    - .save_head(path) / .load_head(path) — checkpoint just the MLP
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade_v.config import DEVICE, EMBEDDING_DIM
from cascade_v.encoders.clap_encoder import (
    CLAP_EMBEDDING_DIM,
    CLAP_SAMPLE_RATE,
    ClapAudioEncoder,
    _resample,
    _to_float32_mono,
)


class ProjectionHead(nn.Module):
    """2-layer MLP with L2-normalized output. ~150k params at 256-D output."""

    def __init__(
        self,
        in_dim: int = CLAP_EMBEDDING_DIM,
        hidden_dim: int = 512,
        out_dim: int = EMBEDDING_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class ClapProjectionEncoder(nn.Module):
    """
    Frozen CLAP backbone + trainable projection head. Looks like an
    AudioEncoder to the rest of the pipeline.
    """

    is_clap = True
    embedding_dim = EMBEDDING_DIM

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ClapAudioEncoder()
        # Backbone is already frozen by ClapAudioEncoder.__init__
        self.head = ProjectionHead(in_dim=CLAP_EMBEDDING_DIM, out_dim=EMBEDDING_DIM)

    def trainable_parameters(self):
        """Only head parameters require grad. Backbone stays frozen."""
        return self.head.parameters()

    def project(self, clap_features: torch.Tensor) -> torch.Tensor:
        return self.head(clap_features)

    @torch.no_grad()
    def _backbone_features(
        self,
        audios: list[np.ndarray],
        sample_rate: int,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Run the frozen CLAP backbone and return the *un-projected* (B, 512)
        feature tensor on `device`. The trainable head consumes these."""
        if sample_rate != CLAP_SAMPLE_RATE:
            audios = [_resample(_to_float32_mono(a), sample_rate, CLAP_SAMPLE_RATE) for a in audios]
        else:
            audios = [_to_float32_mono(a) for a in audios]
        try:
            self.backbone.model.to(device)
        except Exception:
            device = torch.device("cpu")
            self.backbone.model.to(device)

        feats: list[torch.Tensor] = []
        for i in range(0, len(audios), batch_size):
            batch = audios[i : i + batch_size]
            try:
                inputs = self.backbone.processor(
                    audio=batch, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
                )
            except (TypeError, ValueError):
                inputs = self.backbone.processor(
                    audios=batch, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            f = self.backbone.model.get_audio_features(**inputs)
            if hasattr(f, "audio_embeds"):
                f = f.audio_embeds
            elif hasattr(f, "pooler_output"):
                f = f.pooler_output
            elif hasattr(f, "last_hidden_state"):
                f = f.last_hidden_state.mean(dim=1)
            feats.append(f)
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_audio_np(
        self,
        audio: np.ndarray | list[np.ndarray],
        sample_rate: int,
        device: torch.device = DEVICE,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Frozen-backbone forward followed by trainable head. L2-normalized."""
        single = isinstance(audio, np.ndarray) and audio.ndim == 1
        audios = [audio] if single else (list(audio) if not isinstance(audio, list) else audio)

        feats = self._backbone_features(audios, sample_rate, device, batch_size)
        # Move head to the right device (cheap, idempotent)
        self.head.to(feats.device)
        out = self.head(feats)
        out_np = out.cpu().numpy()
        return out_np[0] if single else out_np

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ClapProjectionEncoder bypasses the mel path. "
            "Use encode_audio_np or embeddings.embed_audio."
        )

    # ----- checkpointing only the head (backbone is downloaded from HF cache) ---

    def save_head(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head_state_dict": self.head.state_dict()}, path)

    def load_head(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.head.load_state_dict(ckpt["head_state_dict"])


def load_clap_projection_encoder(checkpoint_path: Path | None = None) -> ClapProjectionEncoder:
    enc = ClapProjectionEncoder()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        enc.load_head(Path(checkpoint_path))
    return enc
