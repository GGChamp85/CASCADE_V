"""
encoder.py — Audio encoder neural network.

A 4-block residual CNN that maps log-mel spectrograms to fixed-dimensional
embeddings. Trained with NT-Xent contrastive loss so augmented versions of
the same source land near each other in embedding space.

Each ResBlock is Conv-BN-ReLU-Conv-BN with a 1x1 skip connection, followed
by 2x2 max-pool. Stack 4 blocks (32→64→128→256), adaptive avg-pool, dropout,
linear projection, L2-normalize.

Param count ~1.4M; runs comfortably on Apple MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade_v.config import EMBEDDING_DIM, ENCODER_CHANNELS, N_MELS


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return self.pool(out)


class AudioEncoder(nn.Module):
    """Log-mel (B, n_mels, T) -> embedding (B, D), L2-normalized."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        channels: list[int] | None = None,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = ENCODER_CHANNELS

        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ResBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        x = self.blocks(mel)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=-1)


def nt_xent_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SimCLR NT-Xent loss. Each (a_i, b_i) pair is positive; rest are negatives."""
    batch_size = embeddings_a.shape[0]
    embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)
    similarity = embeddings @ embeddings.T / temperature

    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e9)

    targets = torch.arange(2 * batch_size, device=similarity.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(similarity, targets)
