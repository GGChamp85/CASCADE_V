"""
embeddings.py — Compute and cache embeddings for catalog and outputs.

Wraps the trained encoder for batch inference. Provides utilities for:
- Embedding the full catalog once and caching to disk
- Embedding a single output (audio) at attribution time
- Computing similarity matrices and top-K retrieval
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cascade_v.config import (
    CATALOG_EMBEDDINGS_PATH,
    DEVICE,
    SAMPLE_RATE,
)
from cascade_v.encoder import AudioEncoder
from cascade_v.utils.audio import audio_to_melspec, batch_audio_to_melspec, fix_length, load_wav


# ---------------------------------------------------------------------------
# Single embedding
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_audio(
    audio: np.ndarray,
    encoder,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """Encode a single audio array. Returns L2-normalized numpy embedding.

    Routes to CLAP's preprocessor if encoder.is_clap is True; otherwise
    uses the custom mel-spectrogram path.
    """
    if getattr(encoder, "is_clap", False):
        return encoder.encode_audio_np(audio, sample_rate=SAMPLE_RATE, device=device)
    encoder.eval()
    audio = fix_length(audio)
    mel = audio_to_melspec(audio).unsqueeze(0).to(device)  # (1, n_mels, T)
    emb = encoder(mel)
    return emb.cpu().numpy()[0]


@torch.no_grad()
def embed_audios(
    audios: list[np.ndarray],
    encoder,
    device: torch.device = DEVICE,
    batch_size: int = 16,
) -> np.ndarray:
    """Encode a list of audio arrays in batches. Returns (N, D) embeddings."""
    if getattr(encoder, "is_clap", False):
        return encoder.encode_audio_np(
            audios, sample_rate=SAMPLE_RATE, device=device, batch_size=batch_size,
        )
    encoder.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(audios), batch_size):
        batch = audios[i:i + batch_size]
        mels = batch_audio_to_melspec(batch).to(device)
        embs = encoder(mels)
        out.append(embs.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Catalog embedding: build and persist
# ---------------------------------------------------------------------------

def build_catalog_embeddings(
    source_paths: list[Path],
    encoder: AudioEncoder,
    out_path: Path = CATALOG_EMBEDDINGS_PATH,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute embeddings for every source in the catalog and save to disk.
    Returns the (N, D) embedding matrix.
    """
    if verbose:
        print(f"[embed] computing embeddings for {len(source_paths)} sources...")
    audios = [load_wav(p) for p in source_paths]
    embeddings = embed_audios(audios, encoder, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    if verbose:
        print(f"[embed] saved embeddings shape={embeddings.shape} to {out_path}")
    return embeddings


def load_catalog_embeddings(path: Path = CATALOG_EMBEDDINGS_PATH) -> np.ndarray:
    """Load cached catalog embeddings."""
    if not path.exists():
        raise FileNotFoundError(
            f"No catalog embeddings at {path}. Run scripts/build_catalog.py first."
        )
    return np.load(path)


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of L2-normalized vectors.

    a: (N, D), b: (M, D) -> (N, M)
    """
    return a @ b.T


def top_k_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-K largest values, sorted descending."""
    if k >= len(similarities):
        return np.argsort(-similarities)
    # argpartition is O(n), then sort just the top-k
    top_k = np.argpartition(-similarities, k)[:k]
    return top_k[np.argsort(-similarities[top_k])]


# ---------------------------------------------------------------------------
# BPM and key estimation
# ---------------------------------------------------------------------------
# The actual implementation lives in cascade_v.utils.audio_meta (no torch
# dependency); we re-export here for backwards compatibility with callers
# that already import from cascade_v.embeddings.
from cascade_v.utils.audio_meta import (  # noqa: E402, F401
    estimate_bpm as _estimate_bpm_impl,
    estimate_key as _estimate_key_impl,
    keys_compatible,
)


def estimate_bpm(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return _estimate_bpm_impl(audio, sample_rate)


def estimate_key(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    return _estimate_key_impl(audio, sample_rate)
