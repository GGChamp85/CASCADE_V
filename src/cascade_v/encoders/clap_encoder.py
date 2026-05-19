"""
clap_encoder.py — LAION CLAP audio encoder via HuggingFace transformers.

CLAP (Contrastive Language-Audio Pretraining) is a foundation-model audio
encoder trained on 633K audio-text pairs. We use it as a drop-in replacement
for our custom encoder for the production-quality accuracy upgrade.

Reference: Wu et al. 2023, "Large-scale Contrastive Language-Audio Pretraining
with Feature Fusion and Keyword-to-Caption Augmentation."
   https://arxiv.org/abs/2211.06687
   Model: laion/larger_clap_general (HuggingFace)

We expose the same interface as our custom AudioEncoder so embed_audio() etc.
work without changes:
   - .eval(), .to(device)
   - .__call__(mel: tensor) → embedding tensor
   - param_count_for_logging() — diagnostic only

We bypass the mel-spectrogram path because CLAP has its own preprocessor;
to keep `embed_audio(np.ndarray, encoder)` working we add encode_audio_np()
which our embeddings.py will dispatch to when this encoder type is active.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cascade_v.config import DEVICE, MODELS_DIR


CLAP_MODEL_ID = "laion/larger_clap_general"
CLAP_SAMPLE_RATE = 48_000      # CLAP expects 48 kHz mono
CLAP_EMBEDDING_DIM = 512


class ClapAudioEncoder(nn.Module):
    """Wrapper around HuggingFace ClapModel exposing our encoder interface."""

    is_clap = True
    embedding_dim = CLAP_EMBEDDING_DIM

    def __init__(self, model_id: str = CLAP_MODEL_ID, cache_dir: Path | None = None):
        super().__init__()
        from transformers import ClapModel, ClapProcessor  # heavy import — lazy

        cache = str(cache_dir) if cache_dir else str(MODELS_DIR / "hf_cache")
        Path(cache).mkdir(parents=True, exist_ok=True)
        self.processor = ClapProcessor.from_pretrained(model_id, cache_dir=cache)
        self.model = ClapModel.from_pretrained(model_id, cache_dir=cache)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # The default torch nn module interface — but we expect raw audio np arrays
    # to come in via encode_audio_np. embed_audio() in embeddings.py checks for
    # `is_clap` and routes accordingly.
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ClapAudioEncoder bypasses the mel-spec path. "
            "Use encode_audio_np(audio_np) or embeddings.embed_audio(audio, encoder)."
        )

    @torch.no_grad()
    def encode_audio_np(
        self,
        audio: np.ndarray | list[np.ndarray],
        sample_rate: int,
        device: torch.device = DEVICE,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Encode one or more raw-audio arrays. Returns (N, D) L2-normalized."""
        single = isinstance(audio, np.ndarray) and audio.ndim == 1
        if single:
            audios = [audio]
        else:
            audios = list(audio) if not isinstance(audio, list) else audio

        # CLAP expects 48 kHz mono float
        audios = [_to_float32_mono(a) for a in audios]
        # Resample if needed
        if sample_rate != CLAP_SAMPLE_RATE:
            audios = [_resample(a, sample_rate, CLAP_SAMPLE_RATE) for a in audios]

        out_chunks: list[np.ndarray] = []
        # Move model to device once (idempotent)
        try:
            self.model.to(device)
        except Exception:
            device = torch.device("cpu")
            self.model.to(device)

        for i in range(0, len(audios), batch_size):
            batch = audios[i : i + batch_size]
            try:
                inputs = self.processor(
                    audio=batch, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
                )
            except (TypeError, ValueError):
                inputs = self.processor(
                    audios=batch, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = self.model.get_audio_features(**inputs)  # may be (B, 512) or BaseModelOutput
            if hasattr(features, "audio_embeds"):
                features = features.audio_embeds
            elif hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "last_hidden_state"):
                features = features.last_hidden_state.mean(dim=1)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            out_chunks.append(features.cpu().numpy())

        out = np.concatenate(out_chunks, axis=0)
        return out[0] if single else out

    def param_count_for_logging(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interp resample. Good enough for embedding extraction."""
    if src_sr == dst_sr:
        return audio
    n_dst = int(round(len(audio) * dst_sr / src_sr))
    return np.interp(
        np.linspace(0, len(audio), n_dst, endpoint=False),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def load_clap_encoder() -> ClapAudioEncoder:
    return ClapAudioEncoder()
