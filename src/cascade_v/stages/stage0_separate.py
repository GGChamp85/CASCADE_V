"""
stage0_separate.py — Optional Stage 0: source separation via Demucs.

Demucs (Défossez et al., Meta 2019-2023) is the state-of-the-art waveform
music source separator. It splits a mix into 4 stems: drums, bass, other,
vocals. We use it OPTIONALLY before the attribution pipeline:

  raw mix
    │
    ▼   Stage 0 — Demucs separation (optional)
  [drums, bass, other, vocals]
    │
    ▼   Encode each separately + run Stage 1..V on each
  [receipt_drums, receipt_bass, ...]
    │
    ▼   Aggregate — sum per-source weights across stems, renormalize
  unified receipt

Why this helps: when the raw mix has one dominant source masking another,
Stage 1 cosine retrieval finds the dominant source and misses the masked
one. After Demucs separates the mix into 4 channels, each channel has
fewer (often just 1) dominant contributors and triage retrieves the right
candidates much more reliably.

References:
  - Défossez et al. 2019, "Music Source Separation in the Waveform Domain"
  - Défossez 2021, "Hybrid Spectrogram and Waveform Source Separation"
  - https://github.com/facebookresearch/demucs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# Resolved at first call — populated either from the explicit
# CASCADE_DEMUCS_MODEL setting or by inspecting catalog metadata when the
# setting is "auto" (see _resolve_demucs_model below).
_RESOLVED_DEMUCS_MODEL: str | None = None
_DEMUCS_DECISION: dict | None = None


@dataclass
class SeparationResult:
    """Demucs output: per-stem audio + metadata."""
    stem_names: list[str]              # ["drums", "bass", "other", "vocals", ...]
    stems: dict[str, np.ndarray]       # name → audio at sample_rate
    sample_rate: int
    metadata: dict


_demucs_model = None


def _resolve_demucs_model() -> str:
    """
    Decide which Demucs checkpoint to use.

    If CASCADE_DEMUCS_MODEL is "auto", inspect the catalog metadata and
    pick the model with best stem-coverage of the union of catalog
    categories. Otherwise use the explicit value as-is. Cached on the
    module so the decision happens exactly once.
    """
    global _RESOLVED_DEMUCS_MODEL, _DEMUCS_DECISION
    if _RESOLVED_DEMUCS_MODEL is not None:
        return _RESOLVED_DEMUCS_MODEL

    from cascade_v.config import CATALOG_METADATA_PATH, DEMUCS_MODEL

    if DEMUCS_MODEL != "auto":
        _RESOLVED_DEMUCS_MODEL = DEMUCS_MODEL
        _DEMUCS_DECISION = {"chosen": DEMUCS_MODEL, "reason": "explicit setting"}
        return DEMUCS_MODEL

    # Auto: read catalog categories from metadata, pick best fit.
    from cascade_v.stages.demucs_selector import select_demucs_model

    try:
        import json
        with open(CATALOG_METADATA_PATH) as f:
            meta = json.load(f)
        categories = sorted({s.get("category", "") for s in meta.get("sources", [])} - {""})
    except Exception:
        categories = []

    chosen, decision = select_demucs_model(categories)
    _RESOLVED_DEMUCS_MODEL = chosen
    _DEMUCS_DECISION = decision
    return chosen


def get_demucs_decision() -> dict | None:
    """Return the auto-selection decision log (or None until first resolve)."""
    return _DEMUCS_DECISION


def _load_demucs():
    """Lazy-load Demucs to keep startup fast."""
    global _demucs_model
    if _demucs_model is None:
        from demucs.pretrained import get_model

        model_name = _resolve_demucs_model()
        model = get_model(model_name)
        model.eval()
        _demucs_model = model
    return _demucs_model


def separate(
    audio: np.ndarray,
    sample_rate: int = 22050,
    device: str = "cpu",
) -> SeparationResult:
    """Run Demucs separation on a mono audio array.

    Demucs expects 44.1 kHz stereo. We resample + duplicate mono to stereo,
    run separation, then collapse back to mono per stem.
    """
    from demucs.apply import apply_model

    model = _load_demucs()
    target_sr = model.samplerate

    # Resample to target_sr
    if sample_rate != target_sr:
        audio_resampled = _linear_resample(audio, sample_rate, target_sr)
    else:
        audio_resampled = audio

    # Mono → stereo by duplication (Demucs needs stereo)
    if audio_resampled.ndim == 1:
        wav = np.stack([audio_resampled, audio_resampled], axis=0)
    else:
        wav = audio_resampled
    wav_t = torch.from_numpy(wav).float()
    if wav_t.ndim == 2:
        wav_t = wav_t.unsqueeze(0)  # (1, 2, T)

    # Apply model (on CPU is fine for our short clips)
    with torch.no_grad():
        # apply_model returns (B, sources, channels, time)
        out = apply_model(model, wav_t.to(device), device=device, progress=False)
    out = out[0].cpu().numpy()  # (sources, channels, time)

    # Collapse stereo → mono per stem; resample back to source SR
    sources = list(model.sources)
    stems: dict[str, np.ndarray] = {}
    for i, name in enumerate(sources):
        mono = out[i].mean(axis=0)
        if sample_rate != target_sr:
            mono = _linear_resample(mono, target_sr, sample_rate)
        stems[name] = mono.astype(np.float32)

    return SeparationResult(
        stem_names=sources,
        stems=stems,
        sample_rate=sample_rate,
        metadata={
            "model": _resolve_demucs_model(),
            "demucs_sample_rate": int(target_sr),
            "input_samples": int(len(audio)),
            "auto_select_decision": _DEMUCS_DECISION,
        },
    )


def _linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    n_dst = int(round(len(audio) * dst_sr / src_sr))
    return np.interp(
        np.linspace(0, len(audio), n_dst, endpoint=False),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
