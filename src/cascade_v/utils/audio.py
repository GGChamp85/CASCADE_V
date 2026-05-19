"""
audio.py — Audio I/O and feature extraction.

Loading and saving .wav files (no librosa dependency — soundfile is enough).
Mel-spectrogram extraction via torchaudio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from cascade_v.config import (
    DURATION_SEC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_SAMPLES,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load .wav, return mono float32 at target sample rate."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio), new_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        )
    return audio.astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save mono float32 audio to .wav, with peak normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(audio)) + 1e-9
    audio = (audio / peak * 0.9).astype(np.float32)
    sf.write(str(path), audio, sample_rate)


def fix_length(audio: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Pad or truncate to exact length."""
    if len(audio) < n_samples:
        return np.pad(audio, (0, n_samples - len(audio)))
    return audio[:n_samples]


# ---------------------------------------------------------------------------
# Mel spectrogram extraction (cached transform)
# ---------------------------------------------------------------------------

_mel_transform: T.MelSpectrogram | None = None


def get_mel_transform() -> T.MelSpectrogram:
    """Lazily build and cache the MelSpectrogram transform on CPU."""
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
    return _mel_transform


def audio_to_melspec(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert audio waveform to log-mel spectrogram.

    Returns a tensor of shape (n_mels, n_frames). Always returns CPU float32.
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # (1, n_samples)

    mel = get_mel_transform()(audio)
    log_mel = torch.log1p(mel)
    return log_mel.squeeze(0)  # (n_mels, n_frames)


def batch_audio_to_melspec(audios: list[np.ndarray]) -> torch.Tensor:
    """Batched mel-spec for a list of audio arrays."""
    fixed = [fix_length(a) for a in audios]
    tensor = torch.from_numpy(np.stack(fixed)).float()  # (B, n_samples)
    mel = get_mel_transform()(tensor)
    return torch.log1p(mel)  # (B, n_mels, n_frames)


# ---------------------------------------------------------------------------
# Augmentation (for contrastive training)
# ---------------------------------------------------------------------------

def augment_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Audio augmentations creating positive pairs for contrastive training.
    Each step preserves source identity. Stronger augmentation = encoder
    that's robust to mix-induced distortions at attribution time.

    Pipeline:
      1. Random gain (-4 to +4 dB)
      2. Random time shift (up to 10% of length)
      3. Random polarity flip (50%)
      4. Light additive Gaussian noise
      5. Random short fade-in/fade-out (up to 5% of length, edges)
      6. Random low-pass filter (~70% of nyquist, occasionally) — simulates
         the bandwidth limitation that EQ / mix bus filtering imposes.
      7. (mel-spec time/freq masking is applied separately on the spectrogram
         in audio_to_melspec_aug; it's a torchaudio transform.)
    """
    a = audio.copy()

    # gain
    gain_db = rng.uniform(-4.0, 4.0)
    a = a * (10.0 ** (gain_db / 20.0))

    # time shift
    max_shift = int(0.1 * len(a))
    shift = rng.integers(-max_shift, max_shift + 1)
    if shift != 0:
        a = np.roll(a, shift)

    # polarity flip
    if rng.random() < 0.5:
        a = -a

    # noise
    noise_level = rng.uniform(0.0, 0.006)
    a = a + rng.standard_normal(len(a)).astype(np.float32) * noise_level

    # short edge fades
    fade_n = rng.integers(0, int(0.05 * len(a)))
    if fade_n > 0:
        env = np.linspace(0, 1, fade_n).astype(np.float32)
        if rng.random() < 0.5:
            a[:fade_n] *= env
        else:
            a[-fade_n:] *= env[::-1]

    # occasional low-pass via vectorized FIR (mimics EQ rolloff).
    # Cheap moving-average — wins over a Python-loop IIR by ~50× and gives
    # very similar mild low-pass character for augmentation purposes.
    if rng.random() < 0.35:
        kernel_len = int(rng.integers(3, 12))
        kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
        a = np.convolve(a, kernel, mode="same")

    return np.clip(a, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram-level augmentation (SpecAugment-style)
# ---------------------------------------------------------------------------

def spec_augment(mel: torch.Tensor, time_mask_w: int = 12,
                 freq_mask_w: int = 8, n_masks: int = 2,
                 rng: np.random.Generator | None = None) -> torch.Tensor:
    """SpecAugment: random time + frequency masking on the log-mel.

    Robust to either (n_mels, T) or (B, n_mels, T) tensors. Applied to
    a fresh COPY so the underlying mel isn't mutated.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = mel.clone()
    if out.ndim == 2:
        out = out.unsqueeze(0)
    B, M, T = out.shape
    for b in range(B):
        for _ in range(n_masks):
            # time mask
            tw = int(rng.integers(0, time_mask_w + 1))
            if tw > 0 and T > tw:
                t0 = int(rng.integers(0, T - tw))
                out[b, :, t0:t0 + tw] = 0
            # freq mask
            fw = int(rng.integers(0, freq_mask_w + 1))
            if fw > 0 and M > fw:
                f0 = int(rng.integers(0, M - fw))
                out[b, f0:f0 + fw, :] = 0
    return out.squeeze(0) if mel.ndim == 2 else out
