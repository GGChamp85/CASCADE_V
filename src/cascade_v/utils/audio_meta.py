"""
audio_meta.py — BPM / key estimation + key-compatibility logic.

Lives in utils (not embeddings.py) because it has no torch dependency, so
modules that just need `keys_compatible` (e.g. stage1_triage.py) can import
it without pulling in the encoder.

`estimate_bpm` and `estimate_key` use librosa lazily — they're only called
when the catalog metadata is missing pre-computed values, so we don't pay
the librosa import cost on every pipeline run.
"""

from __future__ import annotations

import numpy as np


# Krumhansl-Schmuckler major / minor chroma profiles (1990).
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_bpm(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate BPM with librosa's beat tracker. Returns 0.0 on failure."""
    try:
        import librosa  # type: ignore[import-not-found]

        tempo, _ = librosa.beat.beat_track(y=audio.astype(np.float32), sr=sample_rate)
        return float(np.atleast_1d(tempo)[0])
    except Exception:
        return 0.0


def estimate_key(audio: np.ndarray, sample_rate: int) -> str:
    """
    Estimate musical key from chroma + KS profiles. Returns a string like
    "C major" / "A minor", or "" on failure.
    """
    try:
        import librosa  # type: ignore[import-not-found]

        chroma = librosa.feature.chroma_cqt(
            y=audio.astype(np.float32), sr=sample_rate
        )
        chroma_mean = chroma.mean(axis=1)
        if chroma_mean.sum() < 1e-9:
            return ""
        chroma_mean = chroma_mean / chroma_mean.sum()

        major_corrs = [
            float(np.corrcoef(np.roll(_KS_MAJOR, i), chroma_mean)[0, 1])
            for i in range(12)
        ]
        minor_corrs = [
            float(np.corrcoef(np.roll(_KS_MINOR, i), chroma_mean)[0, 1])
            for i in range(12)
        ]
        best_major = int(np.argmax(major_corrs))
        best_minor = int(np.argmax(minor_corrs))
        if major_corrs[best_major] >= minor_corrs[best_minor]:
            return f"{_PITCH_NAMES[best_major]} major"
        return f"{_PITCH_NAMES[best_minor]} minor"
    except Exception:
        return ""


def keys_compatible(key_a: str, key_b: str) -> bool:
    """
    Compatible if equal, relative major/minor, or parallel major/minor.
    Empty strings ⇒ "no info" ⇒ compatible (don't drop).
    """
    if not key_a or not key_b:
        return True
    if key_a == key_b:
        return True
    parts_a = key_a.split()
    parts_b = key_b.split()
    if len(parts_a) != 2 or len(parts_b) != 2:
        return False
    pitch_a, mode_a = parts_a
    pitch_b, mode_b = parts_b
    if pitch_a not in _PITCH_NAMES or pitch_b not in _PITCH_NAMES:
        return False
    semitone_a = _PITCH_NAMES.index(pitch_a)
    semitone_b = _PITCH_NAMES.index(pitch_b)
    delta = (semitone_a - semitone_b) % 12
    # Relative major/minor: minor's tonic is 3 semitones below the relative major
    if mode_a == "major" and mode_b == "minor" and delta == 3:
        return True
    if mode_a == "minor" and mode_b == "major" and delta == 9:
        return True
    # Parallel major/minor (same tonic, different mode) — common in production
    if delta == 0 and mode_a != mode_b:
        return True
    return False
