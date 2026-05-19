"""
synth.py — Splice-style synthesizer for the catalog stems.

Each creator has a stylistic signature (preferred frequency band, decay character,
harmonic content, rhythm density, modulation depth) shared across all their
stems. The synthesizers produce realistic-sounding loops similar to what Splice
distributes (drum hits with body+transient+air, layered basses with sub +
saturation, polyphonic chord pads, FM-ish vocal chops, reverberant FX) so the
encoder is exercised with content that better resembles real-world inputs.

Key design choices:
- Drums: filtered noise click + tonal body + envelope follower for realism.
- Bass: sub sine + saturated mid + low-pass-shaped tail.
- Lead/pluck/arp: detuned saw-like additive synthesis with note envelope.
- Pad: multi-voice supersaw with slow LFO + reverberant tail.
- FX/vocals: FM-modulated formant filter + reverb wash.
- Output: peak-normalized to ~-1 dBFS, with a creator-specific noise floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from cascade_v.config import (
    CATALOG_CATEGORIES,
    CATALOG_SIZE,
    DURATION_SEC,
    GLOBAL_SEED,
    N_CREATORS,
    N_SAMPLES,
    SAMPLE_RATE,
)
from cascade_v.utils.audio import save_wav


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class CreatorSignature:
    creator_id: str
    name: str
    base_freq_hz: float          # fundamental band
    detune_cents: float          # detune amount in cents
    decay_factor: float          # envelope decay rate
    harmonic_richness: float     # 0=pure, 1=many harmonics
    noise_floor: float           # background noise level
    rhythm_density: float        # 0=sparse, 1=dense
    swing: float                 # 0..0.4 — drum swing amount
    formant_shift: float         # vocal/FX formant resonance
    transient_sharpness: float   # drum/pluck attack tightness
    reverb_amount: float         # 0..0.6 — wet/dry blend

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceRecord:
    source_id: str
    creator_id: str
    creator_name: str
    category: str
    bpm: int
    key: str
    file_path: str

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Creator generation
# ---------------------------------------------------------------------------

CREATOR_NAMES = [
    "DJ Atlas", "Ms Mavy", "Koretzky", "Producer Noir", "Synthia",
    "Lowend Lex", "Topline Tara", "Beatsmith", "Felix Verde", "Echo Park",
    "Nova Beats", "Riff Forge", "Indigo Wave", "Stella Drum", "Crate Mosaic",
    "Ravi Pulse", "Ada Lattice", "Onyx Channel", "Quasar Quill", "Vega Vault",
    "Kira Kiln", "Jove Jive", "Tundra Tine", "Polaris Hum",
    "Helio Dawn", "Mira Fox", "Cobalt Ray", "Delta Storm", "Aether Bloom",
    "Cinder Vox", "Lyric Dynamo", "Sable Forge",
]

KEYS = ["C", "Cm", "C#m", "D", "Dm", "Em", "F", "F#m", "G", "Gm", "Am", "Bm"]


def make_creators(n: int = N_CREATORS, seed: int = GLOBAL_SEED) -> list[CreatorSignature]:
    rng = np.random.default_rng(seed)
    creators = []
    for i in range(n):
        name = CREATOR_NAMES[i % len(CREATOR_NAMES)]
        if i >= len(CREATOR_NAMES):
            name = f"{name} {i // len(CREATOR_NAMES) + 1}"
        creators.append(CreatorSignature(
            creator_id=f"creator_{i:02d}",
            name=name,
            base_freq_hz=float(rng.uniform(60, 520)),
            detune_cents=float(rng.uniform(0, 35)),
            decay_factor=float(rng.uniform(0.8, 10.0)),
            harmonic_richness=float(rng.uniform(0.15, 0.95)),
            noise_floor=float(rng.uniform(0.0, 0.018)),
            rhythm_density=float(rng.uniform(0.25, 0.95)),
            swing=float(rng.uniform(0.0, 0.35)),
            formant_shift=float(rng.uniform(0.7, 1.6)),
            transient_sharpness=float(rng.uniform(0.5, 1.0)),
            reverb_amount=float(rng.uniform(0.05, 0.55)),
        ))
    return creators


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def _onepole_lp(x: np.ndarray, cutoff_hz: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cheap one-pole low-pass filter (in-place safe)."""
    if cutoff_hz <= 0:
        return x
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / sr
    a = dt / (rc + dt)
    out = np.zeros_like(x)
    out[0] = x[0] * a
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _onepole_hp(x: np.ndarray, cutoff_hz: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return x - _onepole_lp(x, cutoff_hz, sr)


def _short_reverb(x: np.ndarray, amount: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cheap convolutional reverb — exponentially-decaying noise IR (~250 ms)."""
    if amount <= 0:
        return x
    rng = np.random.default_rng(int(amount * 1e6))
    ir_len = int(0.25 * sr)
    ir = rng.standard_normal(ir_len).astype(np.float32) * 0.1
    ir[0] = 1.0
    ir = ir * np.exp(-np.linspace(0, 4, ir_len))
    wet = np.convolve(x, ir, mode="same")
    return ((1 - amount) * x + amount * 0.6 * wet).astype(np.float32)


def _adsr(n: int, attack: float, decay: float, sustain: float, release: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simple ADSR envelope of length n samples."""
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    s = max(1, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(s, sustain),
        np.linspace(sustain, 0, r, endpoint=True),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n].astype(np.float32)


def _supersaw(freq: float, n: int, n_voices: int = 5,
              detune_cents: float = 12.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Detuned multi-voice saw, normalized."""
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    for k in range(n_voices):
        c = (k - (n_voices - 1) / 2) * detune_cents / max(1, n_voices - 1)
        f = freq * 2 ** (c / 1200)
        # Synth saw: sum of sine harmonics (band-limited approximation)
        v = np.zeros(n, dtype=np.float32)
        for h in range(1, 9):
            v += (((-1) ** (h + 1)) / h) * np.sin(2 * np.pi * f * h * t)
        out += v
    out /= n_voices
    return out


def _drum_hit(category: str, length_n: int, creator: CreatorSignature,
              rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize a single realistic-sounding drum hit."""
    t = np.arange(length_n) / sr

    if category == "kick":
        # body sweep + click + low noise body
        f0 = max(40, creator.base_freq_hz * 0.45)
        f_end = f0 * 0.45
        sweep_dur = 0.06
        sweep_n = int(sweep_dur * sr)
        sweep = np.linspace(f0, f_end, sweep_n)
        body = np.sin(2 * np.pi * np.cumsum(sweep) / sr).astype(np.float32)
        body *= np.exp(-np.linspace(0, 6, sweep_n))
        out = np.zeros(length_n, dtype=np.float32)
        out[:sweep_n] = body * 1.1
        # tail body (sustained low sine for chest punch)
        tail_dur = min(0.18, length_n / sr)
        tail_n = int(tail_dur * sr)
        tail_t = np.arange(tail_n) / sr
        tail = np.sin(2 * np.pi * f_end * tail_t)
        tail *= np.exp(-tail_t * (creator.decay_factor + 5))
        out[:tail_n] += 0.35 * tail.astype(np.float32)
        # transient click
        click_n = int(0.005 * sr * (2.0 - creator.transient_sharpness))
        click = rng.standard_normal(click_n).astype(np.float32)
        click *= np.exp(-np.linspace(0, 8, click_n))
        out[:click_n] += 0.4 * click * creator.transient_sharpness
        out = _onepole_lp(out, 220 + 80 * creator.harmonic_richness, sr)
        return out

    if category == "snare":
        # tonal body (~190 Hz) + filtered noise burst, transient click
        body_freq = creator.base_freq_hz * 1.3 + 180
        body = np.sin(2 * np.pi * body_freq * t)
        body += 0.5 * np.sin(2 * np.pi * body_freq * 1.5 * t)
        body *= np.exp(-t * (creator.decay_factor + 8))
        noise = rng.standard_normal(length_n).astype(np.float32)
        noise = _onepole_hp(noise, 1200, sr)
        noise = _onepole_lp(noise, 6000, sr)
        noise *= np.exp(-t * (creator.decay_factor + 5))
        click = np.zeros(length_n, dtype=np.float32)
        cn = int(0.003 * sr)
        click[:cn] = (rng.standard_normal(cn) * np.exp(-np.linspace(0, 7, cn))).astype(np.float32)
        return (0.45 * body + 0.55 * noise + 0.4 * click * creator.transient_sharpness).astype(np.float32)

    # hat / closed hat
    noise = rng.standard_normal(length_n).astype(np.float32)
    noise = _onepole_hp(noise, 6500, sr)
    decay = creator.decay_factor + 22
    env = np.exp(-t * decay)
    return (noise * env * 0.8).astype(np.float32)


def _grid_times(n_samples: int, n_steps: int, swing: float, sr: int = SAMPLE_RATE) -> list[int]:
    """Return sample-indexed positions of beat onsets, with humanized swing."""
    positions = []
    for k in range(n_steps):
        base = int(k * n_samples / n_steps)
        if k % 2 == 1:
            base += int((n_samples / n_steps) * 0.5 * swing)
        positions.append(base)
    return positions


# ---------------------------------------------------------------------------
# Synthesizers (per category)
# ---------------------------------------------------------------------------

def _synth_drums(category: str, t: np.ndarray, creator: CreatorSignature,
                 rng: np.random.Generator) -> np.ndarray:
    """Drum loops: 2-bar patterns at the creator's preferred density."""
    n = len(t)
    out = np.zeros(n, dtype=np.float32)
    # 2 bars × 16 sixteenths = 32 steps over the full duration
    n_steps_grid = 32

    if category == "kick":
        # 4-on-the-floor backbone (every 4 sixteenths) + ghost variation
        kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
        if creator.rhythm_density < 0.5:
            kick_pattern = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 2
        for k, hit in enumerate(kick_pattern):
            if not hit:
                continue
            pos = int(k * n / n_steps_grid)
            length = min(int(0.35 * SAMPLE_RATE), n - pos)
            if length <= 0:
                continue
            vel = 0.9 + 0.1 * rng.random()
            out[pos:pos + length] += _drum_hit("kick", length, creator, rng) * vel
        # ghost rolls when dense
        if creator.rhythm_density > 0.7:
            for _ in range(int(creator.rhythm_density * 6)):
                p = rng.integers(int(0.05 * n), int(0.92 * n))
                length = min(int(0.12 * SAMPLE_RATE), n - p)
                out[p:p + length] += _drum_hit("kick", length, creator, rng) * 0.3

    elif category == "snare":
        # backbeat on every "3rd" sixteenth of each bar (positions 4 and 12)
        snare_steps = [4, 12, 20, 28]
        for k in snare_steps:
            pos = int(k * n / n_steps_grid)
            length = min(int(0.4 * SAMPLE_RATE), n - pos)
            if length > 0:
                out[pos:pos + length] += _drum_hit("snare", length, creator, rng) * 0.95
        # rim/ghost hits between
        if creator.rhythm_density > 0.5:
            for k in range(int(creator.rhythm_density * 8)):
                pos = rng.integers(int(0.05 * n), int(0.92 * n))
                length = min(int(0.08 * SAMPLE_RATE), n - pos)
                out[pos:pos + length] += _drum_hit("snare", length, creator, rng) * 0.25
        # roll fill at end of bar 2
        if creator.rhythm_density > 0.7:
            roll_start = int(0.85 * n)
            roll_n = n - roll_start
            for k in range(8):
                p = roll_start + int(k * roll_n / 8)
                length = min(int(0.05 * SAMPLE_RATE), n - p)
                if length > 0:
                    out[p:p + length] += _drum_hit("snare", length, creator, rng) * (0.3 + 0.05 * k)

    else:  # hat
        # 16th-note hat pattern with swing + open hat accents
        for k, pos in enumerate(_grid_times(n, n_steps_grid, creator.swing)):
            length = min(int(0.05 * SAMPLE_RATE), n - pos)
            if length <= 0:
                continue
            vel = 0.55 + 0.4 * rng.random()
            if k % 2 == 1:
                vel *= 0.65
            # open hat accent every 8 steps
            if k % 8 == 6 and creator.rhythm_density > 0.4:
                length = min(int(0.18 * SAMPLE_RATE), n - pos)
                vel *= 1.2
            out[pos:pos + length] += _drum_hit("hat", length, creator, rng) * vel

    return out


def _synth_bass(category: str, t: np.ndarray, creator: CreatorSignature,
                rng: np.random.Generator) -> np.ndarray:
    """Layered bass: sub sine + saturated mid harmonics + groove."""
    n = len(t)
    base = creator.base_freq_hz * (0.5 if category == "sub_bass" else 1.0)
    base = max(40, min(180, base))
    detune = 2 ** (creator.detune_cents / 1200)

    # 8-note bass riff, 2 bars, with rests
    riff_minor = [1.0, 1.0, 1.0, 1.5, 1.0, 0.75, 1.0, 1.0,
                  1.0, 1.122, 1.0, 1.5, 1.682, 1.0, 0.75, 0.5]
    riff_major = [1.0, 1.0, 1.0, 1.26, 1.0, 1.498, 1.0, 1.0,
                  1.0, 1.0, 1.26, 1.498, 1.682, 1.498, 1.26, 1.0]
    intervals = riff_minor if creator.harmonic_richness < 0.55 else riff_major
    n_notes = 16 if creator.rhythm_density > 0.55 else 8
    intervals = intervals[:n_notes]

    out = np.zeros(n, dtype=np.float32)
    note_n = n // n_notes
    for k in range(n_notes):
        # rests: skip 10% of notes for groove
        if rng.random() < 0.1 * (1 - creator.rhythm_density):
            continue
        f = base * intervals[k % len(intervals)]
        seg_t = np.arange(note_n) / SAMPLE_RATE
        sub = np.sin(2 * np.pi * f * seg_t).astype(np.float32)
        mid = (np.sin(2 * np.pi * f * detune * seg_t)
               + 0.5 * np.sin(2 * np.pi * f * 2 * seg_t)
               + 0.25 * np.sin(2 * np.pi * f * 3 * seg_t)).astype(np.float32)
        mid = np.tanh(mid * (1.6 + 2.5 * creator.harmonic_richness))
        env = _adsr(note_n, 0.005, 0.06, 0.6, 0.08)
        seg = (0.7 * sub + 0.55 * mid * creator.harmonic_richness) * env
        start = k * note_n
        out[start:start + note_n] += seg

    out = _onepole_lp(out, 600 + 1200 * creator.harmonic_richness)
    return out


def _synth_lead(category: str, t: np.ndarray, creator: CreatorSignature,
                rng: np.random.Generator) -> np.ndarray:
    """Lead / pluck / arp loops: melodic 2-bar sequence with creator's timbre."""
    n = len(t)
    base = max(180, creator.base_freq_hz * 3.5)
    if category == "arp":
        n_notes = 16  # 8th-note arpeggio across 2 bars
    elif category == "pluck":
        n_notes = 12 if creator.rhythm_density > 0.5 else 8
    else:  # lead
        n_notes = 8 if creator.rhythm_density > 0.5 else 6
    note_n = n // n_notes

    # scale ratios
    scale = [1.0, 1.122, 1.26, 1.498, 1.682, 2.0, 2.245, 2.52]
    out = np.zeros(n, dtype=np.float32)
    if category == "arp":
        # cycle a 3-note pattern (root, third, fifth) with octave jumps
        pattern = [0, 2, 4, 2, 5, 4, 2, 0] * 2
        rng_seq = pattern[:n_notes]
    else:
        rng_seq = rng.integers(0, len(scale), n_notes).tolist()

    for k in range(n_notes):
        # occasional rest in lead/pluck
        if category != "arp" and rng.random() < 0.12 * (1 - creator.rhythm_density):
            continue
        f = base * scale[rng_seq[k]]
        seg_t = np.arange(note_n) / SAMPLE_RATE
        if category == "pluck":
            wave = _supersaw(f, note_n, n_voices=3, detune_cents=creator.detune_cents)
            env = _adsr(note_n, 0.002, 0.05 + 0.1 * (1 - creator.decay_factor / 10),
                        0.0, 0.04)
        elif category == "arp":
            wave = _supersaw(f, note_n, n_voices=3, detune_cents=creator.detune_cents * 0.6)
            env = _adsr(note_n, 0.003, 0.02, 0.4, 0.03)
        else:  # lead
            wave = _supersaw(f, note_n, n_voices=5, detune_cents=creator.detune_cents)
            env = _adsr(note_n, 0.01, 0.04, 0.7, 0.08)
        # vibrato on lead
        if category == "lead" and creator.harmonic_richness > 0.4:
            vib = 0.015 * np.sin(2 * np.pi * 5.5 * seg_t)
            wave *= 1 + vib
        seg = wave * env
        start = k * note_n
        out[start:start + note_n] += seg

    out = _onepole_lp(out, 4500 + 3500 * creator.harmonic_richness)
    return _short_reverb(out, creator.reverb_amount * 0.5)


def _synth_pad(category: str, t: np.ndarray, creator: CreatorSignature,
               rng: np.random.Generator) -> np.ndarray:
    """Sustained pad / ambient: 2-chord progression, slow LFO, wet reverb."""
    n = len(t)
    root = creator.base_freq_hz * 1.2
    if root > 350:
        root /= 2
    if root < 110:
        root *= 2
    minor = creator.harmonic_richness < 0.5
    triad = [1.0, 1.189 if minor else 1.26, 1.498]

    # Two chord segments — root, then IV or V
    chord_progression = [(1.0, triad), (1.498 if minor else 1.335, triad)]
    out = np.zeros(n, dtype=np.float32)
    seg_n = n // len(chord_progression)
    for ci, (mult, intervals) in enumerate(chord_progression):
        seg = np.zeros(seg_n, dtype=np.float32)
        for r in intervals:
            seg += _supersaw(root * mult * r, seg_n, n_voices=5,
                             detune_cents=10 + creator.detune_cents)
        seg /= len(intervals)
        # crossfade between chord segments
        fade_n = int(seg_n * 0.15)
        env_seg = np.ones(seg_n, dtype=np.float32)
        env_seg[:fade_n] = np.linspace(0, 1, fade_n)
        env_seg[-fade_n:] = np.linspace(1, 0, fade_n)
        out[ci * seg_n:(ci + 1) * seg_n] += seg * env_seg

    # slow LFO on amplitude
    lfo_freq = 0.4 + creator.rhythm_density * 0.6
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * lfo_freq * t)
    # ADSR with very slow attack
    env = _adsr(n, 0.6 if category == "pad" else 0.3, 0.2, 0.9, 0.6)
    out = out * lfo * env
    out = _onepole_lp(out, 1800 + 2200 * creator.harmonic_richness)
    return _short_reverb(out, 0.3 + creator.reverb_amount * 0.6)


def _synth_fx(category: str, t: np.ndarray, creator: CreatorSignature,
              rng: np.random.Generator) -> np.ndarray:
    """FX / vocal chops: FM-modulated formant filtered with reverb tail."""
    n = len(t)
    out = np.zeros(n, dtype=np.float32)

    if category == "vocal_chop":
        # 5-9 short "vowel" bursts at varied pitches across 2 bars
        n_chops = 5 + int(creator.rhythm_density * 4)
        positions = sorted(rng.choice(int(n * 0.85), n_chops, replace=False))
        for p in positions:
            length = min(int(rng.uniform(0.12, 0.25) * SAMPLE_RATE), n - p)
            if length <= 0:
                continue
            seg_t = np.arange(length) / SAMPLE_RATE
            f0 = creator.base_freq_hz * rng.uniform(2.0, 4.0)
            # carrier saw
            carrier = _supersaw(f0, length, n_voices=3, detune_cents=8)
            # formant filter approximation: emphasize 2 bands
            formant1 = 700 * creator.formant_shift
            formant2 = 1800 * creator.formant_shift
            band1 = _onepole_lp(carrier, formant1 + 200) - _onepole_lp(carrier, formant1 - 200)
            band2 = _onepole_lp(carrier, formant2 + 400) - _onepole_lp(carrier, formant2 - 400)
            voiced = (band1 + 0.6 * band2).astype(np.float32)
            env = _adsr(length, 0.01, 0.05, 0.6, 0.06)
            out[p:p + length] += voiced * env * rng.uniform(0.65, 1.0)
        out = _short_reverb(out, 0.35 + creator.reverb_amount)
        return out

    # fx — sweep / riser / impact
    style = rng.integers(0, 3)
    if style == 0:
        # rising sweep
        f_start = creator.base_freq_hz * 0.5
        f_end = f_start * 8
        freq = np.linspace(f_start, f_end, n)
        wave = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
        wave += 0.4 * np.sin(2 * np.pi * np.cumsum(freq * 1.5) / SAMPLE_RATE)
        env = (1 - np.exp(-t * 4)) * np.exp(-((t - DURATION_SEC * 0.85) ** 2) * 50)
        out = (wave * env * 0.7).astype(np.float32)
    elif style == 1:
        # impact (sub boom + transient)
        boom_n = int(0.6 * SAMPLE_RATE)
        boom_t = np.arange(boom_n) / SAMPLE_RATE
        boom = np.sin(2 * np.pi * (60 - 30 * boom_t) * boom_t).astype(np.float32)
        boom *= np.exp(-boom_t * 4)
        out[:boom_n] = boom * 1.0
        click_n = int(0.01 * SAMPLE_RATE)
        out[:click_n] += rng.standard_normal(click_n).astype(np.float32) * 0.5
    else:
        # noise riser
        noise = rng.standard_normal(n).astype(np.float32)
        noise = _onepole_hp(noise, 1500)
        env = (np.linspace(0, 1, n) ** 2)
        out = noise * env * 0.4
    return _short_reverb(out, 0.3 + creator.reverb_amount * 0.6)


# ---------------------------------------------------------------------------
# Top-level synth dispatcher
# ---------------------------------------------------------------------------

def synth_stem(
    category: str,
    creator: CreatorSignature,
    seed: int,
    n_samples: int = N_SAMPLES,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION_SEC, n_samples, endpoint=False)

    if category in ("kick", "snare", "hat"):
        audio = _synth_drums(category, t, creator, rng)
    elif category in ("bass", "sub_bass"):
        audio = _synth_bass(category, t, creator, rng)
    elif category in ("lead", "pluck", "arp"):
        audio = _synth_lead(category, t, creator, rng)
    elif category in ("pad", "ambient"):
        audio = _synth_pad(category, t, creator, rng)
    elif category in ("vocal_chop", "fx"):
        audio = _synth_fx(category, t, creator, rng)
    else:
        audio = _synth_pad("pad", t, creator, rng)

    if creator.noise_floor > 0:
        audio = audio + rng.standard_normal(n_samples).astype(np.float32) * creator.noise_floor

    # Peak-normalize to ~-1 dBFS
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak * 0.89
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------

def build_catalog(
    output_dir: Path,
    metadata_path: Path,
    n_sources: int = CATALOG_SIZE,
    n_creators: int = N_CREATORS,
    seed: int = GLOBAL_SEED,
) -> list[SourceRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    creators = make_creators(n_creators, seed)
    rng = np.random.default_rng(seed)

    sources_per_creator = n_sources // n_creators
    extras = n_sources - sources_per_creator * n_creators

    records: list[SourceRecord] = []
    src_idx = 0
    for ci, creator in enumerate(creators):
        n_for_this = sources_per_creator + (1 if ci < extras else 0)
        for _ in range(n_for_this):
            category = CATALOG_CATEGORIES[rng.integers(0, len(CATALOG_CATEGORIES))]
            audio = synth_stem(category, creator, seed=seed * 1000 + src_idx)

            src_id = f"src_{src_idx:03d}"
            file_path = output_dir / f"{src_id}.wav"
            save_wav(file_path, audio)

            records.append(SourceRecord(
                source_id=src_id,
                creator_id=creator.creator_id,
                creator_name=creator.name,
                category=category,
                bpm=int(rng.integers(85, 145)),
                key=KEYS[rng.integers(0, len(KEYS))],
                file_path=str(file_path.relative_to(output_dir.parent)),
            ))
            src_idx += 1

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump({
            "creators": [c.as_dict() for c in creators],
            "sources": [r.as_dict() for r in records],
        }, f, indent=2)

    return records


def load_catalog_metadata(metadata_path: Path) -> tuple[list[dict], list[dict]]:
    with open(metadata_path) as f:
        data = json.load(f)
    return data["creators"], data["sources"]
