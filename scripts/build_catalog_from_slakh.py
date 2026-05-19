#!/usr/bin/env python
"""
build_catalog_from_slakh.py — Build CASCADE-V catalog from Slakh2100.

Slakh2100 is an open multi-track MIDI dataset (Manilow et al. 2019) with
~2,100 mixes rendered from MIDI through high-quality sample libraries.
Each track folder has the layout:

    Track00001/
        metadata.yaml           # per-stem instrument / program info
        all_src.mid             # full MIDI
        stems/
            S00.wav             # rendered stem audio (typically 44.1 kHz)
            S01.wav
            ...

This script samples N stems from across the dataset, cuts each to 10 s
(by extracting the highest-energy 10-second window), and writes them into
data/catalog/ in the same layout as the synthetic catalog. The metadata
JSON includes a derived `creator_id` keyed off (program family × inst class)
so the pipeline has creator-DNA cases for free — multiple stems by the same
"creator" are stems rendered from the same program family.

Why this matters: the synthetic oscillator-based catalog gave the encoder
no producer/engineer fingerprints to learn, so creator-DNA wasn't visible
in the embedding. Slakh stems carry the rendering chain's actual signal
character — sample-rate conversion artefacts, library-specific timbres,
fixed velocity curves — which is exactly what creator attribution needs.

Usage:
    python scripts/build_catalog_from_slakh.py \\
        --slakh-root /path/to/slakh2100_flac_redux \\
        --catalog-size 5000 \\
        --seed 42

After running this, the rest of the pipeline (build_embeddings_and_tests,
train_encoder, attribute, evaluate_all) sees the new catalog the same way
it saw the synthetic one — no other code changes required.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cascade_v.config import (
    CATALOG_DIR,
    CATALOG_METADATA_PATH,
    CATALOG_SIZE,
    DURATION_SEC,
    GLOBAL_SEED,
    SAMPLE_RATE,
    ensure_dirs,
)
from cascade_v.utils.audio import save_wav
from cascade_v.utils.synth import SourceRecord


# ---------------------------------------------------------------------------
# Program-family → category mapping
# ---------------------------------------------------------------------------
# Slakh follows the General MIDI program-number convention. We collapse the
# 128 programs into the same coarse categories used by CASCADE-V so the
# catalog mix stays comparable to the synthetic baseline.

_PROGRAM_TO_CATEGORY = {
    # Drums (channel 10 / inst_class "Drums")
    "Drums": "kick",        # individual drum stems get bucketed by inst_class below
    # Bass (programs 32-39)
    "Bass": "bass",
    # Piano (0-7) → "pluck" (closest synthetic-catalog category)
    "Piano": "pluck",
    "Chromatic Percussion": "pluck",
    # Organ (16-23) → "pad"
    "Organ": "pad",
    # Guitar (24-31) → "lead"
    "Guitar": "lead",
    # Strings (40-47) → "pad"
    "Strings": "pad",
    "Ensemble": "pad",
    # Brass (56-63) → "lead"
    "Brass": "lead",
    # Reed (64-71) → "lead"
    "Reed": "lead",
    # Pipe (72-79) → "lead"
    "Pipe": "lead",
    # Synth Lead (80-87) → "lead"
    "Synth Lead": "lead",
    # Synth Pad (88-95) → "pad"
    "Synth Pad": "pad",
    # Synth Effects (96-103) → "fx"
    "Synth Effects": "fx",
    # Ethnic (104-111) → "pluck"
    "Ethnic": "pluck",
    # Percussive (112-119) → "hat"
    "Percussive": "hat",
    # Sound Effects (120-127) → "fx"
    "Sound Effects": "fx",
}


def _category_for_stem(inst_class: str, midi_program: int) -> str:
    """Map a Slakh stem's inst_class + program to a CASCADE-V category."""
    cat = _PROGRAM_TO_CATEGORY.get(inst_class)
    if cat is not None:
        return cat
    # Fallbacks based on raw program number ranges
    if 0 <= midi_program <= 7:
        return "pluck"
    if 32 <= midi_program <= 39:
        return "bass"
    if 80 <= midi_program <= 87:
        return "lead"
    if 88 <= midi_program <= 95:
        return "pad"
    return "fx"


def _creator_id_for_stem(inst_class: str, midi_program: int) -> tuple[str, str]:
    """
    Derive (creator_id, creator_name) from stem metadata. We bucket every
    8 consecutive MIDI programs into a "creator" so each rendering family
    (e.g. "Synth Lead 0-7", "Synth Pad 0-7") becomes a creator. Yields
    ~16 distinct creators for a typical Slakh sample, matching the synthetic
    catalog's N_CREATORS=32 ballpark when both classes and program bands
    are factored in.
    """
    band = midi_program // 8
    creator_id = f"creator_{inst_class.replace(' ', '_').lower()}_{band:02d}"
    creator_name = f"{inst_class} band {band}"
    return creator_id, creator_name


# ---------------------------------------------------------------------------
# Audio helpers (Slakh ships at 44.1 kHz; we resample to the project rate)
# ---------------------------------------------------------------------------

def _load_resampled(path: Path, target_sr: int) -> np.ndarray:
    import soundfile as sf  # part of base deps

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # Linear interp resample. Cheap and good enough for embeddings;
        # production would use librosa.resample / julius if available.
        ratio = target_sr / sr
        n_out = int(round(len(audio) * ratio))
        if n_out <= 1:
            return np.zeros(target_sr, dtype=np.float32)
        old_t = np.linspace(0, 1, len(audio), endpoint=False)
        new_t = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(new_t, old_t, audio).astype(np.float32)
    return audio


def _highest_energy_window(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Return the n_samples slice of `audio` with the highest RMS energy.
    If the audio is shorter than n_samples, pad with silence on the right.
    Avoids picking a silent intro / outro when the stem is mostly empty.
    """
    if len(audio) <= n_samples:
        out = np.zeros(n_samples, dtype=np.float32)
        out[: len(audio)] = audio
        return out

    # Coarse scan: 1 Hz hop on a power envelope is plenty for 10s windows
    # in a few-minute stem (~300 candidate windows max).
    hop = max(1, n_samples // 10)
    n_steps = (len(audio) - n_samples) // hop + 1
    best_i = 0
    best_rms = -1.0
    for s in range(n_steps):
        i0 = s * hop
        chunk = audio[i0 : i0 + n_samples]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        if rms > best_rms:
            best_rms = rms
            best_i = i0
    return audio[best_i : best_i + n_samples].astype(np.float32)


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def _enumerate_stems(slakh_root: Path) -> list[dict]:
    """
    Return a list of {track, stem_id, audio_path, inst_class, midi_program,
    bpm} dicts for every stem in the Slakh root. Skips tracks without
    metadata.yaml or whose stems folder is missing.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to read Slakh metadata.\n"
            "Install with: pip install -e \".[audio_meta]\""
        ) from e

    stems: list[dict] = []
    track_dirs = sorted(p for p in slakh_root.iterdir() if p.is_dir() and p.name.startswith("Track"))
    if not track_dirs:
        raise SystemExit(
            f"No Track* directories under {slakh_root}. "
            "Expected Slakh2100 layout: <root>/Track00001/{metadata.yaml,stems/SXX.wav,all_src.mid}."
        )

    for tdir in track_dirs:
        meta_path = tdir / "metadata.yaml"
        stems_dir = tdir / "stems"
        if not meta_path.exists() or not stems_dir.is_dir():
            continue
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception:
            continue

        bpm = 0
        # Slakh metadata.yaml doesn't always carry BPM at top level; we'll
        # try to recover it from the MIDI later if needed (lazy).
        stem_meta = (meta or {}).get("stems", {})
        for stem_id, info in stem_meta.items():
            if not isinstance(info, dict):
                continue
            audio_path = stems_dir / f"{stem_id}.wav"
            if not audio_path.exists():
                # Slakh redux uses .flac in some variants
                flac = stems_dir / f"{stem_id}.flac"
                if flac.exists():
                    audio_path = flac
                else:
                    continue
            stems.append({
                "track": tdir.name,
                "stem_id": stem_id,
                "audio_path": audio_path,
                "inst_class": str(info.get("inst_class", "Unknown")),
                "midi_program": int(info.get("program_num", info.get("midi_program_name_num", 0))),
                "bpm": bpm,
            })
    return stems


def _bpm_for_track(track_dir: Path) -> float:
    """Read the first tempo event from all_src.mid. Returns 0.0 if unavailable."""
    midi_path = track_dir / "all_src.mid"
    if not midi_path.exists():
        return 0.0
    try:
        import pretty_midi  # type: ignore[import-not-found]

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        tempi_times, tempi = pm.get_tempo_changes()
        if len(tempi) == 0:
            return 0.0
        return float(tempi[0])
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CASCADE-V catalog from Slakh2100.")
    parser.add_argument(
        "--slakh-root", type=Path, required=True,
        help="Path to the Slakh2100 root containing Track00001, Track00002, ...",
    )
    parser.add_argument("--catalog-size", type=int, default=CATALOG_SIZE)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument(
        "--output-dir", type=Path, default=CATALOG_DIR,
        help="Where to write the WAV files (default: data/catalog/).",
    )
    parser.add_argument(
        "--metadata-path", type=Path, default=CATALOG_METADATA_PATH,
        help="Where to write catalog_metadata.json.",
    )
    args = parser.parse_args()

    ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[slakh] enumerating stems under {args.slakh_root} …")
    all_stems = _enumerate_stems(args.slakh_root)
    print(f"[slakh] found {len(all_stems)} stems across {len({s['track'] for s in all_stems})} tracks.")

    rng = random.Random(args.seed)
    rng.shuffle(all_stems)

    n_target = min(args.catalog_size, len(all_stems))
    if n_target < args.catalog_size:
        print(f"[slakh] warning: requested {args.catalog_size} stems but only {n_target} available.")

    # Cache BPMs per track so we read each MIDI once
    bpm_cache: dict[str, float] = {}

    # Lazy import the BPM/key estimator (librosa-backed; same one Stage-1
    # uses at runtime) so estimates and runtime filtering use the same
    # algorithm — no train/serve skew.
    from cascade_v.embeddings import estimate_bpm, estimate_key

    n_samples = int(SAMPLE_RATE * DURATION_SEC)
    creators_seen: dict[str, dict] = {}
    records: list[SourceRecord] = []

    skipped = 0
    for i, stem in enumerate(all_stems):
        if len(records) >= n_target:
            break
        try:
            audio = _load_resampled(stem["audio_path"], SAMPLE_RATE)
        except Exception:
            skipped += 1
            continue
        if float(np.sqrt(np.mean(audio * audio) + 1e-12)) < 1e-4:
            # Skip nearly-silent stems
            skipped += 1
            continue

        window = _highest_energy_window(audio, n_samples)

        category = _category_for_stem(stem["inst_class"], stem["midi_program"])
        creator_id, creator_name = _creator_id_for_stem(stem["inst_class"], stem["midi_program"])
        creators_seen.setdefault(creator_id, {
            "creator_id": creator_id, "name": creator_name,
        })

        # BPM: prefer MIDI tempo, else estimate from audio
        track = stem["track"]
        if track not in bpm_cache:
            bpm_cache[track] = _bpm_for_track(args.slakh_root / track)
        bpm = bpm_cache[track]
        if bpm <= 0:
            bpm = estimate_bpm(window, sample_rate=SAMPLE_RATE)

        key = estimate_key(window, sample_rate=SAMPLE_RATE)

        src_id = f"src_{len(records):04d}"
        wav_path = args.output_dir / f"{src_id}.wav"
        save_wav(wav_path, window)

        records.append(SourceRecord(
            source_id=src_id,
            creator_id=creator_id,
            creator_name=creator_name,
            category=category,
            bpm=int(round(bpm)) if bpm > 0 else 0,
            key=key or "",
            file_path=str(wav_path.relative_to(args.output_dir.parent)),
        ))

        if (i + 1) % 200 == 0:
            print(f"[slakh] processed {i + 1}/{len(all_stems)}, kept {len(records)}, skipped {skipped}")

    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata_path, "w") as f:
        json.dump({
            "creators": list(creators_seen.values()),
            "sources": [r.as_dict() for r in records],
            "source_dataset": "slakh2100",
            "duration_sec": DURATION_SEC,
            "sample_rate": SAMPLE_RATE,
        }, f, indent=2)

    print(
        f"[slakh] done. {len(records)} stems, {len(creators_seen)} creators, "
        f"skipped {skipped}. Metadata: {args.metadata_path}"
    )


if __name__ == "__main__":
    main()
