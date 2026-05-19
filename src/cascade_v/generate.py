"""
generate.py — Generate test outputs with known ground-truth source contributions.

This is the analog of "Variations" / "Magic Fit": given a small set of
sources from the catalog and target weights, produce a generated audio
output. We use a *nonlinear* mixing process (weighted sum + spectral
shaping + light convolution) to make the attribution problem nontrivial —
linear mixing would be too easy.

Crucially, we keep ground truth: we know exactly which sources were used
and with what relative contribution. This lets us measure attribution
accuracy quantitatively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from cascade_v.config import (
    GLOBAL_SEED,
    GROUND_TRUTH_PATH,
    N_SAMPLES,
    N_TEST_OUTPUTS,
    SAMPLE_RATE,
    TEST_OUTPUT_MAX_SOURCES,
    TEST_OUTPUT_MIN_SOURCES,
    TEST_OUTPUTS_DIR,
)
from cascade_v.utils.audio import fix_length, load_wav, save_wav


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthRecord:
    """Ground truth for a single test output."""
    output_id: str
    file_path: str
    source_ids: list[str]
    weights: list[float]                # ground-truth contribution weights, sum to 1
    creator_ids: list[str]              # creator_id per source (for group-level GT)
    creator_weights: dict[str, float]   # ground-truth weight per creator (aggregated)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def nonlinear_mix(
    sources: list[np.ndarray],
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Mix sources with given weights using a nonlinear process.

    Steps:
    1. Linear weighted sum of sources
    2. Soft saturation (tanh) to introduce harmonics
    3. Convolution with a short random impulse response (room/character)
    4. Light spectral coloration via a simple band emphasis

    The nonlinearity makes ground-truth recovery harder than linear mixing
    while still preserving the source contributions in a measurable way.
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=np.float32)

    # 1. Align lengths
    max_len = max(len(s) for s in sources)
    aligned = []
    for s in sources:
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)))
        else:
            s = s[:max_len]
        aligned.append(s)
    stacked = np.stack(aligned)  # (n_sources, max_len)

    # 2. Linear weighted sum
    mixed = (stacked * weights[:, None]).sum(axis=0)

    # 3. Soft saturation
    mixed = np.tanh(mixed * 1.5)

    # 4. Light convolution with a short random IR (50 ms)
    ir_len = int(0.05 * SAMPLE_RATE)
    ir = rng.standard_normal(ir_len).astype(np.float32) * 0.1
    ir[0] = 1.0  # dominant direct
    ir = ir * np.exp(-np.linspace(0, 5, ir_len))
    mixed = np.convolve(mixed, ir, mode="same")

    # 5. Normalize
    peak = np.max(np.abs(mixed)) + 1e-9
    mixed = mixed / peak * 0.85

    return mixed.astype(np.float32)[:N_SAMPLES]


# ---------------------------------------------------------------------------
# Test set generation
# ---------------------------------------------------------------------------

def generate_test_outputs(
    sources: list[dict],            # source records from catalog metadata
    catalog_dir: Path,
    output_dir: Path = TEST_OUTPUTS_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    n_outputs: int = N_TEST_OUTPUTS,
    min_sources: int = TEST_OUTPUT_MIN_SOURCES,
    max_sources: int = TEST_OUTPUT_MAX_SOURCES,
    seed: int = GLOBAL_SEED,
    include_creator_dna_cases: bool = True,
) -> list[GroundTruthRecord]:
    """
    Generate N test outputs. Each output mixes 3-6 random catalog sources
    with random Dirichlet-distributed weights.

    If include_creator_dna_cases is True, some outputs are biased to use
    multiple stems from the same creator. This is the case where group-wise
    attribution should outperform instance-level methods.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Pre-load all source audios
    source_audios = {}
    for s in sources:
        audio = load_wav(catalog_dir / f"{s['source_id']}.wav")
        source_audios[s["source_id"]] = fix_length(audio)

    # Group sources by creator for the DNA cases
    by_creator: dict[str, list[dict]] = {}
    for s in sources:
        by_creator.setdefault(s["creator_id"], []).append(s)
    creator_ids = list(by_creator.keys())

    records: list[GroundTruthRecord] = []
    for i in range(n_outputs):
        n_src = int(rng.integers(min_sources, max_sources + 1))

        # Decide whether this is a creator-DNA case (~30% of outputs)
        is_dna_case = (
            include_creator_dna_cases
            and rng.random() < 0.3
            and i < n_outputs - 1
        )

        if is_dna_case:
            # Pick one creator with enough stems, then sample multiple from them
            eligible = [cid for cid in creator_ids if len(by_creator[cid]) >= 2]
            if eligible:
                main_creator = eligible[rng.integers(0, len(eligible))]
                main_stems = by_creator[main_creator]
                n_from_main = min(int(rng.integers(2, 4)), len(main_stems))
                # Pick stems from the main creator
                idx_main = rng.choice(len(main_stems), n_from_main, replace=False)
                chosen = [main_stems[j] for j in idx_main]
                # Fill the rest from other creators
                others = [s for s in sources if s["creator_id"] != main_creator]
                n_others = max(0, n_src - n_from_main)
                if n_others > 0 and others:
                    idx_other = rng.choice(len(others), n_others, replace=False)
                    chosen += [others[j] for j in idx_other]
            else:
                idx = rng.choice(len(sources), n_src, replace=False)
                chosen = [sources[j] for j in idx]
        else:
            idx = rng.choice(len(sources), n_src, replace=False)
            chosen = [sources[j] for j in idx]

        # Ground-truth weights from Dirichlet
        weights = rng.dirichlet(np.ones(len(chosen))).astype(np.float32)

        # Mix
        audios = [source_audios[s["source_id"]] for s in chosen]
        mixed = nonlinear_mix(audios, weights, seed=int(seed * 1000 + i))

        # Write
        output_id = f"output_{i:03d}"
        file_path = output_dir / f"{output_id}.wav"
        save_wav(file_path, mixed)

        # Aggregate to creator level
        creator_weights: dict[str, float] = {}
        for s, w in zip(chosen, weights):
            creator_weights[s["creator_id"]] = creator_weights.get(s["creator_id"], 0.0) + float(w)

        records.append(GroundTruthRecord(
            output_id=output_id,
            file_path=str(file_path.relative_to(output_dir.parent.parent)),
            source_ids=[s["source_id"] for s in chosen],
            weights=weights.tolist(),
            creator_ids=[s["creator_id"] for s in chosen],
            creator_weights=creator_weights,
        ))

    # Save ground-truth JSON
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ground_truth_path, "w") as f:
        json.dump([r.as_dict() for r in records], f, indent=2)

    return records


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[GroundTruthRecord]:
    """Load ground-truth records."""
    if not path.exists():
        raise FileNotFoundError(f"No ground truth at {path}.")
    with open(path) as f:
        data = json.load(f)
    return [GroundTruthRecord(**d) for d in data]
