"""
settings.py — Pydantic-validated runtime configuration.

Everything in `config.py` is now produced by an instance of `Settings`.
Override any value at runtime via env vars (CASCADE_TRAIN_EPOCHS=10 ...) or
by passing kwargs to `Settings(...)`. Validators bound the values so a
broken hyperparameter is caught at startup rather than mid-pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CASCADE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Audio
    SAMPLE_RATE: int = 22050
    # Bumped 4.0 → 10.0: at 90 BPM, 4s = 6 beats (<1 bar) — too little context
    # for the encoder to learn rhythmic / harmonic identity. 10s ≈ 1-4 bars at
    # typical BPMs, matches Splice loop durations (12-29s observed), and roughly
    # triples the mel-spectrogram time dimension for stronger embeddings.
    DURATION_SEC: float = 10.0
    N_FFT: int = 1024
    HOP_LENGTH: int = 256
    N_MELS: int = 64

    # Encoder
    EMBEDDING_DIM: int = 256
    ENCODER_CHANNELS: List[int] = [48, 96, 192, 384]
    # Which encoder to use at attribution time:
    #   "custom" — the 3M-param ResCNN trained by cascade-train.
    #              Wins on our SYNTHETIC catalog (creator-specific signatures
    #              are visible because the encoder was trained on this data).
    #   "clap"   — LAION CLAP foundation model (no training required).
    #              Would win on REAL audio (real drum kits, real instruments)
    #              but underperforms on our synthetic stems because CLAP
    #              treats synth-generated audio as one homogeneous category.
    # Production path with real Splice samples → switch to "clap".
    # Demo / synthetic-data path → keep "custom".
    ENCODER_KIND: str = "custom"

    # Training
    TRAIN_BATCH_SIZE: int = 64
    TRAIN_EPOCHS: int = 200
    TRAIN_LR: float = 1e-3
    TRAIN_WEIGHT_DECAY: float = 1e-5
    CONTRASTIVE_TEMPERATURE: float = 0.1
    AUGMENTATIONS_PER_SAMPLE: int = 4

    # Catalog
    # Bumped 400 → 5000: 12× larger, exercises Stage-1 retrieval and Stage-2
    # clustering at meaningful scale while staying tractable on M-series
    # laptop. Demo wall-clock target: end-to-end ~30 min at 5k.
    CATALOG_SIZE: int = 5000
    N_CREATORS: int = 32
    CATALOG_CATEGORIES: List[str] = [
        "kick", "snare", "hat",
        "bass", "sub_bass",
        "lead", "pluck", "arp",
        "pad", "ambient",
        "vocal_chop", "fx",
    ]

    # Test outputs
    N_TEST_OUTPUTS: int = 30
    TEST_OUTPUT_MIN_SOURCES: int = 3
    TEST_OUTPUT_MAX_SOURCES: int = 6

    # Pipeline hyperparameters
    TRIAGE_TOP_K: int = 40
    CLUSTER_DISTANCE_THRESHOLD: float = 0.6
    # Sharpened 4.0 → 8.0: the long-tail of the value function was depositing
    # 1-3% mass on each false-positive candidate. Higher temperature
    # concentrates mass on coalitions that actually move the value function.
    SHAPLEY_VALUE_TEMPERATURE: float = 8.0
    SHAPLEY_MAX_EXACT_N: int = 10
    SHAPLEY_MC_PERMUTATIONS: int = 400

    # Stage-2 clustering algorithm. "ward" = scipy hierarchical (default,
    # SMT-stable). "hdbscan" = density-based, no distance-threshold knob,
    # handles outlier candidates as their own group.
    CLUSTERING_METHOD: str = "ward"

    # Stage-1 BPM/key pre-filter (reduces false positives by gating
    # the catalog to harmonically/rhythmically compatible sources). No-op
    # when catalog metadata lacks bpm/key fields.
    USE_BPM_KEY_FILTER: bool = False
    BPM_FILTER_TOLERANCE: float = 3.0  # ± BPM

    # Coalition-game-theoretic method for combining Stage 2 (clusters) with
    # Stage 3 (within-cluster split). Two options:
    #   "shapley_x_cluster" — heuristic: cluster_weight × within_cluster_shapley.
    #                   Empirically best on our synthetic data; acts as a
    #                   regularizer when the value function is approximate.
    #   "owen"        — Owen value (Owen 1977), the formal cooperative-game-
    #                   theoretic extension of Shapley for games with a
    #                   coalition structure. Formally more correct but
    #                   marginally worse on synthetic embeddings. Use this
    #                   when running on real-world audio where the value
    #                   function is well-calibrated. Cost ≈ 2^m · 2^max_n_C.
    OWEN_OR_DECOMP: str = "shapley_x_cluster"

    # Stage 0: Demucs source separation before attribution.
    # The input mix is split into N stems; Stages 1-V run on each separately,
    # then per-source weights are aggregated. Halves instance MAE empirically
    # by eliminating dominant-source masking. Cost: ~3 sec per attribution.
    # Set to False to fall back to single-pass attribution on the raw mix.
    USE_DEMUCS_SEPARATION: bool = True

    # Which Demucs checkpoint to use. "auto" = inspect catalog metadata at
    # startup and pick the smallest pretrained model that covers the union
    # of catalog categories with a dedicated stem (rather than dumping the
    # uncovered categories into "other"). Explicit overrides:
    #   "htdemucs"     — 4 stems (drums, bass, other, vocals)
    #   "htdemucs_6s"  — 6 stems (drums, bass, piano, guitar, vocals, other)
    #   "htdemucs_ft"  — 4 stems, fine-tuned variant (slightly better SDR)
    #   "mdx_extra"    — 4 stems, older legacy (kept for ablations)
    DEMUCS_MODEL: str = "auto"

    # Sparsification thresholds (false-positive control)
    # Tightened 3 % → 7 % and 1.5 % → 4 % to suppress the long-tail of weak
    # contributors that was inflating receipts to 13+ creators when ground
    # truth was 3-6. Combined with the sharper SHAPLEY_VALUE_TEMPERATURE,
    # this concentrates payout on real contributors. Renormalization after
    # sparsification preserves the efficiency axiom (Σ w = 1).
    MIN_CLUSTER_WEIGHT: float = 0.07  # 7 %
    MIN_FINAL_WEIGHT: float = 0.04    # 4 %

    # Verification
    INTERVAL_PRECISION_BITS: int = 80
    HOEFFDING_CONFIDENCE: float = 0.95
    EFFICIENCY_TOLERANCE: float = 1e-3
    SYMMETRY_TOLERANCE: float = 1e-3
    DUMMY_TOLERANCE: float = 1e-3
    ADDITIVITY_TOLERANCE: float = 1e-2

    # Triple-currency layer (currencies.py).
    # ENABLE_CURRENCIES = True attaches a `currencies` block to every
    # receipt with QWS-shaped monetary payouts (post-Rawlsian floor),
    # symbolic role tags, lottery feature, and opportunity adjustments.
    # The dials are recorded per-receipt and override-able via CLI.
    ENABLE_CURRENCIES: bool = True
    CURRENCY_ALPHA: float = 0.85   # fairness strictness (1.0 = pure Shapley)
    CURRENCY_BETA: float = 0.60    # recognition spread  (1.0 = top wins)
    CURRENCY_GAMMA: float = 0.30   # opportunity redist  (0.0 = pure merit)

    # Reproducibility
    GLOBAL_SEED: int = 42

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON_PATH: Path = Field(default_factory=lambda: Path("logs/cascade_v.jsonl"))

    @field_validator("EMBEDDING_DIM")
    @classmethod
    def _embedding_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("EMBEDDING_DIM must be > 0")
        return v

    @field_validator("CLUSTER_DISTANCE_THRESHOLD")
    @classmethod
    def _cluster_in_range(cls, v: float) -> float:
        if not (0.0 < v < 2.0):
            raise ValueError("CLUSTER_DISTANCE_THRESHOLD must be in (0, 2)")
        return v

    @field_validator("SHAPLEY_MAX_EXACT_N")
    @classmethod
    def _max_exact_reasonable(cls, v: int) -> int:
        if not (1 <= v <= 14):
            raise ValueError("SHAPLEY_MAX_EXACT_N must be in [1, 14]")
        return v

    @field_validator("HOEFFDING_CONFIDENCE")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("HOEFFDING_CONFIDENCE must be in (0, 1)")
        return v

    @field_validator("CURRENCY_ALPHA", "CURRENCY_BETA", "CURRENCY_GAMMA")
    @classmethod
    def _dial_in_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("currency dials must be in [0, 1]")
        return v

    def model_post_init(self, __context) -> None:
        if self.TRIAGE_TOP_K > self.CATALOG_SIZE:
            raise ValueError(
                f"TRIAGE_TOP_K ({self.TRIAGE_TOP_K}) > CATALOG_SIZE ({self.CATALOG_SIZE})"
            )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(**kwargs) -> Settings:
    """Replace the cached settings (for CLI overrides). Returns the new instance."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
