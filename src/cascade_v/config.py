"""
config.py — Central configuration for CASCADE-V.

Constants are derived from a `Settings` (pydantic-settings) instance defined in
`cascade_v.settings`. Override at runtime via env vars (CASCADE_*) or by
calling `cascade_v.settings.override_settings(...)` before importing this
module's consumers.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from cascade_v.settings import get_settings

_s = get_settings()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Allow deployments (Docker, CI) to pin PROJECT_ROOT explicitly. Without this,
# a non-editable install would resolve __file__ into site-packages and break
# every data path. CASCADE_V_PROJECT_ROOT takes precedence when set.
_env_root = os.environ.get("CASCADE_V_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_env_root).resolve()
    if _env_root
    else Path(__file__).resolve().parent.parent.parent
)
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
TEST_OUTPUTS_DIR = DATA_DIR / "test_outputs"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RECEIPTS_DIR = OUTPUTS_DIR / "receipts"
PROOFS_DIR = OUTPUTS_DIR / "proofs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

CATALOG_METADATA_PATH = DATA_DIR / "catalog_metadata.json"
CATALOG_EMBEDDINGS_PATH = DATA_DIR / "catalog_embeddings.npy"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
ENCODER_CHECKPOINT = MODELS_DIR / "encoder.pt"
TRAINING_LOG = LOGS_DIR / "training.json"
CASCADE_LOG_PATH = LOGS_DIR / "cascade_v.jsonl"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

SAMPLE_RATE = _s.SAMPLE_RATE
DURATION_SEC = _s.DURATION_SEC
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
N_FFT = _s.N_FFT
HOP_LENGTH = _s.HOP_LENGTH
N_MELS = _s.N_MELS


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

EMBEDDING_DIM = _s.EMBEDDING_DIM
ENCODER_CHANNELS = list(_s.ENCODER_CHANNELS)
ENCODER_KIND = _s.ENCODER_KIND


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN_BATCH_SIZE = _s.TRAIN_BATCH_SIZE
TRAIN_EPOCHS = _s.TRAIN_EPOCHS
TRAIN_LR = _s.TRAIN_LR
TRAIN_WEIGHT_DECAY = _s.TRAIN_WEIGHT_DECAY
CONTRASTIVE_TEMPERATURE = _s.CONTRASTIVE_TEMPERATURE
AUGMENTATIONS_PER_SAMPLE = _s.AUGMENTATIONS_PER_SAMPLE


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CATALOG_SIZE = _s.CATALOG_SIZE
N_CREATORS = _s.N_CREATORS
CATALOG_CATEGORIES = list(_s.CATALOG_CATEGORIES)


# ---------------------------------------------------------------------------
# Test outputs
# ---------------------------------------------------------------------------

N_TEST_OUTPUTS = _s.N_TEST_OUTPUTS
TEST_OUTPUT_MIN_SOURCES = _s.TEST_OUTPUT_MIN_SOURCES
TEST_OUTPUT_MAX_SOURCES = _s.TEST_OUTPUT_MAX_SOURCES


# ---------------------------------------------------------------------------
# Pipeline hyperparameters
# ---------------------------------------------------------------------------

TRIAGE_TOP_K = _s.TRIAGE_TOP_K
CLUSTER_DISTANCE_THRESHOLD = _s.CLUSTER_DISTANCE_THRESHOLD
SHAPLEY_VALUE_TEMPERATURE = _s.SHAPLEY_VALUE_TEMPERATURE
SHAPLEY_MAX_EXACT_N = _s.SHAPLEY_MAX_EXACT_N
SHAPLEY_MC_PERMUTATIONS = _s.SHAPLEY_MC_PERMUTATIONS
MIN_CLUSTER_WEIGHT = _s.MIN_CLUSTER_WEIGHT
MIN_FINAL_WEIGHT = _s.MIN_FINAL_WEIGHT
OWEN_OR_DECOMP = _s.OWEN_OR_DECOMP
USE_DEMUCS_SEPARATION = _s.USE_DEMUCS_SEPARATION
DEMUCS_MODEL = _s.DEMUCS_MODEL
CLUSTERING_METHOD = _s.CLUSTERING_METHOD
USE_BPM_KEY_FILTER = _s.USE_BPM_KEY_FILTER
BPM_FILTER_TOLERANCE = _s.BPM_FILTER_TOLERANCE
ENABLE_CURRENCIES = _s.ENABLE_CURRENCIES
CURRENCY_ALPHA = _s.CURRENCY_ALPHA
CURRENCY_BETA = _s.CURRENCY_BETA
CURRENCY_GAMMA = _s.CURRENCY_GAMMA

INTERVAL_PRECISION_BITS = _s.INTERVAL_PRECISION_BITS
HOEFFDING_CONFIDENCE = _s.HOEFFDING_CONFIDENCE

EFFICIENCY_TOLERANCE = _s.EFFICIENCY_TOLERANCE
SYMMETRY_TOLERANCE = _s.SYMMETRY_TOLERANCE
DUMMY_TOLERANCE = _s.DUMMY_TOLERANCE
ADDITIVITY_TOLERANCE = _s.ADDITIVITY_TOLERANCE


# ---------------------------------------------------------------------------
# Device + reproducibility
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = select_device()
GLOBAL_SEED = _s.GLOBAL_SEED


def ensure_dirs() -> None:
    for d in [
        DATA_DIR, CATALOG_DIR, TEST_OUTPUTS_DIR, UPLOADS_DIR,
        MODELS_DIR, OUTPUTS_DIR, RECEIPTS_DIR, PROOFS_DIR, PLOTS_DIR, LOGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
