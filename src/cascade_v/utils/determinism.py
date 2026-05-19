"""
determinism.py — Centralized seed control.

`set_global_seeds(seed)` configures every RNG used by the pipeline so that
two runs with the same seed produce bit-identical outputs (within the
constraints of the underlying numerical libraries).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # local import so math-only code paths don't need torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
