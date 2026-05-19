from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from cascade_v.config import TRAINING_LOG


router = APIRouter()


@router.get("/api/training")
def training():
    if not TRAINING_LOG.exists():
        raise HTTPException(404, "training log not found - run cascade-train")
    with open(TRAINING_LOG) as f:
        return json.load(f)
