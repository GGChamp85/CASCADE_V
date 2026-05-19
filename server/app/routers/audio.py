from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from cascade_v.config import CATALOG_DIR, TEST_OUTPUTS_DIR, UPLOADS_DIR


router = APIRouter()


@router.get("/api/audio/{kind}/{name}")
def audio(kind: str, name: str):
    if not name.endswith(".wav"):
        name = f"{name}.wav"

    if kind == "catalog":
        p = CATALOG_DIR / name
    elif kind == "test":
        p = TEST_OUTPUTS_DIR / name
    elif kind == "upload":
        p = UPLOADS_DIR / name
    else:
        raise HTTPException(400, "kind must be one of: catalog, test, upload")

    if not p.exists():
        raise HTTPException(404, f"audio not found: {kind}/{name}")
    return FileResponse(str(p), media_type="audio/wav", filename=name)
