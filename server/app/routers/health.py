from __future__ import annotations

from fastapi import APIRouter

from cascade_v.config import DEVICE, RECEIPTS_DIR
from server.app.deps import get_catalog


router = APIRouter()


@router.get("/api/health")
def health():
    catalog = get_catalog()
    n_receipts = len(list(RECEIPTS_DIR.glob("*.json"))) if RECEIPTS_DIR.exists() else 0
    return {
        "status": "ok",
        "device": str(DEVICE),
        "catalog_size": len(catalog["catalog_ids"]),
        "n_creators": len(catalog["creators"]),
        "n_receipts": n_receipts,
    }
