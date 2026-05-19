from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from cascade_v.config import RECEIPTS_DIR


router = APIRouter()


@router.get("/api/receipts")
def list_receipts():
    if not RECEIPTS_DIR.exists():
        return []
    items = []
    for p in sorted(RECEIPTS_DIR.glob("*.json")):
        try:
            with open(p) as f:
                r = json.load(f)
            top = r["per_creator"][0] if r.get("per_creator") else {}
            items.append({
                "receipt_id": r.get("receipt_id"),
                "created_at_utc": r.get("created_at_utc"),
                "overall_status": r.get("verification", {}).get("overall_status"),
                "top_creator": top.get("creator_name"),
                "top_creator_weight": top.get("weight_point"),
                "n_clusters": r.get("pipeline_metadata", {}).get("n_clusters"),
                "candidates_from_triage": r.get("pipeline_metadata", {}).get("candidates_from_triage"),
                "total_latency_ms": r.get("latency_ms", {}).get("total"),
            })
        except Exception:
            continue
    return items


@router.get("/api/receipts/{receipt_id}")
def get_receipt(receipt_id: str):
    p = RECEIPTS_DIR / f"{receipt_id}.json"
    if not p.exists():
        raise HTTPException(404, f"receipt {receipt_id} not found")
    with open(p) as f:
        return json.load(f)
