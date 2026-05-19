from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

import z3

from cascade_v.config import PROOFS_DIR, RECEIPTS_DIR
from server.app.services.interpret import annotate_proof


router = APIRouter()


def _load_receipt(receipt_id: str) -> dict | None:
    p = RECEIPTS_DIR / f"{receipt_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@router.get("/api/proofs/{receipt_id}")
def get_proof(receipt_id: str):
    p = PROOFS_DIR / f"{receipt_id}.smt2"
    if not p.exists():
        raise HTTPException(404, f"proof {receipt_id} not found")
    text = p.read_text()
    s = z3.Solver()
    try:
        s.from_string(text)
        verdict = str(s.check())
    except Exception as e:
        verdict = f"error: {e}"

    receipt = _load_receipt(receipt_id)
    annotated = annotate_proof(receipt, text) if receipt else None

    return {
        "receipt_id": receipt_id,
        "smt_lib": text,
        "annotated": annotated,
        "z3_verdict": verdict,
    }
