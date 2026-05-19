from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from cascade_v.config import GROUND_TRUTH_PATH
from server.app.deps import get_catalog


router = APIRouter()


@router.get("/api/catalog")
def catalog():
    c = get_catalog()
    return {"creators": c["creators"], "sources": c["sources"]}


@router.get("/api/test-outputs")
def test_outputs():
    if not GROUND_TRUTH_PATH.exists():
        raise HTTPException(404, "ground truth not found - run cascade-embed")
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    return gt


@router.get("/api/ground-truth/{output_id}")
def ground_truth_for(output_id: str):
    """Lookup ground truth for a single output_id (test outputs only)."""
    if not GROUND_TRUTH_PATH.exists():
        raise HTTPException(404, "ground truth not found")
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    record = next((r for r in gt if r["output_id"] == output_id), None)
    if record is None:
        raise HTTPException(404, f"no ground truth for {output_id}")

    # Annotate with creator names so the dashboard doesn't need to re-join
    cat = get_catalog()
    creator_names = cat["creator_names"]
    src_to_creator = cat["source_to_creator"]
    record["sources_annotated"] = [
        {
            "source_id": sid,
            "weight": float(w),
            "creator_id": src_to_creator.get(sid, "unknown"),
            "creator_name": creator_names.get(
                src_to_creator.get(sid, ""), "unknown"
            ),
        }
        for sid, w in zip(record["source_ids"], record["weights"])
    ]
    record["creators_annotated"] = sorted(
        [
            {
                "creator_id": cid,
                "creator_name": creator_names.get(cid, cid),
                "weight": float(w),
            }
            for cid, w in record["creator_weights"].items()
        ],
        key=lambda d: -d["weight"],
    )
    return record
