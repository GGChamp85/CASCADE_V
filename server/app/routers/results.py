from __future__ import annotations

import csv

from fastapi import APIRouter, HTTPException

from cascade_v.config import OUTPUTS_DIR


router = APIRouter()


@router.get("/api/results")
def results():
    p = OUTPUTS_DIR / "results.csv"
    if not p.exists():
        raise HTTPException(404, "results.csv not found - run cascade-evaluate")
    rows = []
    with open(p) as f:
        for row in csv.DictReader(f):
            for k in ("instance_mae", "creator_mae", "coverage_at_k"):
                if k in row:
                    row[k] = float(row[k])
            for k in ("axioms_proven", "axioms_total"):
                if k in row:
                    row[k] = int(row[k])
            row["top1_hit"] = row.get("top1_hit") in ("True", "true", True)
            row["is_dna_case"] = row.get("is_dna_case") in ("True", "true", True)
            rows.append(row)
    return rows
