"""Run a single attribution end-to-end. Reuses cascade_v.* helpers exclusively."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cascade_v.config import PROOFS_DIR, RECEIPTS_DIR
from cascade_v.embeddings import embed_audio
from cascade_v.pipeline import run_cascade_v
from cascade_v.receipts import make_receipt, save_receipt


def run_attribution(
    target_id: str,
    audio: np.ndarray,
    encoder,
    catalog_embeddings: np.ndarray,
    catalog: dict,
    total_payout: float = 1.0,
    on_event=None,
) -> dict:
    target_embedding = embed_audio(audio, encoder)
    result = run_cascade_v(
        target_id=target_id,
        target_embedding=target_embedding,
        catalog_embeddings=catalog_embeddings,
        catalog_ids=catalog["catalog_ids"],
        proofs_dir=PROOFS_DIR,
        raise_on_validation_failure=False,
        on_event=on_event,
    )
    receipt = make_receipt(
        result,
        source_to_creator=catalog["source_to_creator"],
        creator_names=catalog["creator_names"],
        total_payout_usd=total_payout,
    )
    save_receipt(receipt, RECEIPTS_DIR)
    return receipt
