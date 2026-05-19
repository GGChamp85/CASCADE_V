"""Lazy singletons for shared resources (encoder, embeddings, metadata)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from cascade_v.config import (
    CATALOG_EMBEDDINGS_PATH,
    CATALOG_METADATA_PATH,
)
from cascade_v.embeddings import load_catalog_embeddings
from cascade_v.train import load_encoder
from cascade_v.utils.synth import load_catalog_metadata


@lru_cache(maxsize=1)
def get_encoder():
    return load_encoder()


@lru_cache(maxsize=1)
def get_catalog_embeddings() -> np.ndarray:
    return load_catalog_embeddings(CATALOG_EMBEDDINGS_PATH)


@lru_cache(maxsize=1)
def get_catalog():
    creators, sources = load_catalog_metadata(CATALOG_METADATA_PATH)
    catalog_ids = [s["source_id"] for s in sources]
    source_to_creator = {s["source_id"]: s["creator_id"] for s in sources}
    creator_names = {c["creator_id"]: c["name"] for c in creators}
    return {
        "creators": creators,
        "sources": sources,
        "catalog_ids": catalog_ids,
        "source_to_creator": source_to_creator,
        "creator_names": creator_names,
    }
