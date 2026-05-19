"""FastAPI entrypoint for CASCADE-V dashboard backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cascade_v.config import CASCADE_LOG_PATH, ensure_dirs
from cascade_v.logging_setup import configure_logging
from server.app.routers import (
    attribute,
    audio,
    catalog,
    health,
    proofs,
    receipts,
    results,
    training,
)


def create_app() -> FastAPI:
    ensure_dirs()
    configure_logging(CASCADE_LOG_PATH)

    app = FastAPI(title="CASCADE-V API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for r in (health, catalog, receipts, proofs, results, training, audio, attribute):
        app.include_router(r.router)

    return app


app = create_app()
