"""Live attribution via Server-Sent Events.

POST /api/attribute/{output_id}      — run on a known test output
POST /api/attribute/upload            — multipart-upload a WAV and run

Both endpoints stream SSE events as the pipeline progresses. The final event
(`receipt.ready`) carries the full receipt JSON. The pipeline runs in a
worker thread; we use asyncio.Queue (with threadsafe injection) so SSE
streams don't burn thread-pool slots.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from cascade_v.config import TEST_OUTPUTS_DIR, UPLOADS_DIR
from cascade_v.utils.audio import load_wav
from server.app.deps import get_catalog, get_catalog_embeddings, get_encoder
from server.app.services.run import run_attribution


router = APIRouter()


def _stream_attribution(target_id: str, audio_path: Path, total_payout: float = 1.0):
    """Run the pipeline in a worker thread; yield SSE events as they fire."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def emit(name: str, payload: dict):
        loop.call_soon_threadsafe(
            q.put_nowait, {"event": name, "data": json.dumps(payload)}
        )

    def worker():
        try:
            audio = load_wav(audio_path)
            receipt = run_attribution(
                target_id=target_id,
                audio=audio,
                encoder=get_encoder(),
                catalog_embeddings=get_catalog_embeddings(),
                catalog=get_catalog(),
                total_payout=total_payout,
                on_event=emit,
            )
            loop.call_soon_threadsafe(
                q.put_nowait,
                {"event": "receipt.ready", "data": json.dumps(receipt)},
            )
        except Exception as e:
            loop.call_soon_threadsafe(
                q.put_nowait,
                {"event": "pipeline.error",
                 "data": json.dumps({"error": str(e), "type": type(e).__name__})},
            )
        finally:
            loop.call_soon_threadsafe(q.put_nowait, SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        while True:
            item = await q.get()
            if item is SENTINEL:
                return
            yield item

    return gen


@router.post("/api/attribute/{output_id}")
async def attribute_known(output_id: str, total_payout: float = 1.0):
    audio_path = TEST_OUTPUTS_DIR / f"{output_id}.wav"
    if not audio_path.exists():
        raise HTTPException(404, f"{output_id}.wav not found")
    gen = _stream_attribution(output_id, audio_path, total_payout)
    return EventSourceResponse(gen())


@router.post("/api/attribute/upload")
async def attribute_upload(file: UploadFile = File(...), total_payout: float = 1.0):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "only .wav uploads supported")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = f"upload_{uuid.uuid4().hex[:12]}"
    audio_path = UPLOADS_DIR / f"{upload_id}.wav"
    with open(audio_path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    gen = _stream_attribution(upload_id, audio_path, total_payout)
    return EventSourceResponse(gen())
