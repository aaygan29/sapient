"""POST /v1/encode — submit a stimulus, get a job id back.

Auth + rate limit enforced before any bytes are read. Uploads are validated
(type + size cap), processed in a background task, and the raw bytes are dropped
immediately after (data minimization — nothing persisted).
"""
from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from ...engine.base import Stimulus
from ...schemas import EncodeAccepted
from ...security.keys import Client
from ...security.ratelimit import enforce_rate_limit
from ...settings import settings
from ...storage.jobs import get_store
from .. import deps

router = APIRouter()


async def _read_capped(upload: UploadFile, cap: int) -> bytes:
    data = await upload.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
        )
    return data


@router.post("/encode", response_model=EncodeAccepted, status_code=status.HTTP_202_ACCEPTED)
async def encode(
    background: BackgroundTasks,
    client: Client = Depends(enforce_rate_limit),  # depends -> current_client -> rate limit
    video: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
    transcript: str | None = Form(default=None),
    subject_idx: int = Form(default=0),
    include_full_parcels: bool = Form(default=False),
) -> EncodeAccepted:
    if video is None and audio is None and not transcript:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one of: video, audio, transcript.",
        )
    if video is not None and not (video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail="`video` must be a video/* file.")
    if audio is not None and not (audio.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=415, detail="`audio` must be an audio/* file.")

    cap = settings.max_upload_bytes
    video_bytes = await _read_capped(video, cap) if video is not None else None
    audio_bytes = await _read_capped(audio, cap) if audio is not None else None
    filename = (video.filename if video else None) or (audio.filename if audio else None)

    # Full 1000-dim vector only if this client is explicitly allowed (IP control).
    allow_full = bool(include_full_parcels) and client.allow_full_parcels

    stim = Stimulus(
        video_bytes=video_bytes,
        audio_bytes=audio_bytes,
        transcript=transcript,
        filename=filename,
        subject_idx=subject_idx,
    )
    rec = get_store().create(client.client_id)
    background.add_task(_process, rec.job_id, stim, allow_full)
    return EncodeAccepted(job_id=rec.job_id, status="queued")


def _process(job_id: str, stim: Stimulus, include_full_parcels: bool) -> None:
    store = get_store()
    store.set_processing(job_id)
    try:
        pred = deps.get_engine().predict(stim, include_full_parcels=include_full_parcels)
        store.set_result(job_id, pred)
    except Exception as exc:  # noqa: BLE001 - surface a safe error string, not internals
        store.set_error(job_id, f"{type(exc).__name__}: {exc}")
    finally:
        # data minimization: drop raw media ASAP (defensive; stim is discarded anyway)
        stim.video_bytes = None
        stim.audio_bytes = None
