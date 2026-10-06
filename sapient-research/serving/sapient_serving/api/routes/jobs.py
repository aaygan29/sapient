"""GET /v1/jobs/{job_id} — poll status / fetch result (auth-gated, owner-only)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...schemas import JobStatus
from ...security.auth import current_client
from ...security.keys import Client
from ...storage.jobs import get_store
from .. import deps

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, client: Client = Depends(current_client)) -> JobStatus:
    rec = get_store().get(job_id, client.client_id)
    if rec is None:
        # Same 404 whether not-found, expired, or owned by someone else:
        # no existence leak, no cross-tenant (IDOR) read.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return JobStatus(
        job_id=rec.job_id,
        status=rec.status,
        model_version=deps.get_engine().model_version,
        created_at=rec.created_at,
        expires_at=rec.expires_at,
        error=rec.error,
        result=rec.result,
    )
