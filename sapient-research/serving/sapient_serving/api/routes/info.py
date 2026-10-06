"""GET /v1/info — honest provenance (auth-gated)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...engine.provenance import info as provenance_info
from ...schemas import InfoResponse
from ...security.auth import current_client
from ...security.keys import Client

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
def get_info(client: Client = Depends(current_client)) -> InfoResponse:
    return provenance_info()
