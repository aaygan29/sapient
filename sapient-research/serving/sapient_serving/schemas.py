"""API request/response contract (the 'menu'). Shared by the server and the client shim."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JobState = Literal["queued", "processing", "done", "error"]


class ParcelHighlight(BaseModel):
    parcel_id: int
    name: str
    value: float = Field(description="Predicted mean activation for this parcel over the window.")


class ParcelSummary(BaseModel):
    n_parcels: int = 1000
    mean: float
    std: float
    top: list[ParcelHighlight] = Field(description="Highest-activation parcels (coarse highlight).")
    values: Optional[list[float]] = Field(
        default=None,
        description="Full 1000-dim parcel vector. Withheld by default; only for clients with allow_full_parcels.",
    )


class ConstructScore(BaseModel):
    key: str
    label: str
    score: float = Field(description="0–100 strength of this construct over the clip.")
    networks: list[str] = Field(description="Brain networks this construct is read from.")
    description: str


class PeakMoment(BaseModel):
    t_seconds: float
    purchase_intent: float
    label: str


class SignalTimeline(BaseModel):
    seconds: list[float]
    purchase_intent: list[float]
    attention: list[float]
    emotion: list[float]
    network_timeline: Optional[dict[str, list[float]]] = Field(
        default=None,
        description="Per-second Yeo-7 network activation (0-100), one list per network. "
        "Drives the live digital-brain visualization. None on legacy predictions.",
    )


class Signals(BaseModel):
    """Interpretable consumer-neuro layer derived from predicted cortical activation."""
    purchase_intent: float = Field(description="0–100 headline purchase-intent score.")
    recommendation: str = Field(description="Buy | Hold | Pass")
    confidence: float = Field(description="0–1 (signal stability).")
    constructs: list[ConstructScore]
    timeline: SignalTimeline
    peak_moments: list[PeakMoment]


class Prediction(BaseModel):
    roi_scores: dict[str, float] = Field(
        description="Predicted activation per ROI (network). NOTE: predicted activity, not a "
        "correlation-with-measured-fMRI score — see /v1/info."
    )
    parcels: ParcelSummary
    signals: Signals
    image_data_uri: str = Field(description="PNG (base64 data URI) summarizing the predicted response.")
    n_timepoints: int = Field(description="Number of 1 Hz fMRI timepoints predicted for this stimulus.")


class EncodeAccepted(BaseModel):
    job_id: str
    status: JobState
    poll_after_ms: int = 500


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    model_version: str
    created_at: float
    expires_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Prediction] = None


class InfoResponse(BaseModel):
    name: str
    served_model: str
    is_paper_model: bool
    provenance: dict
    output_space: dict
    metric_note: str
    licenses: dict
    notice: str
