"""POST /predict and POST /predict/batch — hot path inference endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class LoginEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str
    device_fingerprint: str
    ip_address: str = ""
    geo_country: str = ""
    geo_lat: float | None = None
    geo_lon: float | None = None
    login_success: bool = True
    mfa_used: bool = False
    mfa_method: str = "none"
    login_duration_ms: float = 0.0
    # Pre-computed offline features (optional — omit for cold-start users)
    failed_attempts_7d: float | None = None
    distinct_ips_7d: float | None = None
    login_success_rate_30d: float | None = None
    avg_login_hour_7d: float | None = None
    account_age_days: float | None = None
    # Online feature overrides (applied only when Redis has no prior state for this user)
    last_login_gap_h: float | None = None
    geo_distance_delta: float | None = None


class PredictionResult(BaseModel):
    event_id: str
    decision: str  # ALLOW | CHALLENGE | BLOCK
    risk_score: float
    calibrated_probability: float
    confidence: str
    context_flags: list[str]
    degraded: bool
    model_version: str


def _get_inference_service():
    from main import inference_service
    return inference_service


@router.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict(
    event: LoginEvent,
    svc=Depends(_get_inference_service),
):
    """
    Real-time ATO risk prediction.

    Steps 1-9 contain zero blocking I/O except step 3 (async Redis, 15ms cap).
    SHAP is NOT computed here — use POST /explain.
    """
    event_dict = event.model_dump()
    # Replace None offline features with schema sentinels
    from core.inference.state import get_model_state
    state = get_model_state()
    if state is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    for field in ["failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
                  "avg_login_hour_7d", "account_age_days"]:
        if event_dict.get(field) is None:
            event_dict[field] = state.transformer.sentinels.get(field, 0.0)

    result = await svc.predict(event_dict)
    return PredictionResult(**result)


@router.post("/predict/batch", response_model=list[PredictionResult], tags=["Inference"])
async def predict_batch(
    events: list[LoginEvent],
    svc=Depends(_get_inference_service),
):
    """Batch prediction — max 500 events per request."""
    if len(events) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500.")

    from core.inference.state import get_model_state
    state = get_model_state()
    if state is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    results = []
    for event in events:
        event_dict = event.model_dump()
        for field in ["failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
                      "avg_login_hour_7d", "account_age_days"]:
            if event_dict.get(field) is None:
                event_dict[field] = state.transformer.sentinels.get(field, 0.0)
        result = await svc.predict(event_dict)
        results.append(PredictionResult(**result))
    return results
