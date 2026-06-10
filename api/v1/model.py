"""GET /model/info and GET /health — model registry endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    loaded_at: str | None
    uptime_s: float
    degraded_mode: bool
    service: str = "vigilant-detect"


class ModelInfoResponse(BaseModel):
    model_id: str | None
    version: str | None
    pr_auc: float | None
    ece: float | None
    thresholds: dict | None
    context_rules: dict | None
    schema_hash: str | None
    feature_names_version: str | None
    trained_at: str | None


_start_time = datetime.now(timezone.utc)


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    from core.inference.state import get_model_state
    state = get_model_state()
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()
    return HealthResponse(
        status="healthy" if state is not None else "no_model",
        model_version=state.metadata.get("version") if state else None,
        loaded_at=state.metadata.get("saved_at") if state else None,
        uptime_s=round(uptime, 1),
        degraded_mode=state is None,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    from core.inference.state import get_model_state
    state = get_model_state()
    if state is None:
        return ModelInfoResponse(
            model_id=None, version=None, pr_auc=None, ece=None,
            thresholds=None, context_rules=None, schema_hash=None,
            feature_names_version=None, trained_at=None,
        )
    m = state.metadata
    return ModelInfoResponse(
        model_id=m.get("model_id"),
        version=m.get("version"),
        pr_auc=m.get("pr_auc"),
        ece=m.get("ece"),
        thresholds=m.get("thresholds"),
        context_rules=m.get("context_rules"),
        schema_hash=m.get("schema_hash"),
        feature_names_version=m.get("feature_names_version"),
        trained_at=m.get("trained_at"),
    )


@router.get("/model/versions", tags=["Model"])
def model_versions():
    """List all models in the registry with their status."""
    from main import db
    rows = db.fetchall(
        "SELECT model_id, status, pr_auc, ece, trained_at, schema_hash "
        "FROM ato_models ORDER BY trained_at DESC LIMIT 20"
    )
    return {"models": rows}
