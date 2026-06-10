"""POST /explain — SHAP explanations. NOT in the hot path. Not SLA-bound."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ExplainRequest(BaseModel):
    event_id: str | None = None
    user_id: str = ""
    device_fingerprint: str = ""
    failed_attempts_7d: float = 0.0
    distinct_ips_7d: float = 0.0
    login_success_rate_30d: float = -1.0
    avg_login_hour_7d: float = -1.0
    account_age_days: float = 0.0
    is_cold_start: float = 0.0
    last_login_gap_h: float = -1.0
    geo_distance_delta: float = -1.0
    device_seen_flag: float = 0.0
    natural_language: bool = False


class FactorEntry(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class ExplainResult(BaseModel):
    event_id: str | None
    top_factors: list[FactorEntry]
    summary: str | None = None


@router.post("/explain", response_model=ExplainResult, tags=["Explainability"])
async def explain(req: ExplainRequest):
    """
    SHAP-based explanation for a login event.
    Permanently excluded from /predict hot path.
    Uses shap.TreeExplainer pre-loaded at model load time.
    """
    from core.inference.state import get_model_state
    import numpy as np

    state = get_model_state()
    if state is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    if state.explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available.")

    feature_names = state.transformer.feature_names_out
    event_dict = req.model_dump(exclude={"event_id", "natural_language"})
    vec = state.transformer.transform(event_dict)

    shap_values = state.explainer.shap_values(vec.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classification: class 1
    shap_row = shap_values[0] if shap_values.ndim == 2 else shap_values

    pairs = sorted(
        zip(feature_names, shap_row, vec),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    factors = [
        FactorEntry(
            feature=name,
            shap_value=round(float(sv), 4),
            feature_value=round(float(fv), 4),
            direction="increases_risk" if sv > 0 else "decreases_risk",
        )
        for name, sv, fv in pairs
    ]

    summary = None
    if req.natural_language:
        summary = _llm_summary(factors)

    return ExplainResult(event_id=req.event_id, top_factors=factors, summary=summary)


def _llm_summary(factors: list[FactorEntry]) -> str | None:
    """Optional LLM enrichment — async, never touches prediction path."""
    try:
        import anthropic
        top3 = ", ".join(
            f"{f.feature}={f.feature_value:.2f} ({f.direction.replace('_', ' ')})"
            for f in factors[:3]
        )
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": f"Explain this login risk in one sentence for an analyst: {top3}"}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None
