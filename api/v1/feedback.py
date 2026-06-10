"""POST /feedback — label collection for retraining."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_MIN_CONFIDENCE = {
    "analyst":   0.80,
    "auto_lock": 0.70,
    "temporal":  0.70,
    "auto_mfa":  0.60,
}

_VALID_SOURCES = set(_MIN_CONFIDENCE.keys())


class FeedbackEvent(BaseModel):
    event_id: str
    true_label: int = Field(..., ge=0, le=1)
    label_source: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    event_at: datetime | None = None


class FeedbackResult(BaseModel):
    accepted: bool
    reason: str


def _get_db():
    from main import db
    return db


@router.post("/feedback", response_model=FeedbackResult, tags=["Feedback"])
def submit_feedback(event: FeedbackEvent, db=Depends(_get_db)):
    """
    Submit a ground-truth label for a previously scored event.

    Enforces:
    - Valid label_source
    - Confidence >= minimum for source
    - label_delay_h computed server-side at confirmed_at time
    """
    if event.label_source not in _VALID_SOURCES:
        return FeedbackResult(
            accepted=False,
            reason=f"Unknown label_source '{event.label_source}'. "
                   f"Valid: {sorted(_VALID_SOURCES)}",
        )

    min_conf = _MIN_CONFIDENCE[event.label_source]
    if event.confidence < min_conf:
        return FeedbackResult(
            accepted=False,
            reason=f"confidence {event.confidence:.2f} below minimum "
                   f"for source {event.label_source} ({min_conf:.2f})",
        )

    confirmed_at = datetime.now(timezone.utc)

    try:
        db.execute(
            """INSERT INTO feedback
               (feedback_id, event_id, true_label, label_source, confidence,
                event_at, confirmed_at, incorporated)
               VALUES (?, ?, ?, ?, ?, ?, ?, FALSE)
               ON CONFLICT (event_id, confirmed_at) DO NOTHING
            """,
            [
                str(uuid.uuid4()),
                event.event_id,
                event.true_label,
                event.label_source,
                event.confidence,
                event.event_at.isoformat() if event.event_at else None,
                confirmed_at.isoformat(),
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {e}")

    return FeedbackResult(accepted=True, reason="Label accepted.")
