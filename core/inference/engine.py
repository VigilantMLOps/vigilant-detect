"""Stateless inference engine — transform → predict → calibrate → decide."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.features.transformer import FeatureTransformer
from core.inference.decision import (
    ContextFlags,
    Decision,
    DecisionRules,
    active_context_flags,
    confidence_label,
    decide,
)
from core.models.calibrator import calibrate


@dataclass
class InferenceResult:
    raw_probability: float
    calibrated_probability: float
    decision: Decision
    context_flags: list[str]
    confidence: str
    is_degraded: bool
    is_cold_start: bool
    model_version: str


def run_inference(
    event: dict,
    transformer: FeatureTransformer,
    xgb_model,
    calibrator,
    validator,
    rules: DecisionRules,
    is_degraded: bool = False,
    model_version: str = "unknown",
) -> InferenceResult:
    """
    Stateless inference: event dict → InferenceResult.

    Steps 4-8 of the hot path — called from inference_service after
    Redis features have been merged into the event dict.

    No I/O. No branching on training vs inference mode.
    """
    # Step 4: Assemble feature vector (enforces schema order)
    vec = transformer.transform(event)

    # Step 5: FeatureContractValidator (sampling mode — does not raise in prod)
    if validator is not None:
        validator.validate_sampled(vec, "inference")

    # Step 6: XGBoost raw probability
    raw_prob = float(xgb_model.predict_proba(vec.reshape(1, -1))[0, 1])

    # Step 7: Isotonic calibration
    cal_prob = calibrate(calibrator, raw_prob)

    # Step 8: Decision
    flags = ContextFlags(
        device_seen_flag=float(event.get("device_seen_flag", 0.0)),
        geo_distance_delta=float(event.get("geo_distance_delta", -1.0)),
        last_login_gap_h=float(event.get("last_login_gap_h", -1.0)),
        is_cold_start=int(event.get("is_cold_start", 0)),
    )
    decision = decide(cal_prob, flags, rules)
    active = active_context_flags(flags, rules)
    confidence = confidence_label(cal_prob, rules)
    is_cold_start = bool(int(event.get("is_cold_start", 0)))

    return InferenceResult(
        raw_probability=raw_prob,
        calibrated_probability=cal_prob,
        decision=decision,
        context_flags=active,
        confidence=confidence,
        is_degraded=is_degraded,
        is_cold_start=is_cold_start,
        model_version=model_version,
    )
