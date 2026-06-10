"""ModelState — immutable snapshot + thread-safe hot-swap."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from core.features.contract import FeatureContractValidator, FeatureContractError
from core.features.transformer import FeatureTransformer
from core.models.calibrator import calibrate, check_calibrator_monotonicity
from core.models.registry import load_model_artifacts
from core.logger import get_logger

_logger = get_logger("vigilant-detect.state")


class ModelLoadError(Exception):
    """Raised when model validation fails during load. Current model keeps serving."""


@dataclass(frozen=True)
class ModelState:
    xgb: XGBClassifier
    calibrator: IsotonicRegression
    transformer: FeatureTransformer
    explainer: Any  # shap.TreeExplainer — loaded once, used by /explain only
    metadata: dict
    validator: FeatureContractValidator


# Module-level state — pinned per-request via `state = _model_state`
_model_state: ModelState | None = None
_swap_lock = threading.Lock()


def get_model_state() -> ModelState | None:
    return _model_state


def load_and_validate_model(model_id: str, hot_path_validation_rate: float = 0.05) -> ModelState:
    """
    Fully construct and validate a new ModelState BEFORE any swap.
    Current production model serves 100% of traffic during this function.

    Validation steps:
    1. Deserialize all artifacts (raises on corrupt pkl)
    2. FeatureContractValidator: artifact feature_names vs schema
    3. Golden vector smoke test: 3 fixed inputs, verify shape + no NaN
    4. Warm-up predict_proba call
    5. Calibrator monotonicity: predict([0.0, 0.5, 1.0]) must be non-decreasing

    Raises ModelLoadError if any step fails — current model keeps serving.
    """
    _logger.info("Loading model artifacts for {}", model_id)

    # Step 1: Deserialize
    try:
        xgb_model, calibrator, transformer, metadata = load_model_artifacts(model_id)
    except Exception as e:
        raise ModelLoadError(f"Artifact deserialization failed for {model_id}: {e}") from e

    # Step 2: Feature contract
    feature_names = transformer.feature_names_out
    validator = FeatureContractValidator(feature_names, sampling_rate=hot_path_validation_rate)
    try:
        smoke_vec = np.zeros((1, len(feature_names)), dtype=np.float32)
        validator.validate_always(smoke_vec, "model_load_contract_check")
    except FeatureContractError as e:
        raise ModelLoadError(f"Feature contract check failed: {e}") from e

    # Step 3: Golden vector smoke test
    _run_smoke_test(xgb_model, calibrator, transformer, feature_names)

    # Step 4: Warm-up predict_proba
    try:
        warm_vec = np.zeros((1, len(feature_names)), dtype=np.float32)
        raw = xgb_model.predict_proba(warm_vec)
        assert raw.shape == (1, 2), f"Unexpected predict_proba shape: {raw.shape}"
    except Exception as e:
        raise ModelLoadError(f"Warm-up predict_proba failed: {e}") from e

    # Step 5: Calibrator monotonicity
    if not check_calibrator_monotonicity(calibrator):
        raise ModelLoadError("Calibrator monotonicity check failed.")

    # Load SHAP explainer (used only by /explain — never in hot path)
    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        _logger.info("SHAP TreeExplainer loaded for {}", model_id)
    except Exception as e:
        _logger.warning("SHAP explainer failed to load (non-fatal): {}", e)

    state = ModelState(
        xgb=xgb_model,
        calibrator=calibrator,
        transformer=transformer,
        explainer=explainer,
        metadata=metadata,
        validator=validator,
    )
    _logger.info("Model {} validated and ready for swap.", model_id)
    return state


def _run_smoke_test(xgb, calibrator, transformer, feature_names: list[str]) -> None:
    """Run 3 synthetic input vectors through the full pipeline. Hard failure on any issue."""
    test_cases = [
        {name: 0.0 for name in feature_names},
        {name: -1.0 for name in feature_names},
        {**{name: 0.5 for name in feature_names}, "device_seen_flag": 1.0, "is_cold_start": 0.0},
    ]
    for i, event in enumerate(test_cases):
        vec = transformer.transform(event)
        if vec.shape != (len(feature_names),):
            raise ModelLoadError(f"Smoke test {i}: wrong shape {vec.shape}")
        if np.any(np.isnan(vec)):
            raise ModelLoadError(f"Smoke test {i}: NaN in output")
        raw = xgb.predict_proba(vec.reshape(1, -1))[0, 1]
        cal = calibrate(calibrator, float(raw))
        if not (0.0 <= cal <= 1.0):
            raise ModelLoadError(f"Smoke test {i}: calibrated probability {cal} out of [0,1]")


def swap_model(new_state: ModelState) -> None:
    """
    Atomic pointer replacement — lock held for one assignment (microseconds).
    In-flight requests that already pinned the old state complete normally.
    """
    global _model_state
    with _swap_lock:
        _model_state = new_state
    _logger.info("Model swapped to {}", new_state.metadata.get("model_id", "unknown"))
