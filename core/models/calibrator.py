"""Isotonic calibration wrapper — fits on val_calibration only."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier


def fit_isotonic(
    model: XGBClassifier,
    X_val_calibration: np.ndarray,
    y_val_calibration: np.ndarray,
) -> IsotonicRegression:
    """
    Fit IsotonicRegression on val_calibration.
    val_calibration must never have been seen by the early-stop optimizer.
    """
    raw_proba = model.predict_proba(X_val_calibration)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_proba, y_val_calibration)
    return calibrator


def calibrate(calibrator: IsotonicRegression, raw_proba: float) -> float:
    """Apply calibration to a single raw probability. Returns float in [0, 1]."""
    result = calibrator.predict(np.array([raw_proba]))[0]
    return float(np.clip(result, 0.0, 1.0))


def calibrate_batch(calibrator: IsotonicRegression, raw_proba: np.ndarray) -> np.ndarray:
    return np.clip(calibrator.predict(raw_proba), 0.0, 1.0)


def check_calibrator_monotonicity(calibrator: IsotonicRegression) -> bool:
    """Sanity check: predict([0.0, 0.5, 1.0]) must be non-decreasing."""
    test_inputs = np.array([0.0, 0.5, 1.0])
    outputs = calibrator.predict(test_inputs)
    return bool(outputs[0] <= outputs[1] <= outputs[2])
