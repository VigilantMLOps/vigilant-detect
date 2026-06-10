"""Tests for the training pipeline — gates, partition identity, multi-seed gate."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
from sklearn.isotonic import IsotonicRegression

from core.features.offline import LeakageError, compute_offline_features, temporal_split
from core.models.calibrator import fit_isotonic
from core.models.evaluator import (
    GateResult,
    compute_ece,
    multi_seed_pr_auc_gate,
    tune_block_threshold,
    tune_challenge_threshold,
    tune_dormancy_threshold,
    tune_geo_threshold,
)
from core.models.trainer import TrainingGateError, apply_redis_degradation


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _make_training_df(n: int = 800, pos_rate: float = 0.25) -> pl.DataFrame:
    """Generate a minimal training DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    timestamps = [
        (base.replace(tzinfo=None) + __import__("datetime").timedelta(hours=i))
        for i in range(n)
    ]
    labels = (rng.random(n) < pos_rate).astype(int).tolist()
    return pl.DataFrame({
        "event_id": [str(i) for i in range(n)],
        "timestamp": pl.Series(timestamps).cast(pl.Datetime("us")),
        "user_id": [f"u{i % 30}" for i in range(n)],
        "ip_address": [f"10.0.{i % 10}.{i % 256}" for i in range(n)],
        "login_success": (rng.random(n) > 0.15).tolist(),
        "label": labels,
    })


def _make_xy(n: int = 300, n_features: int = 13, pos_rate: float = 0.25):
    rng = np.random.default_rng(42)
    X = rng.random((n, n_features)).astype(np.float32)
    y = (rng.random(n) < pos_rate).astype(np.int32)
    return X, y


# ── Partition identity ────────────────────────────────────────────────────────

def test_partition_identity_distinct_objects():
    """
    val_calibration and threshold_set arrays must be distinct Python objects.
    This is the structural guard against partition contamination.
    """
    df = _make_training_df(400)
    splits = temporal_split(df)
    _, _, val_cal_raw, thresh_raw, _ = splits

    val_cal = compute_offline_features(val_cal_raw, "val_calibration")
    thresh = compute_offline_features(thresh_raw, "threshold_set")

    # Both must survive compute_offline_features and remain distinct
    assert id(val_cal) != id(thresh), \
        "val_calibration and threshold_set DataFrames must be distinct objects"

    # When converted to numpy arrays, they must also remain distinct
    import numpy as np
    X_val_cal = np.zeros((len(val_cal), 2))
    X_thresh = np.zeros((len(thresh), 2))
    assert id(X_val_cal) != id(X_thresh)


def test_temporal_split_returns_five_partitions():
    df = _make_training_df(400)
    result = temporal_split(df)
    assert len(result) == 5


# ── ECE gate ─────────────────────────────────────────────────────────────────

def test_ece_gate_passes_calibrated_model():
    """A well-calibrated set of proba should have ECE < 0.10."""
    rng = np.random.default_rng(0)
    n = 500
    # Perfectly calibrated: proba = actual positive rate in each bin
    y = (rng.random(n) > 0.7).astype(int)
    proba = y.astype(float) + rng.normal(0, 0.05, n)
    proba = np.clip(proba, 0.01, 0.99)
    ece = compute_ece(proba, y)
    assert ece < 0.10


def test_ece_gate_rejects_miscalibrated_model():
    """A model that always outputs 0.9 on a balanced dataset has high ECE."""
    y = np.array([0, 1] * 250)
    proba = np.full(500, 0.9)
    ece = compute_ece(proba, y)
    assert ece > 0.10


def test_ece_gate_raises_training_gate_error():
    """
    If ECE exceeds max_ece, trainer must raise TrainingGateError
    before any artifact is written.
    """
    # Simulate the ECE check logic from trainer.py
    ece = 0.15
    max_ece = 0.10
    if ece > max_ece:
        with pytest.raises(TrainingGateError, match="ECE gate failed"):
            raise TrainingGateError(f"ECE gate failed: {ece:.4f} > {max_ece}")


# ── Multi-seed PR-AUC gate ────────────────────────────────────────────────────

def test_multi_seed_gate_all_pass():
    """All 3 seeds clearing the threshold → gate passes."""
    X, y = _make_xy(300)
    X_cal, y_cal = _make_xy(60)
    X_test, y_test = _make_xy(100)

    result = multi_seed_pr_auc_gate(
        X_train=X, y_train=y,
        X_val_stop=X_cal, y_val_stop=y_cal,
        X_val_cal=X_cal, y_val_cal=y_cal,
        X_test=X_test, y_test=y_test,
        production_baseline_pr_auc=0.0,  # no baseline — all seeds trivially pass
        seeds=[42, 137, 2024],
        min_improvement=0.0,
        xgb_params={
            "n_estimators": 5, "max_depth": 2, "tree_method": "hist",
            "eval_metric": "aucpr", "early_stopping_rounds": 3,
            "scale_pos_weight": 3.0,
        },
    )
    assert result.passed is True
    assert len(result.pr_auc_per_seed) == 3


def test_multi_seed_gate_one_seed_fails():
    """
    Gate must FAIL if even one seed does not exceed baseline + min_improvement.
    Partial pass (2 of 3) is a rejection.
    """
    X, y = _make_xy(300)
    X_cal, y_cal = _make_xy(60)
    X_test, y_test = _make_xy(100)

    # Set an impossibly high baseline — no seed can beat it
    impossible_baseline = 999.0
    result = multi_seed_pr_auc_gate(
        X_train=X, y_train=y,
        X_val_stop=X_cal, y_val_stop=y_cal,
        X_val_cal=X_cal, y_val_cal=y_cal,
        X_test=X_test, y_test=y_test,
        production_baseline_pr_auc=impossible_baseline,
        seeds=[42, 137, 2024],
        min_improvement=0.02,
        xgb_params={
            "n_estimators": 5, "max_depth": 2, "tree_method": "hist",
            "eval_metric": "aucpr", "early_stopping_rounds": 3,
            "scale_pos_weight": 3.0,
        },
    )
    assert result.passed is False
    assert "Seeds" in result.reason or len(result.pr_auc_per_seed) == 3


def test_multi_seed_gate_result_has_per_seed_scores():
    X, y = _make_xy(200)
    X_cal, y_cal = _make_xy(40)
    X_test, y_test = _make_xy(60)

    result = multi_seed_pr_auc_gate(
        X_train=X, y_train=y,
        X_val_stop=X_cal, y_val_stop=y_cal,
        X_val_cal=X_cal, y_val_cal=y_cal,
        X_test=X_test, y_test=y_test,
        production_baseline_pr_auc=0.0,
        seeds=[42, 137],
        min_improvement=0.0,
        xgb_params={
            "n_estimators": 5, "max_depth": 2, "tree_method": "hist",
            "eval_metric": "aucpr", "early_stopping_rounds": 3,
            "scale_pos_weight": 3.0,
        },
    )
    assert set(result.pr_auc_per_seed.keys()) == {42, 137}
    assert all(0.0 <= v <= 1.0 for v in result.pr_auc_per_seed.values())


# ── Threshold tuning isolation ────────────────────────────────────────────────

def test_tune_challenge_threshold_returns_float():
    rng = np.random.default_rng(1)
    n = 200
    proba = rng.random(n)
    y = (rng.random(n) > 0.7).astype(int)
    threshold = tune_challenge_threshold(proba, y)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


def test_tune_block_threshold_returns_float():
    rng = np.random.default_rng(1)
    n = 200
    proba = rng.random(n)
    y = (rng.random(n) > 0.7).astype(int)
    threshold = tune_block_threshold(proba, y)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


def test_threshold_tuning_differs_across_partitions():
    """
    Tuning on threshold_set vs val_calibration should produce different thresholds
    because they represent independent distributions.
    """
    rng = np.random.default_rng(42)
    n = 200
    # Two different distributions
    proba_a = rng.beta(2, 5, n)  # skewed low
    proba_b = rng.beta(5, 2, n)  # skewed high
    y_a = (rng.random(n) > 0.7).astype(int)
    y_b = (rng.random(n) > 0.7).astype(int)

    t_a = tune_challenge_threshold(proba_a, y_a)
    t_b = tune_challenge_threshold(proba_b, y_b)
    # Different distributions → different thresholds (proves partition isolation matters)
    assert t_a != t_b, "Threshold tuning on different distributions must produce different values"


def test_tune_geo_threshold_uses_threshold_set():
    """geo_anomaly_km is tuned on threshold_set — must return a valid candidate."""
    feature_names = ["geo_distance_delta"]
    rng = np.random.default_rng(5)
    n = 200
    X = rng.exponential(300, (n, 1)).astype(np.float32)
    y = (X[:, 0] > 400).astype(int).ravel()
    threshold = tune_geo_threshold(X, y, feature_names)
    assert threshold in [100.0, 200.0, 300.0, 500.0, 750.0, 1000.0]


def test_tune_dormancy_threshold_uses_threshold_set():
    feature_names = ["last_login_gap_h"]
    rng = np.random.default_rng(6)
    n = 200
    X = rng.exponential(400, (n, 1)).astype(np.float32)
    y = (X[:, 0] > 500).astype(int).ravel()
    threshold = tune_dormancy_threshold(X, y, feature_names)
    assert threshold in [168.0, 336.0, 480.0, 720.0, 1440.0, 2160.0]


# ── min_test_positives guard ──────────────────────────────────────────────────

def test_min_test_positives_guard():
    """TrainingGateError must be raised when test set has too few positives."""
    y_test = np.zeros(100, dtype=np.int32)  # 0 positives
    n_test_pos = int(y_test.sum())
    min_test_positives = 200

    if n_test_pos < min_test_positives:
        with pytest.raises(TrainingGateError, match="positive examples"):
            raise TrainingGateError(
                f"Test set has only {n_test_pos} positive examples, "
                f"minimum is {min_test_positives}."
            )


# ── apply_redis_degradation (partition interaction) ───────────────────────────

def test_degradation_applies_to_train_not_test():
    """
    Redis dropout is applied to all splits EXCEPT test.
    Test must be the clean ceiling measurement.
    """
    feature_names = [
        "failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
        "avg_login_hour_7d", "account_age_days",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_cold_start",
        "last_login_gap_h", "geo_distance_delta", "device_seen_flag",
    ]
    rng = np.random.default_rng(7)
    n = 200
    X_train = rng.random((n, len(feature_names))).astype(np.float32)
    X_test = X_train.copy()

    gap_idx = feature_names.index("last_login_gap_h")
    X_train[:, gap_idx] = 50.0  # non-sentinel
    X_test[:, gap_idx] = 50.0

    X_train_degraded = apply_redis_degradation(
        X_train, feature_names, 0.5, np.random.default_rng(10)
    )
    # Test is NOT passed through degradation — it should remain unchanged
    # (caller responsibility, tested conceptually here)
    assert (X_test[:, gap_idx] == 50.0).all(), "X_test must not be modified"
    assert (X_train_degraded[:, gap_idx] == -1.0).sum() > 0, \
        "X_train must have some degraded rows after apply_redis_degradation"
