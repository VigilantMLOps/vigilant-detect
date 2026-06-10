"""Shared fixtures for vigilant-detect test suite."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from tests.fake_database import FakeDatabase
from tests.fake_redis import FakeRedis
from core.features.transformer import load_schema, FeatureTransformer
from core.inference.decision import DecisionRules

SCHEMA_PATH = Path("core/features/schema.yaml")


def make_event(**overrides) -> dict:
    """Return a complete, valid feature event dict with realistic defaults."""
    base = {
        "failed_attempts_7d": 2.0,
        "distinct_ips_7d": 1.0,
        "login_success_rate_30d": 0.9,
        "avg_login_hour_7d": 10.0,
        "account_age_days": 365.0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "dow_sin": 0.433,
        "dow_cos": 0.901,
        "is_cold_start": 0,
        "last_login_gap_h": 24.0,
        "geo_distance_delta": 10.0,
        "device_seen_flag": 1.0,
    }
    base.update(overrides)
    return base


def make_cold_start_event(**overrides) -> dict:
    """Return a cold-start event (new user, no history)."""
    base = {
        "failed_attempts_7d": 0.0,    # cold-start sentinel
        "distinct_ips_7d": 0.0,
        "login_success_rate_30d": -1.0,
        "avg_login_hour_7d": -1.0,
        "account_age_days": 0.0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "dow_sin": 0.433,
        "dow_cos": 0.901,
        "is_cold_start": 1,
        "last_login_gap_h": -1.0,
        "geo_distance_delta": -1.0,
        "device_seen_flag": 0.0,
    }
    base.update(overrides)
    return base


def make_degraded_event(**overrides) -> dict:
    """Return a known-user event with Redis-degraded online features."""
    base = {
        "failed_attempts_7d": 3.0,
        "distinct_ips_7d": 2.0,
        "login_success_rate_30d": 0.8,
        "avg_login_hour_7d": 9.5,
        "account_age_days": 180.0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "dow_sin": 0.433,
        "dow_cos": 0.901,
        "is_cold_start": 0,          # NOT cold-start — just degraded
        "last_login_gap_h": -1.0,    # sentinel (Redis down)
        "geo_distance_delta": -1.0,  # sentinel
        "device_seen_flag": 0.0,     # sentinel
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="session")
def schema():
    return load_schema(SCHEMA_PATH)


@pytest.fixture(scope="session")
def transformer(schema):
    return FeatureTransformer(schema)


@pytest.fixture(scope="session")
def feature_names(transformer):
    return transformer.feature_names_out


@pytest.fixture
def fake_db():
    return FakeDatabase()


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture(scope="session")
def default_rules():
    return DecisionRules(
        threshold_challenge=0.35,
        threshold_block=0.75,
        geo_anomaly_km=500.0,
        dormancy_hours=720.0,
    )


@pytest.fixture(scope="session")
def tiny_model(feature_names):
    """
    Minimal XGBoost + calibrator for tests that need a callable model.
    5 estimators — fast, not accurate, but shape-correct.
    """
    n = len(feature_names)
    rng = np.random.default_rng(42)
    n_rows = 400
    X = rng.random((n_rows, n)).astype(np.float32)
    y = (rng.random(n_rows) > 0.7).astype(np.int32)

    model = XGBClassifier(
        n_estimators=5, max_depth=3, random_state=42, seed=42,
        tree_method="hist", scale_pos_weight=3.0,
    )
    model.fit(X, y)

    X_cal = rng.random((80, n)).astype(np.float32)
    y_cal = (rng.random(80) > 0.7).astype(np.int32)
    raw = model.predict_proba(X_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, y_cal)

    return model, calibrator


@pytest.fixture(scope="session")
def minimal_training_config():
    """Override training config for fast unit tests."""
    return {
        "gates": {
            "min_pr_auc_improvement": 0.02,
            "max_ece": 0.10,
            "pr_auc_rolling_window": 3,
            "min_test_positives": 10,  # lowered for small test datasets
        },
        "redis_simulation": {
            "dropout_rate": 0.15,
            "apply_to_splits": ["train", "val_stop", "val_calibration", "threshold_set"],
        },
        "multi_seed": {
            "seeds": [42, 137, 2024],
            "require_all_pass": True,
        },
        "xgboost": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 5,
        },
    }
