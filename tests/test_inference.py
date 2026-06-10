"""Tests for inference engine — correctness, degraded mode, warm latency."""
from __future__ import annotations

import time

import numpy as np
import pytest

from core.features.contract import FeatureContractValidator
from core.features.transformer import FeatureTransformer
from core.inference.decision import Decision, DecisionRules
from core.inference.engine import InferenceResult, run_inference
from tests.conftest import make_cold_start_event, make_degraded_event, make_event


# ── basic correctness ─────────────────────────────────────────────────────────

def test_run_inference_returns_result(transformer, tiny_model, default_rules):
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert isinstance(result, InferenceResult)
    assert result.decision in (Decision.ALLOW, Decision.CHALLENGE, Decision.BLOCK)
    assert 0.0 <= result.calibrated_probability <= 1.0
    assert 0.0 <= result.raw_probability <= 1.0
    assert result.confidence in ("low", "medium", "high")


def test_run_inference_degraded_mode(transformer, tiny_model, default_rules):
    """Degraded mode flag must pass through to result."""
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_degraded_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
        is_degraded=True,
    )
    assert result.is_degraded is True
    assert result.decision in (Decision.ALLOW, Decision.CHALLENGE, Decision.BLOCK)


def test_run_inference_cold_start_flag(transformer, tiny_model, default_rules):
    """is_cold_start from event must propagate to InferenceResult."""
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_cold_start_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert result.is_cold_start is True


def test_run_inference_known_user_not_cold_start(transformer, tiny_model, default_rules):
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_event(is_cold_start=0),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert result.is_cold_start is False


def test_model_version_passes_through(transformer, tiny_model, default_rules):
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
        model_version="v42",
    )
    assert result.model_version == "v42"


def test_calibrated_probability_is_clipped(transformer, tiny_model, default_rules):
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert 0.0 <= result.calibrated_probability <= 1.0


def test_block_decision_for_high_prob(transformer, tiny_model):
    """A model forced to output high probability must trigger BLOCK."""
    xgb, calibrator = tiny_model
    # Use very tight thresholds to guarantee BLOCK
    tight_rules = DecisionRules(
        threshold_challenge=0.0,
        threshold_block=0.0,
        geo_anomaly_km=0.0,
        dormancy_hours=0.0,
    )
    result = run_inference(
        event=make_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=tight_rules,
    )
    assert result.decision == Decision.BLOCK


def test_allow_decision_for_permissive_rules(transformer, tiny_model):
    """Very permissive rules must ALLOW even with elevated probability."""
    xgb, calibrator = tiny_model
    permissive_rules = DecisionRules(
        threshold_challenge=1.1,   # impossible threshold
        threshold_block=1.1,
        geo_anomaly_km=99999.0,
        dormancy_hours=99999.0,
    )
    result = run_inference(
        event=make_event(device_seen_flag=1.0, geo_distance_delta=1.0, last_login_gap_h=1.0),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=permissive_rules,
    )
    assert result.decision == Decision.ALLOW


# ── validator integration ─────────────────────────────────────────────────────

def test_contract_validator_called_when_provided(transformer, tiny_model, default_rules, feature_names):
    """With sampling_rate=1.0, validator must always run on hot path."""
    xgb, calibrator = tiny_model
    validator = FeatureContractValidator(feature_names, sampling_rate=1.0)
    result = run_inference(
        event=make_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=validator,
        rules=default_rules,
    )
    assert result is not None


def test_validator_none_does_not_crash(transformer, tiny_model, default_rules):
    xgb, calibrator = tiny_model
    result = run_inference(
        event=make_event(),
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert result is not None


# ── warm steady-state latency ─────────────────────────────────────────────────

def test_warm_inference_latency_p95(transformer, tiny_model, default_rules):
    """
    100 sequential calls to run_inference (no I/O) must complete with
    P95 << 50ms. With 5-estimator model and no Redis, this should be < 5ms each.
    Asserts P95 < 20ms as a conservative guard.
    """
    xgb, calibrator = tiny_model
    event = make_event()
    latencies_ms = []

    # Warm up
    for _ in range(10):
        run_inference(
            event=event,
            transformer=transformer,
            xgb_model=xgb,
            calibrator=calibrator,
            validator=None,
            rules=default_rules,
        )

    for _ in range(100):
        t0 = time.perf_counter()
        run_inference(
            event=event,
            transformer=transformer,
            xgb_model=xgb,
            calibrator=calibrator,
            validator=None,
            rules=default_rules,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    p50 = latencies_ms[49]
    p95 = latencies_ms[94]

    # These are deliberately conservative — real target is P95 < 50ms.
    assert p50 < 5.0, f"P50 latency {p50:.2f}ms exceeds 5ms for in-process inference"
    assert p95 < 20.0, f"P95 latency {p95:.2f}ms exceeds 20ms for in-process inference"
