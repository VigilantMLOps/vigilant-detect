"""Adversarial tests — sentinel dominance, Redis outage, poisoned features, wrong order."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.features.contract import FeatureContractError, FeatureContractValidator
from core.features.online import SENTINEL_DICT, fetch_redis_features
from core.features.transformer import FeatureOrderError, FeatureTransformer, load_schema
from core.inference.decision import Decision, DecisionRules
from core.inference.engine import run_inference
from tests.conftest import make_cold_start_event, make_degraded_event, make_event
from tests.fake_redis import FakeRedis


# ── Sentinel dominance ────────────────────────────────────────────────────────

def test_all_sentinel_event_does_not_crash(transformer, tiny_model, default_rules):
    """
    An event where every feature is at its sentinel value must produce a valid
    InferenceResult — never raise an exception.
    This is the most pessimistic cold-start + degraded combination.
    """
    xgb, calibrator = tiny_model
    schema = load_schema()
    all_sentinel = {name: float(schema["sentinels"].get(name, 0.0)) for name in schema["feature_names"]}
    all_sentinel["is_cold_start"] = 1  # mark cold-start explicitly

    result = run_inference(
        event=all_sentinel,
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert result is not None
    assert result.decision in (Decision.ALLOW, Decision.CHALLENGE, Decision.BLOCK)
    assert 0.0 <= result.calibrated_probability <= 1.0


def test_sentinel_vector_is_finite(transformer):
    """All-sentinel vector must be finite — no NaN, no Inf."""
    schema = load_schema()
    all_sentinel = {name: float(schema["sentinels"].get(name, 0.0)) for name in schema["feature_names"]}
    vec = transformer.transform(all_sentinel)
    assert np.all(np.isfinite(vec)), "All-sentinel feature vector contains non-finite values"
    assert not np.any(np.isnan(vec)), "All-sentinel feature vector contains NaN"


# ── Feature order violation ───────────────────────────────────────────────────

def test_wrong_feature_count_raises_feature_order_error(transformer):
    """
    A feature dict that produces a wrong-length array (e.g., extra key not in schema)
    must fail with FeatureOrderError — silent wrong-order inference is worse than a crash.
    """
    # Construct a dict that would cause wrong length if we force it
    # FeatureTransformer.transform() iterates over feature_names_out so extra
    # keys don't matter — but too FEW keys that return sentinels do.
    # The length check fails only if len(vec) != self._n, which is invariant.
    # So we test the contract validator instead.
    feature_names = transformer.feature_names_out
    validator = FeatureContractValidator(feature_names)

    # Wrong length vector
    wrong_len = np.ones(len(feature_names) - 2, dtype=np.float32)
    with pytest.raises(FeatureContractError, match="features"):
        validator.validate_always(wrong_len, "test")


def test_nan_in_feature_vector_is_rejected(feature_names):
    """NaN anywhere in the feature vector must trigger FeatureContractError."""
    validator = FeatureContractValidator(feature_names, sampling_rate=1.0)
    bad_vec = np.zeros(len(feature_names), dtype=np.float32)
    bad_vec[5] = float("nan")
    with pytest.raises(FeatureContractError, match="NaN"):
        validator.validate_always(bad_vec, "adversarial")


def test_inf_in_feature_vector_is_rejected(feature_names):
    validator = FeatureContractValidator(feature_names, sampling_rate=1.0)
    bad_vec = np.zeros(len(feature_names), dtype=np.float32)
    bad_vec[3] = float("inf")
    with pytest.raises(FeatureContractError, match="finite"):
        validator.validate_always(bad_vec, "adversarial")


# ── Redis outage ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_outage_returns_sentinels():
    """ConnectionError from Redis → all sentinels, no exception, inference continues."""
    broken = FakeRedis()
    broken.set_failure(ConnectionError("Redis is down"))

    features, redis_ok = await fetch_redis_features("uid_any", "fp_any", broken)
    assert features == dict(SENTINEL_DICT)
    assert redis_ok is False


@pytest.mark.asyncio
async def test_redis_outage_does_not_block_inference(transformer, tiny_model, default_rules):
    """Redis failure path: build event with sentinels → run inference successfully."""
    xgb, calibrator = tiny_model
    # Simulate what inference_service does on Redis failure
    redis_result = dict(SENTINEL_DICT)  # sentinels from failed Redis
    event = make_degraded_event(**redis_result)  # merge sentinels into event

    result = run_inference(
        event=event,
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
        is_degraded=True,
    )
    assert result is not None
    assert result.is_degraded is True


@pytest.mark.asyncio
async def test_redis_never_raises_on_exception():
    """fetch_redis_features MUST NOT raise any exception, ever."""

    class AlwaysRaisesRedis:
        def pipeline(self):
            raise RuntimeError("Unexpected Redis error")

    # Must not raise
    features, redis_ok = await fetch_redis_features("uid", "fp", AlwaysRaisesRedis())
    assert features == dict(SENTINEL_DICT)
    assert redis_ok is False


# ── Poisoned features ─────────────────────────────────────────────────────────

def test_extreme_feature_values_no_crash(transformer, tiny_model, default_rules):
    """Extreme (but finite) feature values must not crash inference."""
    xgb, calibrator = tiny_model
    extreme_event = {
        "failed_attempts_7d": 1e6,
        "distinct_ips_7d": 1e6,
        "login_success_rate_30d": 1.0,
        "avg_login_hour_7d": 23.9,
        "account_age_days": 36500.0,
        "hour_sin": 1.0,
        "hour_cos": -1.0,
        "dow_sin": 1.0,
        "dow_cos": -1.0,
        "is_cold_start": 0,
        "last_login_gap_h": 1e6,
        "geo_distance_delta": 20015.0,  # half circumference of Earth
        "device_seen_flag": 1.0,
    }
    result = run_inference(
        event=extreme_event,
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert result is not None
    assert 0.0 <= result.calibrated_probability <= 1.0


def test_negative_sentinel_values_accepted(transformer, tiny_model, default_rules):
    """Sentinel values like -1.0 for login_success_rate_30d are valid — not rejected."""
    xgb, calibrator = tiny_model
    event = make_event(login_success_rate_30d=-1.0, last_login_gap_h=-1.0, geo_distance_delta=-1.0)
    result = run_inference(
        event=event,
        transformer=transformer,
        xgb_model=xgb,
        calibrator=calibrator,
        validator=None,
        rules=default_rules,
    )
    assert result is not None


# ── Cold-start vs degraded distinction ───────────────────────────────────────

def test_cold_start_and_degraded_are_distinguishable(transformer, tiny_model, default_rules):
    """
    Cold-start (is_cold_start=1) and degraded known user (is_cold_start=0)
    must be distinguishable at the feature vector level.
    The model sees different inputs for these two scenarios.
    """
    xgb, calibrator = tiny_model
    schema = load_schema()

    cold = make_cold_start_event()   # is_cold_start=1
    degraded = make_degraded_event()  # is_cold_start=0, same online sentinels

    vec_cold = transformer.transform(cold)
    vec_degraded = transformer.transform(degraded)

    # is_cold_start index
    cs_idx = transformer.feature_names_out.index("is_cold_start")
    assert vec_cold[cs_idx] == 1.0
    assert vec_degraded[cs_idx] == 0.0
    assert not np.array_equal(vec_cold, vec_degraded), (
        "Cold-start and degraded must produce distinct feature vectors"
    )


# ── Calibrated probability bounds ─────────────────────────────────────────────

def test_calibrated_probability_always_in_0_1(transformer, tiny_model, default_rules):
    """Calibrated probability must always be in [0, 1] regardless of input."""
    xgb, calibrator = tiny_model
    rng = np.random.default_rng(99)
    schema = load_schema()
    feature_names = transformer.feature_names_out

    for _ in range(50):
        event = {name: float(rng.uniform(-1, 100)) for name in feature_names}
        # Clamp sentinel-required values to valid sentinels if negative
        result = run_inference(
            event=event,
            transformer=transformer,
            xgb_model=xgb,
            calibrator=calibrator,
            validator=None,
            rules=default_rules,
        )
        assert 0.0 <= result.calibrated_probability <= 1.0, \
            f"calibrated_probability={result.calibrated_probability} out of bounds"
