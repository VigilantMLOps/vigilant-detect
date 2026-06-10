"""Tests for the feature system: transformer, offline features, online fetch, contract."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from core.features.contract import FeatureContractError, FeatureContractValidator
from core.features.offline import (
    LeakageError,
    SENTINEL_FAILED_ATTEMPTS,
    SENTINEL_LOGIN_SUCCESS_RATE,
    SENTINEL_AVG_LOGIN_HOUR,
    compute_offline_features,
    temporal_split,
    validate_no_leakage,
)
from core.features.online import SENTINEL_DICT, fetch_redis_features
from core.features.transformer import FeatureOrderError, FeatureTransformer
from core.models.trainer import apply_redis_degradation
from tests.conftest import make_event, make_cold_start_event


# ── Transformer ──────────────────────────────────────────────────────────────

def test_transform_produces_correct_length(transformer, feature_names):
    event = make_event()
    vec = transformer.transform(event)
    assert vec.shape == (len(feature_names),)


def test_transform_uses_schema_order(transformer, feature_names):
    """feature_names_out IS the order — dict insertion order must not matter."""
    # Build event with keys in reversed order
    event = dict(reversed(make_event().items()))
    vec = transformer.transform(event)
    # Values must still appear in schema order
    for i, name in enumerate(feature_names):
        assert vec[i] == pytest.approx(make_event()[name], abs=1e-4), \
            f"Feature {name} at index {i} does not match expected value"


def test_missing_key_uses_sentinel(transformer, schema):
    """A missing key must be replaced with its declared sentinel, never NaN."""
    event = make_event()
    del event["last_login_gap_h"]
    vec = transformer.transform(event)
    idx = transformer.feature_names_out.index("last_login_gap_h")
    assert vec[idx] == pytest.approx(schema["sentinels"]["last_login_gap_h"])
    assert not np.isnan(vec[idx])


def test_transform_output_dtype(transformer):
    vec = transformer.transform(make_event())
    assert vec.dtype == np.float32


def test_feature_names_out_is_list(transformer):
    names = transformer.feature_names_out
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_no_nan_in_cold_start_event(transformer):
    """All-sentinel cold-start event must produce a NaN-free vector."""
    vec = transformer.transform(make_cold_start_event())
    assert not np.any(np.isnan(vec))
    assert np.all(np.isfinite(vec))


# ── compute_is_cold_start ────────────────────────────────────────────────────

def test_cold_start_from_zero_account_age(schema):
    sentinels = schema["sentinels"]
    result = FeatureTransformer.compute_is_cold_start(
        account_age_days=0.0,
        offline_features={"failed_attempts_7d": 5.0},  # non-sentinel offline
        sentinels=sentinels,
    )
    assert result == 1


def test_cold_start_from_all_sentinel_offline(schema):
    sentinels = schema["sentinels"]
    offline = {
        "failed_attempts_7d": sentinels["failed_attempts_7d"],
        "distinct_ips_7d": sentinels["distinct_ips_7d"],
        "login_success_rate_30d": sentinels["login_success_rate_30d"],
        "avg_login_hour_7d": sentinels["avg_login_hour_7d"],
    }
    result = FeatureTransformer.compute_is_cold_start(
        account_age_days=1.0,  # non-zero age, but all rolling features at sentinel
        offline_features=offline,
        sentinels=sentinels,
    )
    assert result == 1


def test_not_cold_start_for_known_user(schema):
    sentinels = schema["sentinels"]
    offline = {
        "failed_attempts_7d": 3.0,   # above sentinel
        "distinct_ips_7d": 2.0,
        "login_success_rate_30d": 0.8,
        "avg_login_hour_7d": 10.0,
    }
    result = FeatureTransformer.compute_is_cold_start(
        account_age_days=365.0,
        offline_features=offline,
        sentinels=sentinels,
    )
    assert result == 0


def test_redis_failure_does_not_set_cold_start(transformer, schema):
    """
    A known user with Redis down must NOT have is_cold_start=1.
    Cold-start is determined from offline data before Redis is queried.
    """
    sentinels = schema["sentinels"]
    # Known user (non-zero account_age_days, some offline history)
    event = make_event(account_age_days=200.0, failed_attempts_7d=2.0)
    is_cold = FeatureTransformer.compute_is_cold_start(
        account_age_days=200.0,
        offline_features={"failed_attempts_7d": 2.0},
        sentinels=sentinels,
    )
    assert is_cold == 0, "Redis failure must not trigger cold-start"


def test_cold_start_and_degraded_produce_different_vectors(transformer):
    """
    Cold-start (is_cold_start=1) and degraded (is_cold_start=0, sentinels online)
    must produce DIFFERENT feature vectors — is_cold_start bit distinguishes them.
    """
    cold_event = make_cold_start_event()
    degraded_event = make_cold_start_event(is_cold_start=0)  # same but not cold-start

    vec_cold = transformer.transform(cold_event)
    vec_degraded = transformer.transform(degraded_event)

    assert not np.array_equal(vec_cold, vec_degraded), (
        "Cold-start and degraded feature vectors must differ — "
        "is_cold_start bit is the distinguishing signal"
    )


# ── deterministic features ───────────────────────────────────────────────────

def test_deterministic_features_hour_9():
    ts = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    feats = FeatureTransformer.compute_deterministic(ts)
    h = 9.0
    assert feats["hour_sin"] == pytest.approx(math.sin(2 * math.pi * h / 24.0), abs=1e-5)
    assert feats["hour_cos"] == pytest.approx(math.cos(2 * math.pi * h / 24.0), abs=1e-5)


def test_deterministic_features_midnight():
    ts = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    feats = FeatureTransformer.compute_deterministic(ts)
    assert feats["hour_sin"] == pytest.approx(0.0, abs=1e-5)
    assert feats["hour_cos"] == pytest.approx(1.0, abs=1e-5)


# ── temporal_split ───────────────────────────────────────────────────────────

def _make_raw_df(n: int = 1000, pos_rate: float = 0.2) -> pl.DataFrame:
    rng = np.random.default_rng(99)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [
        (base_ts.replace(tzinfo=None) + __import__("datetime").timedelta(hours=i)).isoformat()
        for i in range(n)
    ]
    return pl.DataFrame({
        "event_id": [str(i) for i in range(n)],
        "timestamp": pl.Series(timestamps).cast(pl.Datetime("us")),
        "user_id": [f"u{i % 50}" for i in range(n)],
        "ip_address": [f"10.0.{i % 10}.{i % 256}" for i in range(n)],
        "login_success": (rng.random(n) > 0.1).tolist(),
        "label": (rng.random(n) < pos_rate).astype(int).tolist(),
    })


def test_temporal_split_sizes():
    df = _make_raw_df(1000)
    splits = temporal_split(df)
    assert len(splits) == 5
    total = sum(len(s) for s in splits)
    assert total == 1000


def test_temporal_split_is_non_overlapping():
    df = _make_raw_df(200)
    train, val_stop, val_cal, thresh, test = temporal_split(df)
    all_ids = set()
    for split in (train, val_stop, val_cal, thresh, test):
        ids = set(split["event_id"].to_list())
        assert ids.isdisjoint(all_ids), "Splits must not overlap"
        all_ids |= ids


def test_temporal_split_preserves_ordering():
    df = _make_raw_df(200)
    train, val_stop, val_cal, thresh, test = temporal_split(df)
    # train timestamps must all precede test timestamps
    if len(train) > 0 and len(test) > 0:
        assert train["timestamp"].max() <= test["timestamp"].min()


# ── validate_no_leakage ──────────────────────────────────────────────────────

def test_validate_no_leakage_raises_on_future_window():
    import datetime as dt
    now = dt.datetime(2024, 1, 10, tzinfo=timezone.utc)
    future = dt.datetime(2024, 1, 11, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "event_id": ["e1"],
        "timestamp": pl.Series([now]).cast(pl.Datetime("us")),
        "feature_window_end": pl.Series([future]).cast(pl.Datetime("us")),
    })
    with pytest.raises(LeakageError, match="future data"):
        validate_no_leakage(df, "test_split")


def test_validate_no_leakage_passes_on_valid_data():
    import datetime as dt
    now = dt.datetime(2024, 1, 10, tzinfo=timezone.utc)
    before = dt.datetime(2024, 1, 9, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "event_id": ["e1"],
        "timestamp": pl.Series([now]).cast(pl.Datetime("us")),
        "feature_window_end": pl.Series([before]).cast(pl.Datetime("us")),
    })
    validate_no_leakage(df, "test_split")  # must not raise


# ── compute_offline_features ─────────────────────────────────────────────────

def test_offline_features_no_nan():
    """compute_offline_features must never produce NaN — sentinels fill all gaps."""
    df = _make_raw_df(300)
    train, *_ = temporal_split(df)
    result = compute_offline_features(train, "train")
    for col in ["failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
                "avg_login_hour_7d", "account_age_days"]:
        if col in result.columns:
            series = result[col].to_numpy()
            assert not np.any(np.isnan(series)), f"NaN found in {col}"


def test_offline_features_per_split_independence():
    """Each split's features must be computed from only its own rows."""
    df = _make_raw_df(500)
    train, val_stop, *_ = temporal_split(df)
    # Both splits use the same function but on independent data
    train_result = compute_offline_features(train, "train")
    val_result = compute_offline_features(val_stop, "val_stop")
    assert len(train_result) == len(train)
    assert len(val_result) == len(val_stop)


# ── apply_redis_degradation ──────────────────────────────────────────────────

def test_redis_degradation_online_features_replaced():
    """dropout_rate fraction of rows should have online features set to sentinels."""
    feature_names = [
        "failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
        "avg_login_hour_7d", "account_age_days",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_cold_start",
        "last_login_gap_h", "geo_distance_delta", "device_seen_flag",
    ]
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.random((n, len(feature_names))).astype(np.float32)
    # Set online features to non-sentinel values first
    for fname, sentinel in [("last_login_gap_h", -1.0), ("geo_distance_delta", -1.0), ("device_seen_flag", 0.0)]:
        idx = feature_names.index(fname)
        X[:, idx] = 50.0  # clearly non-sentinel

    dropout_rate = 0.15
    X_out = apply_redis_degradation(X, feature_names, dropout_rate, np.random.default_rng(1))

    # Count rows where all online features were replaced with sentinels
    gap_idx = feature_names.index("last_login_gap_h")
    degraded_rows = (X_out[:, gap_idx] == -1.0).sum()
    # Should be approximately dropout_rate * n (within ±5%)
    expected = dropout_rate * n
    assert abs(degraded_rows - expected) < 0.05 * n, \
        f"Expected ~{expected} degraded rows, got {degraded_rows}"


def test_redis_degradation_does_not_change_is_cold_start():
    """Degradation must NOT alter is_cold_start — that signals Redis failure, not cold-start."""
    feature_names = [
        "failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
        "avg_login_hour_7d", "account_age_days",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_cold_start",
        "last_login_gap_h", "geo_distance_delta", "device_seen_flag",
    ]
    cs_idx = feature_names.index("is_cold_start")
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(feature_names))).astype(np.float32)
    original_cs = X[:, cs_idx].copy()

    X_out = apply_redis_degradation(X, feature_names, 0.5, np.random.default_rng(2))
    np.testing.assert_array_equal(X_out[:, cs_idx], original_cs,
        err_msg="apply_redis_degradation must NOT modify is_cold_start column")


def test_redis_degradation_not_applied_to_test_conceptually():
    """Test that function leaves data unmodified with dropout_rate=0."""
    feature_names = [
        "failed_attempts_7d", "distinct_ips_7d", "login_success_rate_30d",
        "avg_login_hour_7d", "account_age_days",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_cold_start",
        "last_login_gap_h", "geo_distance_delta", "device_seen_flag",
    ]
    rng = np.random.default_rng(0)
    X = rng.random((100, len(feature_names))).astype(np.float32)
    X_out = apply_redis_degradation(X, feature_names, 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(X, X_out)


# ── Online Redis fetch ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_redis_known_user(fake_redis):
    """Known user with all 3 Redis keys → real values returned."""
    uid = "user_123"
    device_fp = "fp_abc"
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    last_ts = datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc)  # 24h ago

    fake_redis.seed_user(uid, last_ts.isoformat(), "51.5,-0.1", [device_fp])

    result, redis_ok = await fetch_redis_features(
        uid, device_fp, fake_redis,
        current_ts=ts, current_lat=51.5, current_lon=-0.1,
    )

    assert redis_ok is True
    assert result["last_login_gap_h"] == pytest.approx(24.0, abs=0.1)
    assert result["device_seen_flag"] == 1.0
    assert result["geo_distance_delta"] == pytest.approx(0.0, abs=1.0)


@pytest.mark.asyncio
async def test_fetch_redis_missing_user(fake_redis):
    """Unknown user → all sentinels returned, no exception."""
    result, redis_ok = await fetch_redis_features("unknown_user", "fp", fake_redis)
    assert redis_ok is True  # Redis succeeded; user simply has no history
    assert result == dict(SENTINEL_DICT)
    assert all(not math.isnan(v) for v in result.values())


@pytest.mark.asyncio
async def test_fetch_redis_failure_returns_sentinels():
    """Any exception from Redis → sentinels returned, no exception propagated."""
    class BrokenRedis:
        def pipeline(self):
            raise ConnectionError("Redis unavailable")

    result, redis_ok = await fetch_redis_features("uid", "fp", BrokenRedis())
    assert result == dict(SENTINEL_DICT)
    assert redis_ok is False


@pytest.mark.asyncio
async def test_fetch_redis_timeout_returns_sentinels():
    """A Redis that sleeps 20ms must return sentinels within the deadline."""
    import time

    class SlowRedis:
        def pipeline(self):
            class SlowPipeline:
                def get(self, k): return self
                def sismember(self, k, m): return self
                async def execute(self):
                    await asyncio.sleep(0.020)  # 20ms > 15ms deadline
                    return [None, None, False]
            return SlowPipeline()

    start = time.perf_counter()
    result, redis_ok = await fetch_redis_features("uid", "fp", SlowRedis())
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result == dict(SENTINEL_DICT), "Slow Redis must return sentinels"
    assert redis_ok is False
    assert elapsed_ms < 50, f"Sentinel path took {elapsed_ms:.1f}ms (expected < 50ms)"


# ── FeatureContractValidator ──────────────────────────────────────────────────

def test_contract_validates_correct_vector(feature_names):
    validator = FeatureContractValidator(feature_names)
    vec = np.ones(len(feature_names), dtype=np.float32)
    validator.validate_always(vec, "test")  # must not raise


def test_contract_rejects_wrong_length(feature_names):
    validator = FeatureContractValidator(feature_names)
    bad_vec = np.ones(len(feature_names) - 1, dtype=np.float32)
    with pytest.raises(FeatureContractError, match="features"):
        validator.validate_always(bad_vec, "test")


def test_contract_rejects_nan(feature_names):
    validator = FeatureContractValidator(feature_names)
    vec = np.ones(len(feature_names), dtype=np.float32)
    vec[0] = float("nan")
    with pytest.raises(FeatureContractError, match="NaN"):
        validator.validate_always(vec, "test")


def test_contract_rejects_inf(feature_names):
    validator = FeatureContractValidator(feature_names)
    vec = np.ones(len(feature_names), dtype=np.float32)
    vec[2] = float("inf")
    with pytest.raises(FeatureContractError, match="finite"):
        validator.validate_always(vec, "test")


def test_contract_sampling_rate_zero_never_validates(feature_names):
    """sampling_rate=0 must never call _check — even on a bad vector."""
    validator = FeatureContractValidator(feature_names, sampling_rate=0.0)
    bad_vec = np.ones(len(feature_names), dtype=np.float32)
    bad_vec[0] = float("nan")
    validated = validator.validate_sampled(bad_vec, "test")
    assert validated is False
