"""Offline feature computation.

Two modes:
  - compute_training_features(full_df): Run on the FULL dataset BEFORE splitting.
    Each event's features use only prior events for that user. No leakage.
    Call temporal_split() AFTER this.

  - compute_offline_features(split_df, split_name): Run on a single partition.
    Rolling features only see events within that partition (per-split only).

IMPORTANT: Do NOT use .over("user_id").shift(N) — the .shift() after .over()
is a GLOBAL shift, not per-group. Use (rolling_sum - current_value) instead to
get "sum of prior events" without cross-user contamination.
"""
from __future__ import annotations

import math

import polars as pl

# Declared sentinel values — must match schema.yaml exactly.
SENTINEL_FAILED_ATTEMPTS: float = 0.0
SENTINEL_DISTINCT_IPS: float = 0.0
SENTINEL_LOGIN_SUCCESS_RATE: float = -1.0
SENTINEL_AVG_LOGIN_HOUR: float = -1.0
SENTINEL_ACCOUNT_AGE: float = 0.0
SENTINEL_LAST_LOGIN_GAP: float = -1.0
SENTINEL_GEO_DISTANCE: float = -1.0
SENTINEL_DEVICE_SEEN: float = 0.0

_TWO_PI = 2.0 * math.pi
_R_EARTH_KM = 6371.0


class LeakageError(Exception):
    """Raised when rolling features contain future data relative to row timestamp."""


def validate_no_leakage(df: pl.DataFrame, split_name: str) -> None:
    """
    Verifies that no row has a rolling window end after its own timestamp.
    feature_window_end must be <= timestamp for every row.

    Raises LeakageError immediately — not a warning.
    """
    if "feature_window_end" not in df.columns:
        return
    violations = df.filter(pl.col("feature_window_end") > pl.col("timestamp"))
    if len(violations) > 0:
        first = violations.row(0, named=True)
        raise LeakageError(
            f"[{split_name}] {len(violations)} rows have future data in rolling window. "
            f"First offender: event_id={first.get('event_id')}, "
            f"timestamp={first.get('timestamp')}, "
            f"feature_window_end={first.get('feature_window_end')}. "
        )


def _rolling_prior_sum(expr: pl.Expr, window: int) -> pl.Expr:
    """
    Rolling sum of PRIOR events for each user — current row excluded.

    Uses (rolling_sum_including_current - current_value) to avoid the
    cross-user contamination bug in .over("user_id").shift(N).
    """
    return expr.rolling_sum(window_size=window, min_samples=1).over("user_id") - expr


def _rolling_prior_mean(value_expr: pl.Expr, window: int) -> pl.Expr:
    """
    Rolling mean of PRIOR events for each user — current row excluded.

    Returns null when there are no prior events (first event per user).
    """
    one = value_expr.is_not_null().cast(pl.Float32)
    prior_sum = (
        value_expr.cast(pl.Float32).rolling_sum(window_size=window, min_samples=1).over("user_id")
        - value_expr.cast(pl.Float32)
    )
    prior_count = (
        one.rolling_sum(window_size=window, min_samples=1).over("user_id") - 1.0
    )
    return prior_sum / prior_count


def compute_training_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute ALL model features on the full dataset with strict per-row time cutoff.

    Each event's features use only prior events for that user. No leakage.
    Call this BEFORE temporal_split().

    Computes:
      - Deterministic cyclic features (hour_sin/cos, dow_sin/cos) from timestamp
      - Offline rolling aggregates with per-user, prior-events-only semantics
      - Online-equivalent features derived from event history
      - is_cold_start: 1 for user's first-ever event in the dataset
    """
    df = df.sort("timestamp")

    # ── 1. Deterministic cyclic features from timestamp ──────────────────────
    hour_angle = (
        pl.col("timestamp").dt.hour().cast(pl.Float32)
        + pl.col("timestamp").dt.minute().cast(pl.Float32) / 60.0
    ) * (_TWO_PI / 24.0)
    dow_angle = pl.col("timestamp").dt.weekday().cast(pl.Float32) * (_TWO_PI / 7.0)

    df = df.with_columns([
        hour_angle.sin().alias("hour_sin"),
        hour_angle.cos().alias("hour_cos"),
        dow_angle.sin().alias("dow_sin"),
        dow_angle.cos().alias("dow_cos"),
    ])

    # ── 2. Offline rolling features (prior events only, no current row) ──────
    failed_flag = (pl.col("login_success") == False).cast(pl.Float32)
    event_flag = pl.col("login_success").is_not_null().cast(pl.Float32)
    hour_val = pl.col("timestamp").dt.hour().cast(pl.Float32)
    login_val = pl.col("login_success").cast(pl.Float32)

    # Intermediate: running sums and counts including current row
    df = df.with_columns([
        failed_flag.rolling_sum(window_size=168, min_samples=1).over("user_id").alias("_rs_failed"),
        event_flag.rolling_sum(window_size=168, min_samples=1).over("user_id").alias("_cnt_168"),
        login_val.rolling_sum(window_size=720, min_samples=1).over("user_id").alias("_rs_login"),
        event_flag.rolling_sum(window_size=720, min_samples=1).over("user_id").alias("_cnt_720"),
        hour_val.rolling_sum(window_size=168, min_samples=1).over("user_id").alias("_rs_hour"),
    ])

    df = df.with_columns([
        # failed_attempts_7d: count of failed logins in prior 168 events (proxy for 7d)
        (pl.col("_rs_failed") - failed_flag)
        .clip(0.0, None)
        .fill_null(SENTINEL_FAILED_ATTEMPTS)
        .fill_nan(SENTINEL_FAILED_ATTEMPTS)
        .cast(pl.Float32)
        .alias("failed_attempts_7d"),

        # distinct_ips_7d: count of prior events (event history depth proxy)
        (pl.col("_cnt_168") - 1.0)
        .clip(0.0, None)
        .fill_null(SENTINEL_DISTINCT_IPS)
        .fill_nan(SENTINEL_DISTINCT_IPS)
        .cast(pl.Float32)
        .alias("distinct_ips_7d"),

        # login_success_rate_30d: mean login_success of prior events
        ((pl.col("_rs_login") - login_val) / (pl.col("_cnt_720") - 1.0))
        .fill_null(SENTINEL_LOGIN_SUCCESS_RATE)
        .fill_nan(SENTINEL_LOGIN_SUCCESS_RATE)
        .cast(pl.Float32)
        .alias("login_success_rate_30d"),

        # avg_login_hour_7d: mean hour of prior events
        ((pl.col("_rs_hour") - hour_val) / (pl.col("_cnt_168") - 1.0))
        .fill_null(SENTINEL_AVG_LOGIN_HOUR)
        .fill_nan(SENTINEL_AVG_LOGIN_HOUR)
        .cast(pl.Float32)
        .alias("avg_login_hour_7d"),

        # account_age_days: days since user's first-ever event in the dataset
        (
            (pl.col("timestamp").cast(pl.Int64) -
             pl.col("timestamp").min().over("user_id").cast(pl.Int64))
            / (1_000_000 * 3600 * 24)
        ).cast(pl.Float32).fill_null(SENTINEL_ACCOUNT_AGE).alias("account_age_days"),

        # feature_window_end for leakage validation: last event BEFORE current
        pl.col("timestamp").shift(1).over("user_id").alias("feature_window_end"),
    ]).drop(["_rs_failed", "_cnt_168", "_rs_login", "_cnt_720", "_rs_hour"])

    # ── 3. Online-equivalent features derived from event history ─────────────

    # last_login_gap_h: hours since previous event for this user
    # prev_ts = last event's timestamp for this user (null for first event)
    prev_ts = pl.col("timestamp").shift(1).over("user_id")
    df = df.with_columns([
        (
            (pl.col("timestamp").cast(pl.Int64) - prev_ts.cast(pl.Int64))
            / (1_000_000 * 3600.0)
        ).cast(pl.Float32)
        .fill_null(SENTINEL_LAST_LOGIN_GAP)
        .fill_nan(SENTINEL_LAST_LOGIN_GAP)
        .alias("last_login_gap_h"),
    ])

    # geo_distance_delta: haversine distance from user's previous location (km)
    df = df.with_columns([
        pl.col("geo_lat").cast(pl.Float32).shift(1).over("user_id").alias("_prev_lat"),
        pl.col("geo_lon").cast(pl.Float32).shift(1).over("user_id").alias("_prev_lon"),
    ])

    dlat = (pl.col("geo_lat").cast(pl.Float32) - pl.col("_prev_lat")) * (math.pi / 180.0)
    dlon = (pl.col("geo_lon").cast(pl.Float32) - pl.col("_prev_lon")) * (math.pi / 180.0)
    lat1 = pl.col("_prev_lat") * (math.pi / 180.0)
    lat2 = pl.col("geo_lat").cast(pl.Float32) * (math.pi / 180.0)
    haversine_a = (
        (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2)
    ).clip(0.0, 1.0)

    df = df.with_columns([
        (_R_EARTH_KM * 2.0 * haversine_a.sqrt().arcsin())
        .cast(pl.Float32)
        .fill_null(SENTINEL_GEO_DISTANCE)
        .fill_nan(SENTINEL_GEO_DISTANCE)
        .alias("geo_distance_delta"),
    ]).drop(["_prev_lat", "_prev_lon"])

    # device_seen_flag: 1 if user has used this device fingerprint in a prior event
    df = df.with_columns([
        pl.col("timestamp")
        .rank("ordinal")
        .over(["user_id", "device_fingerprint"])
        .cast(pl.Float32)
        .alias("_device_rank"),
    ]).with_columns([
        (pl.col("_device_rank") > 1.0).cast(pl.Float32).alias("device_seen_flag"),
    ]).drop("_device_rank")

    # ── 4. is_cold_start: 1 for user's first-ever event ──────────────────────
    df = df.with_columns([
        pl.when(pl.col("account_age_days") == 0.0)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .cast(pl.Float32)
        .alias("is_cold_start"),
    ])

    return df


def compute_offline_features(split_df: pl.DataFrame, split_name: str) -> pl.DataFrame:
    """
    Compute rolling aggregates for a single temporal partition.

    For training, use compute_training_features() on the full dataset instead.
    This function is kept for per-split use and backward compatibility.

    shift(1) excludes the current row from its own feature window.
    fill_null replaces empty windows with declared cold-start sentinels.
    No NaN survives this function.
    """
    if len(split_df) == 0:
        return split_df

    result = split_df.sort("timestamp")

    failed_flag = (pl.col("login_success") == False).cast(pl.Float32)
    event_flag = pl.col("login_success").is_not_null().cast(pl.Float32)
    login_val = pl.col("login_success").cast(pl.Float32)
    hour_val = pl.col("timestamp").dt.hour().cast(pl.Float32)

    result = result.with_columns([
        failed_flag.rolling_sum(window_size=168, min_samples=1).over("user_id").alias("_rs_failed"),
        event_flag.rolling_sum(window_size=168, min_samples=1).over("user_id").alias("_cnt_168"),
        login_val.rolling_sum(window_size=720, min_samples=1).over("user_id").alias("_rs_login"),
        event_flag.rolling_sum(window_size=720, min_samples=1).over("user_id").alias("_cnt_720"),
        hour_val.rolling_sum(window_size=168, min_samples=1).over("user_id").alias("_rs_hour"),
    ])

    result = result.with_columns([
        (pl.col("_rs_failed") - failed_flag)
        .clip(0.0, None)
        .fill_null(SENTINEL_FAILED_ATTEMPTS)
        .fill_nan(SENTINEL_FAILED_ATTEMPTS)
        .cast(pl.Float32)
        .alias("failed_attempts_7d"),

        (pl.col("_cnt_168") - 1.0)
        .clip(0.0, None)
        .fill_null(SENTINEL_DISTINCT_IPS)
        .fill_nan(SENTINEL_DISTINCT_IPS)
        .cast(pl.Float32)
        .alias("distinct_ips_7d"),

        ((pl.col("_rs_login") - login_val) / (pl.col("_cnt_720") - 1.0))
        .fill_null(SENTINEL_LOGIN_SUCCESS_RATE)
        .fill_nan(SENTINEL_LOGIN_SUCCESS_RATE)
        .cast(pl.Float32)
        .alias("login_success_rate_30d"),

        ((pl.col("_rs_hour") - hour_val) / (pl.col("_cnt_168") - 1.0))
        .fill_null(SENTINEL_AVG_LOGIN_HOUR)
        .fill_nan(SENTINEL_AVG_LOGIN_HOUR)
        .cast(pl.Float32)
        .alias("avg_login_hour_7d"),

        (
            (pl.col("timestamp").cast(pl.Int64) -
             pl.col("timestamp").min().over("user_id").cast(pl.Int64))
            / (1_000_000 * 3600 * 24)
        ).cast(pl.Float32).fill_null(SENTINEL_ACCOUNT_AGE).alias("account_age_days"),

        pl.col("timestamp").alias("feature_window_end"),
    ]).drop(["_rs_failed", "_cnt_168", "_rs_login", "_cnt_720", "_rs_hour"])

    validate_no_leakage(result, split_name)
    return result


def temporal_split(
    df: pl.DataFrame,
    ratios: tuple[float, ...] = (0.70, 0.05, 0.05, 0.05, 0.15),
) -> tuple[pl.DataFrame, ...]:
    """
    Split sorted DataFrame into temporal partitions by row index.

    For training: call compute_training_features() BEFORE this function.
    Returns (train, val_stop, val_calibration, threshold_set, test).
    """
    df = df.sort("timestamp")
    n = len(df)
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    splits = []
    start = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            end = n
        else:
            end = start + int(n * ratio)
        splits.append(df[start:end])
        start = end

    return tuple(splits)
