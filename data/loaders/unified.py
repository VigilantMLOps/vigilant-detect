"""Unified data loader — merge sources, validate schema, return Polars DataFrame."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

REQUIRED_COLUMNS = {
    "event_id", "timestamp", "user_id", "login_success",
    "device_fingerprint", "geo_lat", "geo_lon", "label",
}


def load_dataset(
    synthetic_df: pl.DataFrame | None = None,
    ieee_df: pl.DataFrame | None = None,
    parquet_paths: list[str | Path] | None = None,
) -> pl.DataFrame:
    """
    Merge data sources into a single validated DataFrame sorted by timestamp.
    At least one source must be provided.
    """
    frames = []

    if synthetic_df is not None and len(synthetic_df) > 0:
        frames.append(_normalize(synthetic_df))

    if ieee_df is not None and len(ieee_df) > 0:
        frames.append(_normalize(ieee_df))

    if parquet_paths:
        for p in parquet_paths:
            df = pl.read_parquet(str(p))
            frames.append(_normalize(df))

    if not frames:
        raise ValueError("At least one data source must be provided.")

    merged = pl.concat(frames, how="diagonal").sort("timestamp")
    _validate_schema(merged)
    return merged


def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure timestamp is datetime, drop private hint columns, fill missing optional cols."""
    # Drop generator hint columns
    hint_cols = [c for c in df.columns if c.startswith("_")]
    if hint_cols:
        df = df.drop(hint_cols)

    # Ensure timestamp is datetime with UTC timezone
    if "timestamp" in df.columns:
        if df["timestamp"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("timestamp").str.to_datetime())
        elif df["timestamp"].dtype in (pl.Int64, pl.Float64):
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp")
            )

    # Add missing optional columns with defaults
    for col, default in [
        ("session_id", ""),
        ("ip_address", ""),
        ("geo_country", ""),
        ("geo_lat", 0.0),
        ("geo_lon", 0.0),
        ("mfa_used", False),
        ("mfa_method", "none"),
        ("login_duration_ms", 0.0),
    ]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    return df


def _validate_schema(df: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if df["label"].null_count() > 0:
        raise ValueError("label column contains nulls.")

    label_vals = set(df["label"].unique().to_list())
    if not label_vals.issubset({0, 1}):
        raise ValueError(f"label must be 0 or 1, got: {label_vals}")
