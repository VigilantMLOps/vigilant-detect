"""Push a DATA_EVAL report for the ATO Detector model to vigilant-api.

Generates synthetic ATO training data, computes per-split statistics
(row count, class distribution, per-feature stats), and POSTs them to
POST /api/v1/reporter/ingest-data-eval.

Usage:
    python -m scripts.push_data_eval
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _compute_feature_stats(series) -> dict:
    """Compute basic stats for one feature column."""
    import polars as pl
    try:
        if series.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8, pl.UInt8):
            return {
                "name": series.name,
                "dtype": str(series.dtype),
                "missing_count": series.null_count(),
                "missing_pct": round(series.null_count() / len(series), 4) if len(series) else 0.0,
                "n_unique": series.n_unique(),
                "mean": round(float(series.mean() or 0), 4),
                "std": round(float(series.std() or 0), 4),
                "min": round(float(series.min() or 0), 4),
                "p25": round(float(series.quantile(0.25) or 0), 4),
                "p50": round(float(series.quantile(0.50) or 0), 4),
                "p75": round(float(series.quantile(0.75) or 0), 4),
                "max": round(float(series.max() or 0), 4),
            }
    except Exception:
        pass
    return {
        "name": series.name,
        "dtype": str(series.dtype),
        "missing_count": series.null_count(),
        "missing_pct": round(series.null_count() / len(series), 4) if len(series) else 0.0,
        "n_unique": series.n_unique(),
        "mean": None, "std": None, "min": None,
        "p25": None, "p50": None, "p75": None, "max": None,
    }


def _eval_split(split_df, split_name: str, model_version: str) -> dict:
    """Build a data-eval payload for one temporal split."""
    import polars as pl
    label_col = "label"
    feature_cols = [c for c in split_df.columns if c != label_col]

    dist = split_df[label_col].value_counts().to_dicts()
    class_dist = {str(row[label_col]): int(row["count"]) for row in dist}
    majority = max(class_dist.values()) if class_dist else 1
    minority = min(class_dist.values()) if class_dist else 1
    imbalance = round(majority / minority, 4) if minority > 0 else 0.0

    return {
        "split": split_name,
        "model_version": split_name,
        "n_rows": len(split_df),
        "n_features": len(feature_cols),
        "class_distribution": class_dist,
        "imbalance_ratio": imbalance,
        "duplicate_rows": int(split_df.is_duplicated().sum()),
        "missing_cells": int(split_df.null_count().sum_horizontal().sum()),
        "features": [_compute_feature_stats(split_df[col]) for col in feature_cols[:20]],
    }


def main():
    import httpx

    vigilant_api_url = os.getenv("VIGILANT_API_URL", "http://localhost:8000")

    print("Step 1: Generating ATO training data (50k events) ...")
    from data.generator.ato_simulator import generate_events
    from data.loaders.unified import load_dataset
    from core.features.offline import temporal_split, compute_training_features
    from core.models.registry import get_production_model_id

    from core.database import db
    db.startup()

    model_version = get_production_model_id(db) or "ato-detector-v1"
    print(f"  Production model_id: {model_version}")

    synthetic_df = generate_events(n_total=50_000, seed=42)
    df = load_dataset(synthetic_df=synthetic_df)
    print(f"  Dataset: {len(df):,} events ({int((df['label']==1).sum()):,} ATO)")

    print("Step 2: Computing features and temporal splits ...")
    feat_df = compute_training_features(df)
    splits = temporal_split(feat_df)
    split_names = [
        "1 - Training",
        "2 - Validation (Early Stop)",
        "3 - Validation (Calibration)",
        "4 - Threshold Tuning",
        "5 - Final Evaluation",
    ]

    print("Step 3: Pushing DATA_EVAL reports to vigilant-api ...")
    for name, split_df in zip(split_names, splits):
        payload = _eval_split(split_df, split_name=name, model_version=model_version)
        try:
            resp = httpx.post(
                f"{vigilant_api_url}/api/v1/reporter/ingest-data-eval",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"  {name}: {payload['n_rows']:,} rows, report_id={data.get('report_id', '?')}")
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}")

    db.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
