"""Monitoring payload construction — schema identity fields in every push."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MonitoringPayload:
    # Schema identity — REQUIRED in every push to vigilant-api
    schema_hash: str              # SHA256 of schema.yaml at model training time
    feature_names_version: str    # from schema.yaml header
    model_version: str            # from ModelState.metadata["version"]

    window_start: datetime
    window_end: datetime
    event_count: int
    data: dict


def build_evaluate_model_payload(
    schema_hash: str,
    feature_names_version: str,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    y_true: list[int],
    y_pred: list[float],
) -> dict:
    return {
        "schema_hash": schema_hash,
        "feature_names_version": feature_names_version,
        "model_version": model_version,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "event_count": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def build_evaluate_drift_payload(
    schema_hash: str,
    feature_names_version: str,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    feature_stats: dict,
) -> dict:
    return {
        "schema_hash": schema_hash,
        "feature_names_version": feature_names_version,
        "model_version": model_version,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "event_count": sum(
            v.get("count", 0) for v in feature_stats.values() if isinstance(v, dict)
        ),
        "feature_stats": feature_stats,
    }
