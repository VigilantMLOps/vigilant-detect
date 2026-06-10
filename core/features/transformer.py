"""Stateless feature transformer — the single shared layer for training and inference.

No .fit() method. Constructor takes the schema dict loaded from schema.yaml.
Saved as transformer.pkl alongside model artifacts to pin the schema snapshot.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class FeatureOrderError(Exception):
    """Raised when the assembled feature vector length does not match the schema."""


class FeatureTransformer:
    """
    Stateless. No .fit(). The same instance is used in training and inference.

    feature_names_out is the ONLY source of truth for feature order.
    XGBoost always receives features in this exact order — never inferred
    from dict key insertion order.
    """

    def __init__(self, schema: dict) -> None:
        self.feature_names: list[str] = schema["feature_names"]
        self.sentinels: dict[str, float] = {
            k: float(v) for k, v in schema["sentinels"].items()
        }
        self._n = len(self.feature_names)

    def transform(self, event: dict[str, Any]) -> np.ndarray:
        """
        Assemble feature vector in the exact order from self.feature_names.

        event must contain pre-computed offline aggregates, deterministic
        features, is_cold_start, and online Redis features (or sentinels).
        Missing keys are replaced with the declared sentinel — never NaN.

        Raises FeatureOrderError immediately on length mismatch.
        A wrong-order vector produces silent mis-predictions — crash is safer.
        """
        vec = np.array(
            [event.get(name, self.sentinels[name]) for name in self.feature_names],
            dtype=np.float32,
        )
        if len(vec) != self._n:
            raise FeatureOrderError(
                f"Assembled {len(vec)} features, expected {self._n}. "
                "Do not pass event dicts with extra or missing keys."
            )
        return vec

    @property
    def feature_names_out(self) -> list[str]:
        """Single source of truth for feature order. Used by contract validator,
        XGBoost feature_names param, golden vector tests, schema hash."""
        return self.feature_names

    @staticmethod
    def compute_deterministic(timestamp_utc) -> dict[str, float]:
        """Compute always-available cyclic time features from a UTC datetime."""
        import math
        h = timestamp_utc.hour + timestamp_utc.minute / 60.0
        dow = timestamp_utc.weekday()
        return {
            "hour_sin": math.sin(2 * math.pi * h / 24.0),
            "hour_cos": math.cos(2 * math.pi * h / 24.0),
            "dow_sin":  math.sin(2 * math.pi * dow / 7.0),
            "dow_cos":  math.cos(2 * math.pi * dow / 7.0),
        }

    @staticmethod
    def compute_is_cold_start(
        account_age_days: float,
        offline_features: dict[str, float],
        sentinels: dict[str, float],
    ) -> int:
        """
        Determine cold-start status from OFFLINE data before Redis is contacted.
        Returns 1 for new user with no prior history, 0 for known user.
        NEVER set to 1 because Redis failed — that is degraded mode.
        """
        if account_age_days == sentinels.get("account_age_days", 0.0):
            return 1
        cold_start_offline_keys = [
            "failed_attempts_7d", "distinct_ips_7d",
            "login_success_rate_30d", "avg_login_hour_7d",
        ]
        all_sentinel = all(
            offline_features.get(k, sentinels[k]) == sentinels[k]
            for k in cold_start_offline_keys
        )
        return 1 if all_sentinel else 0


def load_schema(schema_path: str | Path | None = None) -> dict:
    """Load schema.yaml and return as dict."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.yaml"
    with open(schema_path) as f:
        return yaml.safe_load(f)


def make_transformer(schema_path: str | Path | None = None) -> FeatureTransformer:
    """Convenience factory — loads schema.yaml and returns a FeatureTransformer."""
    return FeatureTransformer(load_schema(schema_path))
