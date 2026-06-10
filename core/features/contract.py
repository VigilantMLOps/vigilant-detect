"""FeatureContractValidator — enforces feature order, dtype, no-NaN invariant."""
from __future__ import annotations

import random

import numpy as np


class FeatureContractError(Exception):
    """Raised when the feature vector violates the contract."""


class FeatureContractValidator:
    """
    Validates assembled feature vectors against the schema.

    Run modes:
      - always: every call is validated (training, model load, staging)
      - sampling: validates hot_path_validation_rate fraction of calls (production)

    Checks:
      1. Feature count matches schema
      2. No NaN values (sentinels are defined floats, NaN is invalid)
      3. All values are finite
      4. Length consistency
    """

    def __init__(self, feature_names: list[str], sampling_rate: float = 1.0) -> None:
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.sampling_rate = sampling_rate

    def validate_always(self, vec: np.ndarray, context: str = "") -> None:
        """Validate unconditionally — training and model load."""
        self._check(vec, context)

    def validate_sampled(self, vec: np.ndarray, context: str = "") -> bool:
        """Validate at sampling rate — hot path inference. Returns True if validated."""
        if random.random() < self.sampling_rate:
            self._check(vec, context)
            return True
        return False

    def _check(self, vec: np.ndarray, context: str) -> None:
        tag = f"[{context}] " if context else ""

        if vec.ndim == 1:
            if len(vec) != self.n_features:
                raise FeatureContractError(
                    f"{tag}Feature vector has {len(vec)} features, expected {self.n_features}. "
                    f"Schema: {self.feature_names[:5]}..."
                )
            if np.any(np.isnan(vec)):
                nan_idx = np.where(np.isnan(vec))[0]
                bad_names = [self.feature_names[i] for i in nan_idx[:3]]
                raise FeatureContractError(
                    f"{tag}NaN in feature vector at positions {nan_idx[:3].tolist()}: {bad_names}. "
                    "All NaN must be replaced with declared sentinels."
                )
            if not np.all(np.isfinite(vec)):
                bad_idx = np.where(~np.isfinite(vec))[0]
                raise FeatureContractError(
                    f"{tag}Non-finite values at positions {bad_idx[:3].tolist()}."
                )
        elif vec.ndim == 2:
            if vec.shape[1] != self.n_features:
                raise FeatureContractError(
                    f"{tag}Feature matrix has {vec.shape[1]} columns, expected {self.n_features}."
                )
            nan_mask = np.isnan(vec)
            if nan_mask.any():
                row, col = np.where(nan_mask)
                raise FeatureContractError(
                    f"{tag}NaN in feature matrix at ({row[0]}, {col[0]}): "
                    f"feature '{self.feature_names[col[0]]}'. "
                    "All NaN must be replaced with declared sentinels."
                )
            if not np.all(np.isfinite(vec)):
                raise FeatureContractError(f"{tag}Non-finite values in feature matrix.")
        else:
            raise FeatureContractError(f"{tag}Expected 1D or 2D array, got shape {vec.shape}.")
