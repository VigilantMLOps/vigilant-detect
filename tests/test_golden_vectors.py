"""Golden vector CI gate — fixed inputs must produce stable feature hashes.

First run: creates tests/fixtures/golden_hashes.json.
Subsequent runs: compares against stored hashes.
Any unintentional transformer change fails this test.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from core.features.transformer import FeatureTransformer, load_schema

FIXTURES_DIR = Path("tests/fixtures")
GOLDEN_FILE = FIXTURES_DIR / "golden_hashes.json"

# Fixed input cases — never change without regenerating and reviewing the hashes.
GOLDEN_INPUTS = [
    {
        "name": "normal_user_daytime",
        "event": {
            "failed_attempts_7d": 0.0,
            "distinct_ips_7d": 1.0,
            "login_success_rate_30d": 0.95,
            "avg_login_hour_7d": 9.5,
            "account_age_days": 730.0,
            "hour_sin": 0.7071,
            "hour_cos": 0.7071,
            "dow_sin": 0.4339,
            "dow_cos": 0.9010,
            "is_cold_start": 0,
            "last_login_gap_h": 24.0,
            "geo_distance_delta": 5.0,
            "device_seen_flag": 1.0,
        },
    },
    {
        "name": "cold_start_new_user",
        "event": {
            "failed_attempts_7d": 0.0,
            "distinct_ips_7d": 0.0,
            "login_success_rate_30d": -1.0,
            "avg_login_hour_7d": -1.0,
            "account_age_days": 0.0,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "dow_sin": 0.0,
            "dow_cos": 1.0,
            "is_cold_start": 1,
            "last_login_gap_h": -1.0,
            "geo_distance_delta": -1.0,
            "device_seen_flag": 0.0,
        },
    },
    {
        "name": "high_risk_attacker",
        "event": {
            "failed_attempts_7d": 15.0,
            "distinct_ips_7d": 8.0,
            "login_success_rate_30d": 0.1,
            "avg_login_hour_7d": 3.0,
            "account_age_days": 5.0,
            "hour_sin": 0.707,
            "hour_cos": -0.707,
            "dow_sin": 0.782,
            "dow_cos": 0.623,
            "is_cold_start": 0,
            "last_login_gap_h": 0.5,
            "geo_distance_delta": 8500.0,
            "device_seen_flag": 0.0,
        },
    },
    {
        "name": "degraded_known_user",
        "event": {
            "failed_attempts_7d": 1.0,
            "distinct_ips_7d": 2.0,
            "login_success_rate_30d": 0.85,
            "avg_login_hour_7d": 11.0,
            "account_age_days": 365.0,
            "hour_sin": 0.259,
            "hour_cos": 0.966,
            "dow_sin": 0.0,
            "dow_cos": 1.0,
            "is_cold_start": 0,      # NOT cold-start — Redis degraded
            "last_login_gap_h": -1.0,   # sentinel
            "geo_distance_delta": -1.0, # sentinel
            "device_seen_flag": 0.0,    # sentinel
        },
    },
]


def _hash_vec(vec: np.ndarray) -> str:
    """SHA256 of the float32 array bytes — stable across Python versions."""
    return hashlib.sha256(vec.astype(np.float32).tobytes()).hexdigest()


def _compute_golden(transformer: FeatureTransformer) -> dict[str, str]:
    return {case["name"]: _hash_vec(transformer.transform(case["event"])) for case in GOLDEN_INPUTS}


@pytest.fixture(scope="session")
def golden_transformer():
    return FeatureTransformer(load_schema())


class TestGoldenVectors:
    def test_stable_hashes(self, golden_transformer):
        """
        Feature vectors for fixed inputs must produce stable SHA256 hashes.
        First run bootstraps the fixture; subsequent runs validate.
        """
        current = _compute_golden(golden_transformer)

        if not GOLDEN_FILE.exists():
            FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
            GOLDEN_FILE.write_text(json.dumps(current, indent=2))
            pytest.skip(
                f"Golden hashes bootstrapped at {GOLDEN_FILE}. "
                "Re-run tests to validate."
            )

        stored = json.loads(GOLDEN_FILE.read_text())

        for name, expected_hash in stored.items():
            assert name in current, f"Golden case '{name}' missing from current inputs"
            assert current[name] == expected_hash, (
                f"Feature hash changed for '{name}'.\n"
                f"  Expected: {expected_hash}\n"
                f"  Got:      {current[name]}\n"
                "If this change is intentional, regenerate by deleting "
                f"{GOLDEN_FILE} and re-running tests."
            )

    def test_perturbation_changes_hash(self, golden_transformer):
        """Changing any feature value must change the output hash."""
        base_case = GOLDEN_INPUTS[0]
        base_hash = _hash_vec(golden_transformer.transform(base_case["event"]))

        perturbed = dict(base_case["event"])
        perturbed["failed_attempts_7d"] += 1.0
        perturbed_hash = _hash_vec(golden_transformer.transform(perturbed))

        assert base_hash != perturbed_hash, \
            "Changing a feature value must change the output hash"

    def test_cold_start_hash_differs_from_degraded(self, golden_transformer):
        """Cold-start and degraded events must produce different hashes (is_cold_start bit)."""
        cold_hash = _hash_vec(
            golden_transformer.transform(GOLDEN_INPUTS[1]["event"])  # cold_start_new_user
        )
        degraded_hash = _hash_vec(
            golden_transformer.transform(GOLDEN_INPUTS[3]["event"])  # degraded_known_user
        )
        assert cold_hash != degraded_hash, (
            "Cold-start and degraded events must differ in at least the is_cold_start bit"
        )

    def test_transformer_pkl_matches_live(self, golden_transformer):
        """
        If tests/fixtures/transformer.pkl exists, it must produce identical
        hashes to the live transformer. Catches silent artifact drift.
        """
        import joblib
        pkl_path = FIXTURES_DIR / "transformer.pkl"
        if not pkl_path.exists():
            joblib.dump(golden_transformer, pkl_path)
            pytest.skip(f"transformer.pkl bootstrapped at {pkl_path}.")

        saved = joblib.load(pkl_path)
        for case in GOLDEN_INPUTS:
            live_vec = golden_transformer.transform(case["event"])
            saved_vec = saved.transform(case["event"])
            np.testing.assert_array_equal(
                live_vec, saved_vec,
                err_msg=f"transformer.pkl produces different output for '{case['name']}'"
            )
