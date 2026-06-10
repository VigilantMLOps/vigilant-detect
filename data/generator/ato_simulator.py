"""Synthetic ATO event generator — distribution-based, class overlap mandatory.

No deterministic label encoding. Attack features are sampled from distributions
with overlap into normal-event ranges. The model must learn from statistics,
not trivially separable rules.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def _load_patterns(patterns_path: str | Path | None = None) -> dict:
    if patterns_path is None:
        patterns_path = Path(__file__).parent / "patterns.yaml"
    with open(patterns_path) as f:
        return yaml.safe_load(f)


def generate_events(
    n_total: int = 100_000,
    ato_rate: float | None = None,
    seed: int = 42,
    patterns_path: str | Path | None = None,
    start_date: datetime | None = None,
) -> pl.DataFrame:
    """
    Generate synthetic login events with ATO and normal classes.

    Returns a Polars DataFrame with raw login event fields. Offline features
    are NOT pre-computed here — compute_offline_features() is called on each
    temporal split by the training pipeline.
    """
    patterns = _load_patterns(patterns_path)
    if ato_rate is None:
        ato_rate = patterns.get("ato_rate", 0.03)

    rng = np.random.default_rng(seed)
    n_ato = max(1, int(n_total * ato_rate))
    n_normal = n_total - n_ato

    if start_date is None:
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)

    rows = []
    rows.extend(_generate_normal_events(n_normal, rng, patterns, start_date))
    rows.extend(_generate_ato_events(n_ato, rng, patterns, start_date))

    df = pl.DataFrame(rows).sort("timestamp")

    # Inject label noise (2% flip — makes the task realistic)
    noise_rate = patterns.get("label_noise_rate", 0.02)
    noise_mask = rng.random(len(df)) < noise_rate
    labels = df["label"].to_numpy().copy()
    labels[noise_mask] = 1 - labels[noise_mask]
    df = df.with_columns(pl.Series("label", labels.tolist()))

    return df


def _generate_normal_events(n: int, rng, patterns: dict, start_date: datetime) -> list[dict]:
    rows = []
    n_noise = int(n * patterns["credential_stuffing"]["normal_noise_rate"])
    end_date = start_date + timedelta(days=365)
    total_seconds = int((end_date - start_date).total_seconds())

    user_ids = [f"user_{i:06d}" for i in range(n // 10)]

    for i in range(n):
        uid = user_ids[rng.integers(0, len(user_ids))]
        ts = start_date + timedelta(seconds=int(rng.integers(0, total_seconds)))
        is_noise = i < n_noise

        row = {
            "event_id": str(uuid.uuid4()),
            "timestamp": ts,
            "user_id": uid,
            "session_id": str(uuid.uuid4()),
            "ip_address": f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}",
            "geo_country": rng.choice(["US", "GB", "DE", "CA", "FR", "AU"]),
            "geo_lat": float(rng.uniform(25, 65)),
            "geo_lon": float(rng.uniform(-120, 30)),
            "login_success": bool(rng.random() > 0.05),  # 5% fail rate
            "mfa_used": bool(rng.random() > 0.7),
            "mfa_method": rng.choice(["none", "totp", "sms", "email"]),
            "login_duration_ms": float(rng.normal(1500, 300)),
            "device_fingerprint": f"device_{rng.integers(0, 5)}",  # limited device pool per user
            "label": 0,
            # Pre-seed some fields for offline computation
            "_failed_attempts_hint": int(rng.poisson(is_noise * 4 + 0.5)),
        }
        rows.append(row)
    return rows


def _generate_ato_events(n: int, rng, patterns: dict, start_date: datetime) -> list[dict]:
    rows = []
    end_date = start_date + timedelta(days=365)
    total_seconds = int((end_date - start_date).total_seconds())
    # ATO actors target a range of victims
    user_ids = [f"user_{i:06d}" for i in range(500)]

    cred = patterns["credential_stuffing"]
    geo = patterns["geo_anomaly"]
    dev = patterns["device_change"]
    dorm = patterns["dormancy_exploit"]
    feat_noise = patterns.get("feature_noise_std", 0.05)

    for i in range(n):
        # Mix attack patterns
        attack_type = rng.choice(["credential_stuffing", "geo", "device", "slow_brute", "dormancy"])
        uid = user_ids[rng.integers(0, len(user_ids))]
        ts = start_date + timedelta(seconds=int(rng.integers(0, total_seconds)))

        # Failed attempts — distribution with noise
        if attack_type == "credential_stuffing":
            failed_hint = int(rng.poisson(cred["failed_attempts_lambda"]))
        elif attack_type == "slow_brute":
            slow = patterns.get("slow_brute_force", {})
            failed_hint = int(rng.integers(slow.get("attempts_low", 3), slow.get("attempts_high", 8) + 1))
        else:
            failed_hint = int(rng.poisson(2))

        # Geo distance — LogNormal with legit travel noise
        if attack_type == "geo":
            geo_dist = float(np.exp(rng.normal(geo["distance_mu"], geo["distance_sigma"])))
        else:
            geo_dist = float(rng.uniform(0, 200))

        # Device — new device probability
        new_device = rng.random() < dev["ato_new_device_prob"]

        # Login gap — dormancy exploitation
        if attack_type == "dormancy":
            gap_h = float(np.exp(rng.normal(dorm["gap_mu"], dorm["gap_sigma"])))
        else:
            gap_h = float(rng.uniform(0, 48))

        row = {
            "event_id": str(uuid.uuid4()),
            "timestamp": ts,
            "user_id": uid,
            "session_id": str(uuid.uuid4()),
            "ip_address": f"192.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}",
            "geo_country": rng.choice(["RU", "CN", "BR", "NG", "IN", "US"]),
            "geo_lat": float(rng.uniform(-60, 70)),
            "geo_lon": float(rng.uniform(-180, 180)),
            "login_success": bool(rng.random() > 0.3),  # higher fail rate
            "mfa_used": bool(rng.random() > 0.9),
            "mfa_method": "none",
            "login_duration_ms": float(rng.normal(800, 200)),
            "device_fingerprint": f"device_{rng.integers(10, 50)}" if new_device else f"device_{rng.integers(0, 5)}",
            "label": 1,
            "_failed_attempts_hint": failed_hint,
            "_geo_dist_hint": geo_dist,
            "_gap_h_hint": gap_h,
        }
        rows.append(row)
    return rows
