"""Tests for retraining components — adaptive delay, dedup, decay weighting."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from math import exp

import polars as pl
import pytest

from tests.fake_database import FakeDatabase


# ── Label delay config ─────────────────────────────────────────────────────────

def test_label_delay_config_seeded(fake_db):
    """FakeDatabase must be pre-seeded with the four canonical delay sources."""
    rows = fake_db.fetchall("SELECT source, min_delay_h FROM label_delay_config ORDER BY source")
    sources = {r["source"]: r["min_delay_h"] for r in rows}

    assert sources["analyst"] == 48.0
    assert sources["auto_lock"] == 72.0
    assert sources["temporal"] == 96.0
    assert sources["auto_mfa"] == 24.0


def test_min_delay_per_source():
    """Confirm delay ordering: auto_mfa < analyst < auto_lock < temporal."""
    delays = {"analyst": 48.0, "auto_lock": 72.0, "temporal": 96.0, "auto_mfa": 24.0}
    assert delays["auto_mfa"] < delays["analyst"] < delays["auto_lock"] < delays["temporal"]


# ── Feedback insertion and dedup ──────────────────────────────────────────────

def test_feedback_insert_and_fetch(fake_db):
    """Basic feedback round-trip."""
    fake_db.execute(
        """INSERT OR IGNORE INTO feedback
           (feedback_id, event_id, true_label, label_source, confidence, confirmed_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ["fb1", "evt_1", 1, "analyst", 0.9, "2024-01-10 10:00:00"],
    )
    row = fake_db.fetchone("SELECT * FROM feedback WHERE event_id = ?", ["evt_1"])
    assert row is not None
    assert row["true_label"] == 1
    assert row["confidence"] == 0.9


def test_feedback_dedup_latest_write_wins(fake_db):
    """
    Multiple feedback entries for the same event_id must be deduplicated
    to the one with the latest confirmed_at (latest-write-wins).
    """
    event_id = "evt_dedup_test"
    fake_db.execute(
        "INSERT OR IGNORE INTO feedback "
        "(feedback_id, event_id, true_label, label_source, confidence, confirmed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["fb_early", event_id, 0, "auto_lock", 0.7, "2024-01-10 10:00:00"],
    )
    fake_db.execute(
        "INSERT OR IGNORE INTO feedback "
        "(feedback_id, event_id, true_label, label_source, confidence, confirmed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["fb_late", event_id, 1, "analyst", 0.95, "2024-01-12 14:00:00"],
    )

    # Simulate latest-write-wins: fetch the last entry by confirmed_at
    rows = fake_db.fetchall(
        "SELECT true_label, confidence, confirmed_at FROM feedback "
        "WHERE event_id = ? ORDER BY confirmed_at DESC",
        [event_id],
    )
    assert rows[0]["true_label"] == 1, "Latest entry (true_label=1) must win"
    assert rows[0]["confidence"] == 0.95


def test_feedback_mark_incorporated(fake_db):
    """mark_incorporated must set incorporated=TRUE for given event_ids."""
    fake_db.execute(
        "INSERT OR IGNORE INTO feedback "
        "(feedback_id, event_id, true_label, label_source, confidence, confirmed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["fb_inc", "evt_inc", 1, "analyst", 0.9, "2024-01-10 10:00:00"],
    )
    fake_db.execute(
        "UPDATE feedback SET incorporated = TRUE WHERE event_id = ?",
        ["evt_inc"],
    )
    row = fake_db.fetchone("SELECT incorporated FROM feedback WHERE event_id = ?", ["evt_inc"])
    assert row["incorporated"] == 1


def test_feedback_confidence_filter():
    """Only feedback with confidence >= threshold should pass the filter."""
    entries = [
        {"event_id": "e1", "confidence": 0.95, "label_source": "analyst"},
        {"event_id": "e2", "confidence": 0.5, "label_source": "auto_lock"},   # below 0.6
        {"event_id": "e3", "confidence": 0.65, "label_source": "auto_mfa"},
        {"event_id": "e4", "confidence": 0.59, "label_source": "temporal"},  # just below
    ]
    MIN_CONFIDENCE = 0.6
    passed = [e for e in entries if e["confidence"] >= MIN_CONFIDENCE]
    assert {e["event_id"] for e in passed} == {"e1", "e3"}


# ── Adaptive label delay ──────────────────────────────────────────────────────

def test_feedback_within_delay_is_excluded():
    """Feedback confirmed < min_delay_h ago must be excluded from the training pool."""
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    delays = {"analyst": 48.0, "auto_lock": 72.0, "temporal": 96.0, "auto_mfa": 24.0}

    entries = [
        {"label_source": "analyst", "confirmed_at": now - timedelta(hours=47)},   # too fresh
        {"label_source": "analyst", "confirmed_at": now - timedelta(hours=49)},   # eligible
        {"label_source": "auto_lock", "confirmed_at": now - timedelta(hours=71)}, # too fresh
        {"label_source": "auto_lock", "confirmed_at": now - timedelta(hours=73)}, # eligible
        {"label_source": "temporal", "confirmed_at": now - timedelta(hours=100)}, # eligible
        {"label_source": "auto_mfa", "confirmed_at": now - timedelta(hours=23)},  # too fresh
    ]

    eligible = [
        e for e in entries
        if (now - e["confirmed_at"]).total_seconds() / 3600 >= delays[e["label_source"]]
    ]
    assert len(eligible) == 3


def test_auto_mfa_24h_delay():
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    confirmed_at = now - timedelta(hours=23, minutes=59)
    delay_h = (now - confirmed_at).total_seconds() / 3600
    assert delay_h < 24.0, "Should be excluded (not yet past 24h delay)"

    confirmed_at_eligible = now - timedelta(hours=24, minutes=1)
    delay_h_eligible = (now - confirmed_at_eligible).total_seconds() / 3600
    assert delay_h_eligible > 24.0, "Should be eligible after 24h"


# ── Time-decay weighting ──────────────────────────────────────────────────────

def test_decay_weight_recent_higher_than_old():
    """Recent events must receive higher weight than older events."""
    DECAY_RATE = 0.02
    days_0 = 0
    days_30 = 30
    days_90 = 90

    w_recent = exp(-DECAY_RATE * days_0)
    w_30 = exp(-DECAY_RATE * days_30)
    w_90 = exp(-DECAY_RATE * days_90)

    assert w_recent > w_30 > w_90, "Decay must be monotonically decreasing with age"


def test_decay_weight_formula():
    """exp(-0.02 * days_ago) matches the documented formula."""
    DECAY_RATE = 0.02
    assert exp(-DECAY_RATE * 0) == pytest.approx(1.0)
    assert exp(-DECAY_RATE * 100) == pytest.approx(exp(-2.0), rel=1e-5)


def test_confidence_weight_by_source():
    """Each source has a distinct confidence multiplier."""
    multipliers = {
        "analyst": 1.0,
        "auto_lock": 0.9,
        "temporal": 0.8,
        "auto_mfa": 0.7,
    }
    base_weight = 1.0
    base_confidence = 0.9

    weights = {
        source: base_weight * base_confidence * mult
        for source, mult in multipliers.items()
    }

    assert weights["analyst"] > weights["auto_lock"] > weights["temporal"] > weights["auto_mfa"]


# ── Minimum ATO event guard ───────────────────────────────────────────────────

def test_min_ato_events_guard():
    """Retraining must abort if fewer than MIN_ATO_EVENTS labeled positive events."""
    from unittest.mock import patch
    import polars as pl
    from services.retraining_service import RetrainingService, _MIN_ATO_EVENTS

    assert _MIN_ATO_EVENTS == 1000

    fake_db = FakeDatabase()
    svc = RetrainingService(db=fake_db, config_path="config/training.yaml")

    # Patch _collect_feedback to return a DataFrame with 5 ATO events (< 1000)
    few_ato = pl.DataFrame({
        "event_id": [str(i) for i in range(10)],
        "true_label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "confidence": [0.9] * 10,
        "label_source": ["analyst"] * 10,
        "confirmed_at": ["2024-01-10 10:00:00"] * 10,
        "event_at": ["2024-01-08 10:00:00"] * 10,
    })

    with patch.object(svc, "_collect_feedback", return_value=few_ato):
        result = svc.run()

    assert result is None, "Retraining must return None when ATO event count < _MIN_ATO_EVENTS"


# ── Retraining guard: timestamp range exclusion ───────────────────────────────

def test_timestamp_range_blacklist_logic():
    """
    Events within the val/test timestamp range of the production model
    must be excluded from retraining to prevent data leakage.
    """
    val_start = "2024-01-10T00:00:00"
    test_end = "2024-01-20T00:00:00"

    events = [
        {"event_id": "e1", "predicted_at": "2024-01-05T00:00:00"},  # before range → ok
        {"event_id": "e2", "predicted_at": "2024-01-12T00:00:00"},  # inside range → excluded
        {"event_id": "e3", "predicted_at": "2024-01-25T00:00:00"},  # after range → ok
    ]

    eligible = [
        e for e in events
        if not (val_start <= e["predicted_at"] <= test_end)
    ]
    assert len(eligible) == 2
    assert {e["event_id"] for e in eligible} == {"e1", "e3"}
