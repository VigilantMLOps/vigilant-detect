"""Tests for the decision system — ALLOW/CHALLENGE/BLOCK invariants."""
from __future__ import annotations

import pytest

from core.inference.decision import (
    ContextFlags,
    Decision,
    DecisionRules,
    active_context_flags,
    confidence_label,
    decide,
)


def flags(**overrides) -> ContextFlags:
    base = dict(
        device_seen_flag=1.0,
        geo_distance_delta=10.0,
        last_login_gap_h=24.0,
        is_cold_start=0,
    )
    base.update(overrides)
    return ContextFlags(**base)


# ── BLOCK gate (Gate 1) ──────────────────────────────────────────────────────

def test_block_when_prob_at_threshold(default_rules):
    result = decide(0.75, flags(), default_rules)
    assert result == Decision.BLOCK


def test_block_when_prob_above_threshold(default_rules):
    result = decide(0.99, flags(), default_rules)
    assert result == Decision.BLOCK


def test_no_block_when_prob_just_below(default_rules):
    result = decide(0.74, flags(), default_rules)
    assert result != Decision.BLOCK


def test_block_ignores_rules_entirely(default_rules):
    """
    Rules never produce BLOCK. Even with all rules cleared (normal device,
    small geo, low gap), probability alone triggers BLOCK.
    """
    safe_flags = flags(device_seen_flag=1.0, geo_distance_delta=1.0, last_login_gap_h=1.0)
    result = decide(0.90, safe_flags, default_rules)
    assert result == Decision.BLOCK


def test_rules_cannot_escalate_to_block(default_rules):
    """
    All context rules triggered simultaneously must NOT produce BLOCK
    when probability is below threshold_block. Max escalation = CHALLENGE.
    """
    all_rules_triggered = flags(device_seen_flag=0, geo_distance_delta=9999.0, last_login_gap_h=9999.0)
    result = decide(0.10, all_rules_triggered, default_rules)
    assert result == Decision.CHALLENGE, (
        "Rules can escalate ALLOW→CHALLENGE but NEVER →BLOCK; "
        f"got {result} for prob=0.10 with all rules triggered"
    )


# ── CHALLENGE gate (Gate 2) ──────────────────────────────────────────────────

def test_challenge_from_probability_alone(default_rules):
    result = decide(0.40, flags(), default_rules)
    assert result == Decision.CHALLENGE


def test_challenge_from_new_device(default_rules):
    result = decide(0.10, flags(device_seen_flag=0.0), default_rules)
    assert result == Decision.CHALLENGE


def test_challenge_from_geo_anomaly(default_rules):
    result = decide(0.10, flags(geo_distance_delta=600.0), default_rules)
    assert result == Decision.CHALLENGE


def test_challenge_from_dormancy(default_rules):
    result = decide(0.10, flags(last_login_gap_h=800.0), default_rules)
    assert result == Decision.CHALLENGE


def test_challenge_at_geo_boundary(default_rules):
    """geo_distance_delta exactly at threshold must NOT trigger challenge."""
    at_threshold = decide(0.10, flags(geo_distance_delta=500.0), default_rules)
    above_threshold = decide(0.10, flags(geo_distance_delta=500.001), default_rules)
    assert at_threshold == Decision.ALLOW
    assert above_threshold == Decision.CHALLENGE


def test_challenge_at_dormancy_boundary(default_rules):
    at_threshold = decide(0.10, flags(last_login_gap_h=720.0), default_rules)
    above_threshold = decide(0.10, flags(last_login_gap_h=720.001), default_rules)
    assert at_threshold == Decision.ALLOW
    assert above_threshold == Decision.CHALLENGE


# ── ALLOW ────────────────────────────────────────────────────────────────────

def test_allow_low_prob_no_rules(default_rules):
    result = decide(0.10, flags(), default_rules)
    assert result == Decision.ALLOW


def test_allow_at_challenge_boundary_exclusive(default_rules):
    """Probability just below threshold_challenge with no rules → ALLOW."""
    result = decide(0.349, flags(), default_rules)
    assert result == Decision.ALLOW


# ── Rule escalation cap ──────────────────────────────────────────────────────

def test_rule_escalation_is_one_level_only(default_rules):
    """
    A request with prob just above threshold_challenge is already CHALLENGE.
    Adding rules on top does not push it to BLOCK.
    """
    all_rules = flags(device_seen_flag=0, geo_distance_delta=9999.0, last_login_gap_h=9999.0)
    result = decide(0.50, all_rules, default_rules)
    assert result == Decision.CHALLENGE


def test_block_not_reachable_via_rules(default_rules):
    """
    Exhaustive: for all combination of rule triggers with prob < threshold_block,
    result must never be BLOCK.
    """
    for device in [0.0, 1.0]:
        for geo in [1.0, 9999.0]:
            for gap in [1.0, 9999.0]:
                f = flags(device_seen_flag=device, geo_distance_delta=geo, last_login_gap_h=gap)
                # Just below block threshold
                result = decide(0.749, f, default_rules)
                assert result != Decision.BLOCK, (
                    f"Rules triggered BLOCK at prob=0.749 with "
                    f"device={device}, geo={geo}, gap={gap}"
                )


# ── active_context_flags ─────────────────────────────────────────────────────

def test_active_flags_new_device(default_rules):
    f = flags(device_seen_flag=0)
    active = active_context_flags(f, default_rules)
    assert "new_device" in active


def test_active_flags_geo_anomaly(default_rules):
    f = flags(geo_distance_delta=600.0)
    active = active_context_flags(f, default_rules)
    assert "geo_anomaly" in active


def test_active_flags_dormant(default_rules):
    f = flags(last_login_gap_h=800.0)
    active = active_context_flags(f, default_rules)
    assert "dormant_account" in active


def test_active_flags_cold_start(default_rules):
    f = flags(is_cold_start=1)
    active = active_context_flags(f, default_rules)
    assert "cold_start" in active


def test_no_active_flags_for_normal_user(default_rules):
    f = flags()
    active = active_context_flags(f, default_rules)
    assert active == [] or "new_device" not in active


# ── confidence_label ─────────────────────────────────────────────────────────

def test_confidence_low_near_challenge_threshold(default_rules):
    label = confidence_label(0.34, default_rules)  # 0.01 below threshold_challenge
    assert label == "low"


def test_confidence_low_near_block_threshold(default_rules):
    label = confidence_label(0.74, default_rules)  # 0.01 below threshold_block
    assert label == "low"


def test_confidence_high_for_clear_allow(default_rules):
    label = confidence_label(0.05, default_rules)  # far from any threshold
    assert label == "high"


# ── DecisionRules.from_config ────────────────────────────────────────────────

def test_rules_from_config_defaults():
    cfg = {}
    rules = DecisionRules.from_config(cfg)
    assert rules.threshold_challenge == 0.35
    assert rules.threshold_block == 0.75
    assert rules.geo_anomaly_km == 500.0
    assert rules.dormancy_hours == 720.0


def test_rules_from_config_override():
    cfg = {
        "decision": {
            "threshold_challenge": 0.40,
            "threshold_block": 0.80,
            "rules": {
                "geo_anomaly_km": 300.0,
                "dormancy_hours": 480.0,
            },
        }
    }
    rules = DecisionRules.from_config(cfg)
    assert rules.threshold_challenge == 0.40
    assert rules.threshold_block == 0.80
    assert rules.geo_anomaly_km == 300.0
    assert rules.dormancy_hours == 480.0
