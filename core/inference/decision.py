"""Decision system — ALLOW/CHALLENGE/BLOCK with config-driven rule escalation.

Rules encode domain knowledge deterministically. Probability captures learned signal.
BLOCK is probability-only — rules never produce or escalate to BLOCK.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


@dataclass
class ContextFlags:
    device_seen_flag: float
    geo_distance_delta: float
    last_login_gap_h: float
    is_cold_start: int = 0


@dataclass
class DecisionRules:
    threshold_challenge: float
    threshold_block: float
    geo_anomaly_km: float
    dormancy_hours: float

    @classmethod
    def from_config(cls, cfg: dict) -> "DecisionRules":
        decision = cfg.get("decision", {})
        rules = decision.get("rules", {})
        return cls(
            threshold_challenge=float(decision.get("threshold_challenge", 0.35)),
            threshold_block=float(decision.get("threshold_block", 0.75)),
            geo_anomaly_km=float(rules.get("geo_anomaly_km", 500.0)),
            dormancy_hours=float(rules.get("dormancy_hours", 720.0)),
        )


def decide(prob: float, flags: ContextFlags, rules: DecisionRules) -> Decision:
    """
    Determine decision from calibrated probability + context flags.

    GATE 1: BLOCK — probability only. Rules never touch this gate.
            Evaluated unconditionally first. Always wins.

    GATE 2: CHALLENGE — probability OR rule escalation.
            Rules operate only below threshold_block.
            Max escalation: ALLOW → CHALLENGE. Never → BLOCK.

    DEFAULT: ALLOW.
    """
    # GATE 1: BLOCK — UNCONDITIONAL, probability only
    if prob >= rules.threshold_block:
        return Decision.BLOCK

    # GATE 2: CHALLENGE — probability or rule escalation (one level only)
    rule_triggered = (
        flags.device_seen_flag == 0
        or flags.geo_distance_delta > rules.geo_anomaly_km
        or flags.last_login_gap_h > rules.dormancy_hours
    )

    if prob >= rules.threshold_challenge or rule_triggered:
        return Decision.CHALLENGE

    return Decision.ALLOW


def active_context_flags(flags: ContextFlags, rules: DecisionRules) -> list[str]:
    """Return names of triggered context flags for the response payload."""
    active = []
    if flags.device_seen_flag == 0:
        active.append("new_device")
    if flags.geo_distance_delta > rules.geo_anomaly_km:
        active.append("geo_anomaly")
    if flags.last_login_gap_h > rules.dormancy_hours:
        active.append("dormant_account")
    if flags.is_cold_start:
        active.append("cold_start")
    return active


def confidence_label(prob: float, rules: DecisionRules) -> str:
    """Qualitative confidence label for the response."""
    margin_to_challenge = abs(prob - rules.threshold_challenge)
    margin_to_block = abs(prob - rules.threshold_block)
    min_margin = min(margin_to_challenge, margin_to_block)
    if min_margin < 0.05:
        return "low"
    if min_margin < 0.15:
        return "medium"
    return "high"
