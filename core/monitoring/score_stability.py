"""Score stability monitor — entropy, mean drift, class distribution signals."""
from __future__ import annotations

import math
import threading
from collections import deque
from datetime import datetime, timezone
from typing import NamedTuple

from core.inference.decision import Decision
from core.logger import get_logger

_logger = get_logger("vigilant-detect.stability")

_WINDOW_SIZE = 500


class StabilitySignal(NamedTuple):
    entropy_warning: bool
    mean_drift_warning: bool
    class_dist_warning: bool

    def warning_count(self) -> int:
        return sum([self.entropy_warning, self.mean_drift_warning, self.class_dist_warning])

    def should_retrain(self) -> bool:
        return self.warning_count() >= 2

    def any_warning(self) -> bool:
        return self.warning_count() >= 1


class ScoreStabilityMonitor:
    """
    In-memory score stability monitoring using three combined signals.
    Internal only — signals trigger pushes to vigilant-api, not exposed directly.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.3,
        mean_drift_threshold: float = 0.15,
        block_rate_multiplier_high: float = 5.0,
        block_rate_multiplier_low: float = 0.2,
        window_size: int = _WINDOW_SIZE,
    ) -> None:
        self._entropy_threshold = entropy_threshold
        self._mean_drift_threshold = mean_drift_threshold
        self._block_rate_high = block_rate_multiplier_high
        self._block_rate_low = block_rate_multiplier_low
        self._window: deque = deque(maxlen=window_size)
        self._baseline_mean: float | None = None
        self._baseline_block_rate: float | None = None
        self._lock = threading.Lock()

    def record(self, calibrated_prob: float, decision: Decision) -> None:
        with self._lock:
            self._window.append((calibrated_prob, decision))

    def check(self) -> StabilitySignal:
        with self._lock:
            window = list(self._window)

        if len(window) < 50:
            return StabilitySignal(False, False, False)

        probs = [p for p, _ in window]
        decisions = [d for _, d in window]

        # Signal 1: Entropy of decision distribution
        n = len(decisions)
        counts = {Decision.ALLOW: 0, Decision.CHALLENGE: 0, Decision.BLOCK: 0}
        for d in decisions:
            counts[d] = counts.get(d, 0) + 1

        entropy = 0.0
        for c in counts.values():
            if c > 0:
                p = c / n
                entropy -= p * math.log(p)
        entropy_warning = entropy < self._entropy_threshold

        # Signal 2: Mean drift vs startup baseline
        current_mean = sum(probs) / len(probs)
        if self._baseline_mean is None:
            self._baseline_mean = current_mean
        mean_drift_warning = abs(current_mean - self._baseline_mean) > self._mean_drift_threshold

        # Signal 3: BLOCK rate vs 7d baseline
        block_count = counts.get(Decision.BLOCK, 0)
        block_rate = block_count / n
        if self._baseline_block_rate is None:
            self._baseline_block_rate = max(block_rate, 1e-6)

        class_dist_warning = (
            block_rate > self._block_rate_high * self._baseline_block_rate
            or (self._baseline_block_rate > 1e-5 and block_rate < self._block_rate_low * self._baseline_block_rate)
        )

        return StabilitySignal(entropy_warning, mean_drift_warning, class_dist_warning)

    def reset_baseline(self) -> None:
        with self._lock:
            self._baseline_mean = None
            self._baseline_block_rate = None
