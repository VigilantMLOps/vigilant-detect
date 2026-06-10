"""Latency tracker — ring buffer P50/P95/P99, ClickHouse flush every 60s."""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from core.logger import get_logger

_logger = get_logger("vigilant-detect.latency")

_FLUSH_INTERVAL_S = 60
_RING_SIZE = 10_000


class LatencyTracker:
    def __init__(self, model_version: str = "unknown", schema_hash: str = "") -> None:
        self.model_version = model_version
        self.schema_hash = schema_hash
        self._ring: deque[float] = deque(maxlen=_RING_SIZE)
        self._degraded_count = 0
        self._cold_start_count = 0
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def record(self, latency_ms: float, is_degraded: bool = False, is_cold_start: bool = False) -> None:
        with self._lock:
            self._ring.append(latency_ms)
            if is_degraded:
                self._degraded_count += 1
            if is_cold_start:
                self._cold_start_count += 1

    def percentiles(self) -> tuple[float, float, float]:
        with self._lock:
            if not self._ring:
                return 0.0, 0.0, 0.0
            arr = np.array(list(self._ring), dtype=np.float32)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        return p50, p95, p99

    def should_flush(self) -> bool:
        return (time.monotonic() - self._last_flush) >= _FLUSH_INTERVAL_S

    def flush_to_clickhouse(self, db) -> None:
        """Write P50/P95/P99 snapshot to ClickHouse ato_inference_metrics."""
        with self._lock:
            if not self._ring:
                return
            arr = np.array(list(self._ring), dtype=np.float32)
            n = len(arr)
            degraded = self._degraded_count
            cold = self._cold_start_count
            # Reset counters
            self._degraded_count = 0
            self._cold_start_count = 0
            self._last_flush = time.monotonic()

        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))

        try:
            import uuid as _uuid
            db.execute(
                "INSERT INTO ato_inference_metrics "
                "(metric_id, recorded_at, model_id, model_version, schema_hash, "
                "window_size, p50_ms, p95_ms, p99_ms, degraded_count, cold_start_count) "
                "VALUES (?, now64(3), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    str(_uuid.uuid4()),
                    self.model_version,
                    self.model_version,
                    self.schema_hash,
                    n, p50, p95, p99, degraded, cold,
                ],
            )
            _logger.debug("Flushed latency metrics: P50={:.1f}ms P95={:.1f}ms P99={:.1f}ms", p50, p95, p99)
        except Exception as e:
            _logger.warning("ClickHouse latency flush failed: {}", e)
