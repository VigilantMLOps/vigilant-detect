"""Inference service — orchestrates the 9-step hot path + background logging."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import httpx

from core.features.online import fetch_redis_features, update_redis_state, SENTINEL_DICT
from core.features.transformer import FeatureTransformer
from core.inference.decision import DecisionRules
from core.inference.engine import InferenceResult, run_inference
from core.inference.state import get_model_state, ModelState
from core.monitoring.latency import LatencyTracker
from core.monitoring.payload import build_evaluate_model_payload, build_evaluate_drift_payload
from core.monitoring.score_stability import ScoreStabilityMonitor
from core.logger import get_logger

_logger = get_logger("vigilant-detect.inference")

# Per-500-event window for monitoring push
_PUSH_WINDOW_SIZE = 500


class InferenceService:
    def __init__(
        self,
        rules: DecisionRules,
        db,
        redis=None,
        vigilant_api_url: str = "http://localhost:8000",
        latency_tracker: LatencyTracker | None = None,
        stability_monitor: ScoreStabilityMonitor | None = None,
    ) -> None:
        self.rules = rules
        self.db = db
        self.redis = redis
        self.vigilant_api_url = vigilant_api_url
        self.latency_tracker = latency_tracker or LatencyTracker()
        self.stability_monitor = stability_monitor or ScoreStabilityMonitor()
        self._window_events: list[dict] = []
        self._window_start: datetime = datetime.now(timezone.utc)
        self._first_degradation_logged = False
        self._degraded = False

    async def predict(self, event_dict: dict) -> dict:
        """
        Execute the 9-step inference hot path.
        Returns a serialisable result dict.

        Steps 1-9 are zero blocking I/O except step 3 (async Redis, 15ms cap).
        Step 10 runs as a background task after response is returned.
        """
        start_ns = time.perf_counter_ns()

        # Pin model state at request entry — safe even during hot-swap
        state: ModelState | None = get_model_state()
        if state is None:
            raise RuntimeError("No model loaded. Deploy a model first.")

        # Step 1: event_dict is already validated by Pydantic in the route
        event_id = event_dict.get("event_id", str(uuid.uuid4()))
        timestamp = event_dict.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Step 2: Deterministic features (timestamp math — no I/O)
        det_features = FeatureTransformer.compute_deterministic(timestamp)

        # Compute is_cold_start BEFORE Redis (from offline data only)
        offline_features = {
            "failed_attempts_7d": event_dict.get("failed_attempts_7d", state.transformer.sentinels["failed_attempts_7d"]),
            "distinct_ips_7d": event_dict.get("distinct_ips_7d", state.transformer.sentinels["distinct_ips_7d"]),
            "login_success_rate_30d": event_dict.get("login_success_rate_30d", state.transformer.sentinels["login_success_rate_30d"]),
            "avg_login_hour_7d": event_dict.get("avg_login_hour_7d", state.transformer.sentinels["avg_login_hour_7d"]),
        }
        account_age = float(event_dict.get("account_age_days", state.transformer.sentinels["account_age_days"]))
        is_cold_start = FeatureTransformer.compute_is_cold_start(
            account_age, offline_features, state.transformer.sentinels
        )

        # Step 3: Redis fetch (hard 15ms deadline, no retry)
        is_degraded = False
        if self.redis is not None:
            uid = event_dict.get("user_id", "")
            device_fp = event_dict.get("device_fingerprint", "")
            redis_features, redis_ok = await fetch_redis_features(
                uid=uid,
                device_fp=device_fp,
                redis=self.redis,
                current_ts=timestamp,
                current_lat=event_dict.get("geo_lat"),
                current_lon=event_dict.get("geo_lon"),
            )
            is_degraded = not redis_ok  # degraded = Redis failed, not "user has no history"
        else:
            redis_features = dict(SENTINEL_DICT)
            is_degraded = True

        # Alert on first degradation per service lifecycle
        if is_degraded and not self._first_degradation_logged:
            _logger.info("Redis degraded mode — serving predictions with sentinel features.")
            self._first_degradation_logged = True
            self._degraded = True

        # Assemble complete event dict for transformer
        full_event = {
            **event_dict,
            **det_features,
            **offline_features,
            **redis_features,
            "account_age_days": account_age,
            "is_cold_start": float(is_cold_start),
        }

        # Steps 4-8: transform → validate → predict → calibrate → decide
        result: InferenceResult = run_inference(
            event=full_event,
            transformer=state.transformer,
            xgb_model=state.xgb,
            calibrator=state.calibrator,
            validator=state.validator,
            rules=self.rules,
            is_degraded=is_degraded,
            model_version=state.metadata.get("version", "unknown"),
        )

        # Step 9 happens in the route (Pydantic serialization)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        # Async background work (non-blocking)
        await self._background_tasks(
            event_dict=event_dict,
            full_event=full_event,
            result=result,
            state=state,
            elapsed_ms=elapsed_ms,
            timestamp=timestamp,
            is_degraded=is_degraded,
        )

        return {
            "event_id": event_id,
            "decision": result.decision.value,
            "risk_score": round(result.calibrated_probability, 4),
            "calibrated_probability": round(result.calibrated_probability, 4),
            "confidence": result.confidence,
            "context_flags": result.context_flags,
            "degraded": is_degraded,
            "model_version": result.model_version,
        }

    async def _background_tasks(
        self,
        event_dict: dict,
        full_event: dict,
        result: InferenceResult,
        state: ModelState,
        elapsed_ms: float,
        timestamp: datetime,
        is_degraded: bool,
    ) -> None:
        """Non-blocking post-response tasks. Never affects latency measurement."""
        # Latency tracking
        self.latency_tracker.record(
            elapsed_ms,
            is_degraded=is_degraded,
            is_cold_start=result.is_cold_start,
        )

        # Score stability monitoring
        self.stability_monitor.record(result.calibrated_probability, result.decision)

        # ClickHouse production log
        try:
            import json as _json
            feature_vec = {k: float(full_event.get(k, 0)) for k in state.transformer.feature_names_out}
            self.db.execute(
                "INSERT INTO ato_production_log "
                "(log_id, predicted_at, event_id, user_id, model_id, model_version, schema_hash, "
                "risk_score, calibrated_prob, decision, is_degraded, is_cold_start, latency_ms, feature_vector) "
                "VALUES (?, now64(3), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    str(uuid.uuid4()),
                    str(event_dict.get("event_id", "")),
                    str(event_dict.get("user_id", "")),
                    state.metadata.get("model_id", ""),
                    state.metadata.get("version", ""),
                    state.metadata.get("schema_hash", ""),
                    round(result.calibrated_probability, 4),
                    round(result.calibrated_probability, 4),
                    result.decision.value,
                    int(is_degraded),
                    int(result.is_cold_start),
                    round(elapsed_ms, 2),
                    _json.dumps(feature_vec),
                ],
            )
        except Exception as e:
            _logger.warning("ClickHouse log failed (non-fatal): {}", e)

        # Redis state update after successful login
        if self.redis is not None and event_dict.get("login_success"):
            try:
                await update_redis_state(
                    uid=event_dict.get("user_id", ""),
                    device_fp=event_dict.get("device_fingerprint", ""),
                    timestamp=timestamp,
                    lat=event_dict.get("geo_lat"),
                    lon=event_dict.get("geo_lon"),
                    redis=self.redis,
                )
            except Exception:
                pass

        # Window-based monitoring push
        self._window_events.append({
            "y_true": None,  # filled by feedback
            "y_pred": result.calibrated_probability,
            "decision": result.decision.value,
            "features": {k: full_event.get(k) for k in state.transformer.feature_names_out},
        })
        if len(self._window_events) >= _PUSH_WINDOW_SIZE:
            await self._push_monitoring_window(state)

        # Periodic latency flush
        if self.latency_tracker.should_flush():
            self.latency_tracker.flush_to_clickhouse(self.db)

        # Stability check
        signal = self.stability_monitor.check()
        if signal.any_warning():
            _logger.warning(
                "Score stability warning: entropy={}, mean_drift={}, class_dist={}",
                signal.entropy_warning, signal.mean_drift_warning, signal.class_dist_warning,
            )

    async def _push_monitoring_window(self, state: ModelState) -> None:
        """Push per-500-event window to vigilant-api. Failure never blocks inference."""
        events = self._window_events[:]
        self._window_events.clear()
        now = datetime.now(timezone.utc)

        schema_hash = state.metadata.get("schema_hash", "")
        fn_version = state.metadata.get("feature_names_version", "v1")
        model_version = state.metadata.get("version", "unknown")

        y_true = [e["y_true"] for e in events if e["y_true"] is not None]
        y_pred = [e["y_pred"] for e in events if e["y_true"] is not None]

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if y_true:
                    await client.post(
                        f"{self.vigilant_api_url}/api/v1/reporter/evaluate-model",
                        json=build_evaluate_model_payload(
                            schema_hash, fn_version, model_version,
                            self._window_start, now, y_true, y_pred,
                        ),
                    )

                # Drift payload: feature stats
                feature_stats = self._compute_feature_stats(events, state.transformer.feature_names_out)
                await client.post(
                    f"{self.vigilant_api_url}/api/v1/reporter/evaluate-drift",
                    json=build_evaluate_drift_payload(
                        schema_hash, fn_version, model_version,
                        self._window_start, now, feature_stats,
                    ),
                )
        except Exception as e:
            _logger.warning("Monitoring push to vigilant-api failed: {}", e)

        self._window_start = now

    def _compute_feature_stats(self, events: list[dict], feature_names: list[str]) -> dict:
        import statistics
        stats = {}
        for fname in feature_names:
            vals = [e["features"].get(fname) for e in events if e["features"].get(fname) is not None]
            if vals:
                stats[fname] = {
                    "mean": round(statistics.mean(vals), 4),
                    "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                    "count": len(vals),
                }
        return stats
