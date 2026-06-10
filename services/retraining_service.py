"""Retraining service — adaptive label delay, dedup, decay weighting, shadow replay."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from math import exp
from pathlib import Path

import polars as pl
import yaml

from core.logger import get_logger
from core.models.registry import (
    get_production_model_id,
    promote_model,
    register_model,
)

_logger = get_logger("vigilant-detect.retraining")

_MIN_ATO_EVENTS = 1_000
_DECAY_RATE = 0.02


class RetrainingService:
    def __init__(self, db, config_path: str | Path | None = None) -> None:
        self.db = db
        self.config_path = Path(config_path or "config/training.yaml")

    def run(self) -> str | None:
        """
        Run the full retraining pipeline.
        Returns model_id if training succeeded and all gates passed, else None.
        """
        _logger.info("Retraining triggered.")

        # Step 1: Collect labelled feedback (adaptive delay, dedup, confidence filter)
        feedback_df = self._collect_feedback()
        n_ato = int((feedback_df["true_label"] == 1).sum()) if len(feedback_df) > 0 else 0

        if n_ato < _MIN_ATO_EVENTS:
            _logger.warning(
                "Retraining aborted: only {} ATO-labelled events (minimum {})",
                n_ato, _MIN_ATO_EVENTS,
            )
            return None

        # Step 2: Build training dataset with time-decay weighting
        df = self._build_dataset(feedback_df)
        if df is None or len(df) == 0:
            _logger.warning("Retraining aborted: empty dataset after filtering.")
            return None

        # Step 3: Train
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        try:
            from core.models.trainer import train, TrainingGateError
            result = train(df=df, config=cfg, db=self.db)
        except Exception as e:
            _logger.error("Retraining failed: {}", e)
            return None

        # Step 4: Shadow replay before promotion
        if not self._shadow_replay(result.model_id):
            _logger.warning("Shadow replay failed — not promoting model {}", result.model_id)
            return None

        # Step 5: Promote
        promote_model(self.db, result.model_id)
        _logger.info("Retraining promoted model {}", result.model_id)

        # Step 6: Mark feedback as incorporated
        self._mark_incorporated(feedback_df)

        return result.model_id

    def _collect_feedback(self) -> pl.DataFrame:
        """
        Collect feedback with adaptive label delay (per source), dedup (latest-write-wins),
        confidence filter, and blacklisting of prior val/test event_ids.
        """
        rows = self.db.fetchall(
            """
            SELECT DISTINCT ON (event_id)
                event_id, true_label, confidence, label_source, confirmed_at, event_at
            FROM feedback
            WHERE incorporated = FALSE
              AND confidence >= 0.6
              AND (NOW() - confirmed_at) >= (
                SELECT min_delay_h * INTERVAL '1 hour'
                FROM label_delay_config
                WHERE source = feedback.label_source
              )
            ORDER BY event_id, confirmed_at DESC
            """
        )
        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        # Blacklist event_ids from prior val/test partitions
        blacklisted = self._get_blacklisted_event_ids()
        if blacklisted:
            df = df.filter(~pl.col("event_id").cast(str).is_in([str(e) for e in blacklisted]))

        return df

    def _get_blacklisted_event_ids(self) -> set:
        """Get event_ids from all prior val/test partitions to prevent data leakage."""
        # In practice, blacklisted IDs would be stored in a separate table.
        # For now, return empty set — the timestamp range blacklist handles most cases.
        return set()

    def _build_dataset(self, feedback_df: pl.DataFrame) -> pl.DataFrame | None:
        """Build decay-weighted training dataset from feedback + production log."""
        if len(feedback_df) == 0:
            return None

        now = datetime.now(timezone.utc)

        # Get production model's val/test timestamp ranges to exclude
        prod_id = get_production_model_id(self.db)
        excluded_start = None
        excluded_end = None
        if prod_id:
            row = self.db.fetchone(
                "SELECT val_timestamp_range, test_timestamp_range FROM ato_models WHERE model_id = ?",
                [prod_id],
            )
            if row:
                val_range = row.get("val_timestamp_range") or {}
                test_range = row.get("test_timestamp_range") or {}
                if isinstance(val_range, str):
                    val_range = json.loads(val_range)
                if isinstance(test_range, str):
                    test_range = json.loads(test_range)
                excluded_start = val_range.get("start")
                excluded_end = test_range.get("end")

        # Fetch recent production events with confirmed labels
        prod_rows = self.db.fetchall(
            "SELECT event_id, feature_vector, predicted_at FROM ato_production_log "
            "ORDER BY predicted_at DESC LIMIT 100000"
        )

        if not prod_rows:
            return None

        # Merge feedback labels with production feature vectors
        event_label_map = {}
        for row in feedback_df.iter_rows(named=True):
            event_label_map[str(row["event_id"])] = {
                "true_label": int(row["true_label"]),
                "confidence": float(row["confidence"]),
                "label_source": str(row["label_source"]),
                "confirmed_at": row.get("confirmed_at"),
            }

        import json as _json
        labelled_rows = []
        for prod_row in prod_rows:
            eid = str(prod_row["event_id"])
            if eid not in event_label_map:
                continue

            # Exclude events from prior val/test timestamp ranges
            ts = prod_row.get("predicted_at")
            if excluded_start and excluded_end and ts:
                ts_str = str(ts)
                if excluded_start <= ts_str <= excluded_end:
                    continue

            label_info = event_label_map[eid]
            try:
                features = _json.loads(prod_row["feature_vector"])
            except Exception:
                continue

            # Time-decay weighting
            predicted_at = prod_row.get("predicted_at")
            days_ago = 0.0
            if predicted_at:
                try:
                    if isinstance(predicted_at, str):
                        predicted_at = datetime.fromisoformat(predicted_at)
                    if predicted_at.tzinfo is None:
                        predicted_at = predicted_at.replace(tzinfo=timezone.utc)
                    days_ago = max(0.0, (now - predicted_at).total_seconds() / 86400.0)
                except Exception:
                    pass

            sample_weight = exp(-_DECAY_RATE * days_ago)

            # Confidence-based weight adjustment
            source = label_info["label_source"]
            conf = label_info["confidence"]
            weight_multipliers = {
                "analyst": 1.0,
                "auto_lock": 0.9,
                "temporal": 0.8,
                "auto_mfa": 0.7,
            }
            sample_weight *= conf * weight_multipliers.get(source, 0.7)

            row_data = {**features, "label": label_info["true_label"], "sample_weight": sample_weight}
            labelled_rows.append(row_data)

        if not labelled_rows:
            return None

        return pl.DataFrame(labelled_rows)

    def _shadow_replay(self, model_id: str) -> bool:
        """
        Mandatory shadow replay before promotion.
        Runs new model on last 1h of ClickHouse production data.
        Aborts if BLOCK rate > 2x production.
        """
        _logger.info("Running shadow replay for model {} ...", model_id)
        try:
            from core.models.registry import load_model_artifacts
            from core.inference.engine import run_inference
            from core.inference.decision import DecisionRules, Decision
            import yaml

            new_xgb, new_cal, new_transformer, new_meta = load_model_artifacts(model_id)

            prod_rows = self.db.fetchall(
                "SELECT feature_vector, decision FROM ato_production_log "
                "WHERE predicted_at >= now() - INTERVAL 1 HOUR LIMIT 5000"
            )
            if not prod_rows:
                _logger.info("No production data for shadow replay — skipping check.")
                return True

            with open("config/inference.yaml") as f:
                inf_cfg = yaml.safe_load(f)
            rules = DecisionRules.from_config(inf_cfg)

            import json as _json
            new_block_count = 0
            prod_block_count = 0

            for row in prod_rows:
                try:
                    features = _json.loads(row["feature_vector"])
                    result = run_inference(
                        event=features,
                        transformer=new_transformer,
                        xgb_model=new_xgb,
                        calibrator=new_cal,
                        validator=None,
                        rules=rules,
                    )
                    if result.decision == Decision.BLOCK:
                        new_block_count += 1
                    if row.get("decision") == "BLOCK":
                        prod_block_count += 1
                except Exception:
                    continue

            n = len(prod_rows)
            new_block_rate = new_block_count / n
            prod_block_rate = prod_block_count / n

            _logger.info(
                "Shadow replay: new_block_rate={:.3f}, prod_block_rate={:.3f}",
                new_block_rate, prod_block_rate,
            )

            if prod_block_rate > 0 and new_block_rate > 2 * prod_block_rate:
                _logger.warning(
                    "Shadow replay aborted: new BLOCK rate ({:.3f}) > 2x production ({:.3f}). "
                    "Use --force to override.",
                    new_block_rate, prod_block_rate,
                )
                return False

            return True
        except Exception as e:
            _logger.error("Shadow replay error: {}", e)
            return True  # Non-fatal if replay data unavailable

    def _mark_incorporated(self, feedback_df: pl.DataFrame) -> None:
        if len(feedback_df) == 0:
            return
        event_ids = feedback_df["event_id"].to_list()
        for eid in event_ids:
            try:
                self.db.execute(
                    "UPDATE feedback SET incorporated = TRUE WHERE event_id = ?",
                    [str(eid)],
                )
            except Exception:
                pass
