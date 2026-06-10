"""In-memory SQLite stub for tests — mirrors vigilant-api pattern."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager

_TABLE_RE = re.compile(
    r'\b(?:INSERT\s+INTO|TRUNCATE\s+TABLE|UPDATE|DELETE\s+FROM|FROM)\s+(\w+)',
    re.IGNORECASE,
)


def _table_of(sql: str) -> str | None:
    m = _TABLE_RE.search(sql)
    return m.group(1).lower() if m else None


class FakeDatabase:
    """
    In-memory SQLite database. Provides the same interface as core.database.Database.
    Skips ClickHouse tables silently (SQLite doesn't support PARTITION BY etc.).
    """

    _CLICKHOUSE_TABLES = frozenset({"ato_production_log", "ato_inference_metrics"})

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        cur = self._conn.cursor()
        # ato_models (simplified)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ato_models (
                model_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'staging',
                pr_auc REAL,
                ece REAL,
                f1 REAL,
                n_rows INTEGER,
                n_pos INTEGER,
                n_neg INTEGER,
                trained_at TEXT,
                git_sha TEXT,
                schema_hash TEXT,
                feature_names_version TEXT,
                thresholds TEXT,
                context_rules TEXT,
                seeds_pr_auc TEXT,
                val_timestamp_range TEXT,
                test_timestamp_range TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # feedback
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                predicted_label INTEGER,
                true_label INTEGER NOT NULL,
                decision TEXT,
                label_source TEXT NOT NULL,
                confidence REAL NOT NULL,
                event_at TEXT,
                confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
                incorporated INTEGER NOT NULL DEFAULT 0,
                UNIQUE(event_id, confirmed_at)
            )
        """)
        # shadow_predictions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shadow_predictions (
                prediction_id TEXT PRIMARY KEY,
                model_id TEXT,
                event_id TEXT,
                risk_score REAL,
                calibrated_prob REAL,
                decision TEXT,
                is_degraded INTEGER DEFAULT 0,
                predicted_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # label_delay_config
        cur.execute("""
            CREATE TABLE IF NOT EXISTS label_delay_config (
                source TEXT PRIMARY KEY,
                min_delay_h REAL NOT NULL,
                mean_delay_h REAL,
                stddev_delay_h REAL,
                p95_delay_h REAL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.executemany(
            "INSERT OR IGNORE INTO label_delay_config (source, min_delay_h) VALUES (?, ?)",
            [("analyst", 48.0), ("auto_lock", 72.0), ("temporal", 96.0), ("auto_mfa", 24.0)],
        )
        self._conn.commit()

    def startup(self) -> None:
        pass  # Already initialised in __init__

    def shutdown(self) -> None:
        pass  # Keep alive for test duration

    def execute(self, sql: str, params: list | None = None) -> None:
        table = _table_of(sql)
        if table in self._CLICKHOUSE_TABLES:
            return  # Silently skip ClickHouse writes
        params = params or []
        # Convert PostgreSQL ? to SQLite ?  (already ?, nothing to do)
        # Convert ON CONFLICT ... DO UPDATE to SQLite upsert
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "no such function: gen_random_uuid" in str(e):
                import uuid
                sql = sql.replace("gen_random_uuid()", f"'{uuid.uuid4()}'")
                cur.execute(sql, params)
                self._conn.commit()
            else:
                raise

    def fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        table = _table_of(sql)
        if table in self._CLICKHOUSE_TABLES:
            return []
        params = params or []
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def fetchone(self, sql: str, params: list | None = None) -> dict | None:
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None
