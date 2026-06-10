"""Dual-backend database client — PostgreSQL (OLTP) + ClickHouse (OLAP).

Identical routing pattern to vigilant-api. vigilant-detect adds tables:
  PostgreSQL:  ato_models, feedback, shadow_predictions, label_delay_config
  ClickHouse:  ato_production_log, ato_inference_metrics
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

import clickhouse_connect
import psycopg2
import psycopg2.extras

from core.logger import get_logger

_logger = get_logger("vigilant-detect.database")

_PG_SCHEMA = Path(__file__).parent / "postgres" / "schema.sql"
_CH_SCHEMA = Path(__file__).parent / "clickhouse" / "schema.sql"

_CLICKHOUSE_TABLES = frozenset({
    "ato_production_log",
    "ato_inference_metrics",
})

_TABLE_RE = re.compile(
    r'\b(?:INSERT\s+INTO|TRUNCATE\s+TABLE|UPDATE|DELETE\s+FROM|FROM)\s+(\w+)',
    re.IGNORECASE,
)
_BARE_DELETE_RE = re.compile(r'DELETE\s+FROM\s+(\w+)\s*$', re.IGNORECASE)
_INSERT_RE = re.compile(
    r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(',
    re.IGNORECASE,
)


def _table_of(sql: str) -> str | None:
    m = _TABLE_RE.search(sql)
    return m.group(1).lower() if m else None


def _is_clickhouse(sql: str) -> bool:
    table = _table_of(sql)
    return bool(table and table in _CLICKHOUSE_TABLES)


def _to_pg_params(sql: str) -> str:
    return sql.replace("?", "%s")


class Database:
    def __init__(self) -> None:
        self._pg: psycopg2.extensions.connection | None = None
        self._ch_config: dict | None = None

    def startup(self) -> None:
        self._pg = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "vigilant"),
            user=os.getenv("POSTGRES_USER", "vigilant"),
            password=os.getenv("POSTGRES_PASSWORD", "vigilant"),
        )
        self._pg.autocommit = True
        _logger.info("PostgreSQL connection established.")

        ch_db = os.getenv("CLICKHOUSE_DB", "vigilant")
        self._ch_config = {
            "host":     os.getenv("CLICKHOUSE_HOST", "localhost"),
            "port":     int(os.getenv("CLICKHOUSE_PORT", "8123")),
            "database": ch_db,
            "username": os.getenv("CLICKHOUSE_USER", "default"),
            "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        }
        # Create the database first using the default database, then probe the target
        bootstrap = clickhouse_connect.get_client(
            host=self._ch_config["host"],
            port=self._ch_config["port"],
            database="default",
            username=self._ch_config["username"],
            password=self._ch_config["password"],
        )
        bootstrap.command(f"CREATE DATABASE IF NOT EXISTS {ch_db}")
        bootstrap.close()
        probe = clickhouse_connect.get_client(**self._ch_config)
        probe.close()
        _logger.info("ClickHouse connection verified.")

        self._apply_pg_schema()
        self._apply_ch_schema()

    def shutdown(self) -> None:
        if self._pg is not None:
            self._pg.close()
            self._pg = None
        self._ch_config = None

    def execute(self, sql: str, params: list | None = None) -> None:
        params = params or []
        if _is_clickhouse(sql):
            self._ch_execute(sql, params)
        else:
            self._pg_execute(sql, params)

    def fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        params = params or []
        if _is_clickhouse(sql):
            return self._ch_fetchall(sql, params)
        return self._pg_fetchall(sql, params)

    def fetchone(self, sql: str, params: list | None = None) -> dict | None:
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    @property
    def _pg_conn(self) -> psycopg2.extensions.connection:
        if self._pg is None:
            raise RuntimeError("Database not started — call startup() first.")
        return self._pg

    def _pg_execute(self, sql: str, params: list) -> None:
        with self._pg_conn.cursor() as cur:
            cur.execute(_to_pg_params(sql), params)

    def _pg_fetchall(self, sql: str, params: list) -> list[dict]:
        with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_to_pg_params(sql), params)
            return [dict(row) for row in cur.fetchall()]

    def _apply_pg_schema(self) -> None:
        if not _PG_SCHEMA.exists():
            _logger.warning("PostgreSQL schema not found: {}", _PG_SCHEMA)
            return
        with self._pg_conn.cursor() as cur:
            cur.execute(_PG_SCHEMA.read_text())
        _logger.info("PostgreSQL schema applied.")

    @contextmanager
    def _ch_client(self):
        if self._ch_config is None:
            raise RuntimeError("Database not started.")
        client = clickhouse_connect.get_client(**self._ch_config)
        try:
            yield client
        finally:
            client.close()

    def _ch_execute(self, sql: str, params: list) -> None:
        stripped = sql.strip()
        m = _BARE_DELETE_RE.match(stripped)
        if m:
            with self._ch_client() as ch:
                ch.command(f"TRUNCATE TABLE IF EXISTS {m.group(1)}")
            return
        m = _INSERT_RE.match(stripped)
        if m:
            table = m.group(1)
            columns = [c.strip() for c in m.group(2).split(",")]
            with self._ch_client() as ch:
                ch.insert(table, [params], column_names=columns)
            return
        with self._ch_client() as ch:
            ch.command(stripped)

    def _ch_fetchall(self, sql: str, params: list) -> list[dict]:
        with self._ch_client() as ch:
            result = ch.query(sql)
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    def _apply_ch_schema(self) -> None:
        if not _CH_SCHEMA.exists():
            _logger.warning("ClickHouse schema not found: {}", _CH_SCHEMA)
            return
        with self._ch_client() as ch:
            for stmt in _split_statements(_CH_SCHEMA.read_text()):
                ch.command(stmt)
        _logger.info("ClickHouse schema applied.")


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r'--[^\n]*', '', sql)
    return [s.strip() for s in sql.split(';') if s.strip()]


db = Database()
