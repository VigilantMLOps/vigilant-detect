-- ClickHouse schema for vigilant-detect (append-only OLAP)

-- ---------------------------------------------------------------------------
-- ato_production_log — per-request inference record
-- Written after each /predict response (BackgroundTask).
-- Read by shadow replay (last 1h) and monitoring push windows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ato_production_log (
    log_id              UUID                        DEFAULT generateUUIDv4(),
    predicted_at        DateTime64(3, 'UTC')        DEFAULT now64(3),
    event_id            UUID,
    user_id             String                      DEFAULT '',
    model_id            LowCardinality(String)      DEFAULT '',
    model_version       LowCardinality(String)      DEFAULT '',
    schema_hash         LowCardinality(String)      DEFAULT '',
    risk_score          Float32                     DEFAULT 0,
    calibrated_prob     Float32                     DEFAULT 0,
    decision            LowCardinality(String)      DEFAULT '',  -- ALLOW | CHALLENGE | BLOCK
    is_degraded         UInt8                       DEFAULT 0,
    is_cold_start       UInt8                       DEFAULT 0,
    latency_ms          Float32                     DEFAULT 0,
    feature_vector      String                      DEFAULT '{}'  -- JSON for drift push
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(predicted_at)
ORDER BY (model_id, predicted_at, log_id)
TTL toDateTime(predicted_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;

-- ---------------------------------------------------------------------------
-- ato_inference_metrics — P50/P95/P99 flushed every 60s
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ato_inference_metrics (
    metric_id           UUID                        DEFAULT generateUUIDv4(),
    recorded_at         DateTime64(3, 'UTC')        DEFAULT now64(3),
    model_id            LowCardinality(String),
    model_version       LowCardinality(String)      DEFAULT '',
    schema_hash         LowCardinality(String)      DEFAULT '',
    window_size         UInt32,
    p50_ms              Float32,
    p95_ms              Float32,
    p99_ms              Float32,
    degraded_count      UInt32                      DEFAULT 0,
    cold_start_count    UInt32                      DEFAULT 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(recorded_at)
ORDER BY (model_id, recorded_at)
TTL toDateTime(recorded_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;
