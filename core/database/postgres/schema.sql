-- PostgreSQL schema for vigilant-detect
-- OLTP tables: ato_models, feedback, shadow_predictions, label_delay_config

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Model status lifecycle: staging → production → archived
DO $$ BEGIN
    CREATE TYPE ato_model_status AS ENUM ('staging', 'production', 'archived');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE label_source_enum AS ENUM ('analyst', 'auto_lock', 'auto_mfa', 'temporal');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE OR REPLACE FUNCTION _ato_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$;

-- ---------------------------------------------------------------------------
-- ato_models — file-based registry with PostgreSQL lifecycle tracking
-- One row has status='production' at all times (or none on fresh install).
-- Thresholds and context_rules are stored from training-time tuning on threshold_set.
-- schema_hash is SHA256 of schema.yaml at training time.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ato_models (
    model_id            VARCHAR(255)        PRIMARY KEY,
    status              ato_model_status    NOT NULL DEFAULT 'staging',
    pr_auc              FLOAT,
    ece                 FLOAT,
    f1                  FLOAT,
    n_rows              INTEGER,
    n_pos               INTEGER,
    n_neg               INTEGER,
    trained_at          TIMESTAMPTZ,
    git_sha             VARCHAR(40),
    schema_hash         VARCHAR(64),            -- SHA256 hex of schema.yaml
    feature_names_version VARCHAR(20),
    thresholds          JSONB,                  -- {challenge: float, block: float}
    context_rules       JSONB,                  -- {geo_anomaly_km: float, dormancy_hours: float}
    seeds_pr_auc        JSONB,                  -- {seed: pr_auc} from multi-seed gate
    val_timestamp_range JSONB,                  -- {start: iso, end: iso}
    test_timestamp_range JSONB,                 -- {start: iso, end: iso}
    metadata            JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DROP TRIGGER IF EXISTS trg_ato_models_updated_at ON ato_models;
CREATE TRIGGER trg_ato_models_updated_at
    BEFORE UPDATE ON ato_models
    FOR EACH ROW EXECUTE FUNCTION _ato_set_updated_at();

-- Only one model may have status='production' at a time.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ato_models_single_production
    ON ato_models (status) WHERE status = 'production';

CREATE INDEX IF NOT EXISTS idx_ato_models_status ON ato_models (status, trained_at DESC);

-- ---------------------------------------------------------------------------
-- feedback — label collection for retraining
-- UNIQUE(event_id, confirmed_at) allows revisions while preserving history.
-- Latest-write-wins via query: SELECT DISTINCT ON (event_id) ... ORDER BY confirmed_at DESC
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id            UUID            NOT NULL,
    predicted_label     SMALLINT,
    true_label          SMALLINT        NOT NULL,
    decision            VARCHAR(20),    -- ALLOW | CHALLENGE | BLOCK
    label_source        label_source_enum NOT NULL,
    confidence          FLOAT           NOT NULL,
    event_at            TIMESTAMPTZ,
    confirmed_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    label_delay_h       FLOAT GENERATED ALWAYS AS (
                            EXTRACT(EPOCH FROM (confirmed_at - event_at)) / 3600
                        ) STORED,
    incorporated        BOOLEAN         NOT NULL DEFAULT FALSE,
    UNIQUE(event_id, confirmed_at)
);

CREATE INDEX IF NOT EXISTS idx_feedback_event_id ON feedback (event_id);
CREATE INDEX IF NOT EXISTS idx_feedback_incorporated ON feedback (incorporated, confirmed_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_source ON feedback (label_source, confirmed_at DESC);

-- ---------------------------------------------------------------------------
-- shadow_predictions — for shadow replay and live shadow mode
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shadow_predictions (
    prediction_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id            VARCHAR(255)    REFERENCES ato_models(model_id) ON DELETE CASCADE,
    event_id            UUID,
    risk_score          FLOAT,
    calibrated_prob     FLOAT,
    decision            VARCHAR(20),
    is_degraded         BOOLEAN         DEFAULT FALSE,
    predicted_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shadow_model ON shadow_predictions (model_id, predicted_at DESC);

-- ---------------------------------------------------------------------------
-- label_delay_config — per-source adaptive label delay thresholds
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS label_delay_config (
    source              label_source_enum PRIMARY KEY,
    min_delay_h         FLOAT           NOT NULL,
    mean_delay_h        FLOAT,
    stddev_delay_h      FLOAT,
    p95_delay_h         FLOAT,
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO label_delay_config (source, min_delay_h) VALUES
    ('analyst',   48.0),
    ('auto_lock', 72.0),
    ('temporal',  96.0),
    ('auto_mfa',  24.0)
ON CONFLICT (source) DO NOTHING;
