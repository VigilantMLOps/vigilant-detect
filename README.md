# vigilant-detect

ML inference service for Account Takeover (ATO) detection. Part of the Vigilant MLOps platform.

vigilant-detect owns the full ML lifecycle — training, serving, retraining, and explainability. It exposes a REST API on port 8001 and integrates with **vigilant-api** (the observability plane) by pushing evaluation results and drift statistics in 500-event windows.

## Architecture

```
vigilant-api  →  POST /api/v1/events/login  →  vigilant-detect :8001
                                                     │
                                 ┌───────────────────┤
                                 │                   │
                             Redis (online       PostgreSQL +
                             features:           ClickHouse (shared
                             last_login,         with vigilant-api):
                             geo_delta,          ato_models, feedback,
                             device_flag)        ato_production_log
                                                     │
                                 ┌───────────────────┘
                                 │  per-500-event window push
                                 ↓
                             vigilant-api :8000
                             (observability plane)
```

**vigilant-api** is the observability plane — it handles monitoring, drift detection, incident alerting, and the React dashboard. vigilant-detect is the intelligence plane — it scores every login event and pushes aggregated metrics to vigilant-api.

Both services share the same PostgreSQL and ClickHouse instances. Redis is local to vigilant-detect.

### Inference hot path (P95 < 50ms)

1. Pydantic schema validation
2. Deterministic features (hour_sin/cos, dow_sin/cos — no I/O)
3. Redis fetch with 15ms hard deadline via `asyncio.timeout` → sentinels on failure
4. Assemble feature vector via `FeatureTransformer` (stateless, no `.fit()`)
5. `FeatureContractValidator` — sampling mode (5% of requests in production)
6. `xgb.predict_proba()` — in-memory XGBoost
7. `isotonic.predict()` — probability calibration
8. Decision logic: `ALLOW / CHALLENGE / BLOCK` (probability + context rules from config)
9. Serialize response
10. Background: ClickHouse log + Redis state update + monitoring window accumulation

SHAP is permanently excluded from the hot path. It runs only on `POST /explain`.

### Cold-start vs degraded — distinct states

| State | Cause | `is_cold_start` | `degraded` |
|---|---|---|---|
| New user | No prior history in the system | `1` | `false` |
| Redis down | Redis unavailable or timed out | unchanged | `true` |

These are not the same. `is_cold_start` is a model feature. `degraded` is an infrastructure flag. A new user on a healthy system gets `is_cold_start=1, degraded=false`. A known user during a Redis outage gets `is_cold_start=0, degraded=true`. The model was trained with both scenarios (15% Redis dropout simulation) so both produce valid predictions.

## Tech Stack

- **Python 3.12** + **FastAPI** + **Uvicorn**
- **XGBoost 2.x** — tabular binary classifier with `scale_pos_weight` for class imbalance
- **scikit-learn** — `IsotonicRegression` for probability calibration
- **SHAP** — `TreeExplainer` for `/explain` endpoint
- **Polars** — batch feature computation with strict per-row time cutoffs
- **Redis (redis-py async)** — online feature store (last login, geo delta, device fingerprint)
- **PostgreSQL 16** — model registry (`ato_models`), feedback, shadow predictions
- **ClickHouse 24** — production inference log (`ato_production_log`)
- **APScheduler** — weekly retraining cron
- **Typer** — CLI (`python -m cli.main`)
- **Poetry** — dependency management
- **Docker Compose** — runs as part of vigilant-api's shared compose stack in production

## Quick Start (Local)

```bash
cp .env.example .env          # fill in passwords if needed
make install                  # poetry install
make up                       # start Redis (postgres/clickhouse come from vigilant-api)
make seed                     # generate → train → deploy a baseline model
make run                      # start the service on :8001 with hot reload
```

First-time flow assumes vigilant-api is already running (postgres + clickhouse must be accessible). See [Environment Variables](#environment-variables) if the databases are not on localhost.

The service starts in `no_model` status if no model has been deployed. Call `make seed` to bootstrap.

Health check:
```bash
curl http://localhost:8001/health
```

Score a login event:
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_abc",
    "ip_address": "198.51.100.42",
    "device_fingerprint": "fp_xyz",
    "login_success": true,
    "geo_lat": 37.7749,
    "geo_lon": -122.4194
  }' | python3 -m json.tool
```

## API Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service status + model version |
| `GET` | `/model/info` | Model metadata, thresholds, PR-AUC, ECE |
| `POST` | `/predict` | Score a single login event → ALLOW / CHALLENGE / BLOCK |
| `POST` | `/predict/batch` | Score up to 500 events in one call |
| `POST` | `/feedback` | Submit a ground-truth label for retraining |
| `POST` | `/explain` | SHAP-based explanation for a login event (not latency-bound) |

### POST /predict

**Request** — all fields except `user_id` are optional:

```json
{
  "user_id": "hashed-uid",
  "session_id": "sess-123",
  "ip_address": "198.51.100.42",
  "geo_country": "US",
  "geo_lat": 37.77,
  "geo_lon": -122.41,
  "user_agent": "Mozilla/5.0...",
  "device_fingerprint": "fp-abc",
  "login_success": true,
  "mfa_used": false,
  "mfa_method": "none",
  "login_duration_ms": 842.0,
  "account_age_days": 120,
  "failed_attempts_7d": 0,
  "distinct_ips_7d": 1,
  "login_success_rate_30d": 0.98,
  "avg_login_hour_7d": 9.5
}
```

**Response:**

```json
{
  "event_id": "uuid",
  "decision": "CHALLENGE",
  "risk_score": 0.38,
  "calibrated_probability": 0.38,
  "confidence": "medium",
  "context_flags": ["new_device"],
  "degraded": false,
  "model_version": "v1"
}
```

`decision` is one of `ALLOW`, `CHALLENGE`, or `BLOCK`.

`degraded: true` means Redis was unavailable — the decision is still valid, made on offline features only.

### POST /feedback

```json
{
  "event_id": "uuid-of-original-event",
  "true_label": 1,
  "label_source": "analyst",
  "confidence": 0.95
}
```

Accepted label sources: `analyst`, `auto_lock`, `auto_mfa`, `temporal`. Each has a minimum confidence threshold and adaptive label delay before it enters the retraining pool.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8001` | Port the service listens on |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host (shared with vigilant-api) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `vigilant` | Database name |
| `POSTGRES_USER` | `vigilant` | Database user |
| `POSTGRES_PASSWORD` | `vigilant` | Database password |
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse host (shared with vigilant-api) |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DB` | `vigilant` | ClickHouse database |
| `CLICKHOUSE_USER` | `default` | ClickHouse user |
| `CLICKHOUSE_PASSWORD` | (empty) | ClickHouse password |
| `REDIS_HOST` | `localhost` | Redis host (local to vigilant-detect) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database index |
| `REDIS_PASSWORD` | (empty) | Redis password |
| `VIGILANT_API_URL` | `http://localhost:8000` | vigilant-api base URL for monitoring pushes |
| `MODEL_ARTIFACTS_DIR` | `models` | Directory where model artifacts are stored |

Minimal `.env` for local development against a running vigilant-api:

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=vigilant
CLICKHOUSE_HOST=localhost
REDIS_HOST=localhost
VIGILANT_API_URL=http://localhost:8000
```

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Install dependencies via Poetry |
| `make up` | Start Redis (infrastructure) |
| `make down` | Stop containers |
| `make logs` | Tail docker-compose logs |
| `make test` | Run full test suite |
| `make test-fast` | Run tests excluding adversarial suite |
| `make lint` | Syntax check all Python modules |
| `make seed` | Full bootstrap: generate → train → deploy |
| `make train` | Train via CLI (requires running DB) |
| `make deploy MODEL_ID=<id>` | Promote a staged model to production |
| `make rollback MODEL_ID=<id>` | Roll back to a previous model |
| `make run` | Start on :8001 with hot reload (dev) |
| `make run-prod` | Start on :8001 (production) |

## Project Layout

```
vigilant-detect/
├── api/v1/
│   ├── predict.py          # POST /predict, POST /predict/batch
│   ├── explain.py          # POST /explain (SHAP)
│   ├── feedback.py         # POST /feedback
│   └── model.py            # GET /health, GET /model/info
├── core/
│   ├── features/
│   │   ├── schema.yaml     # Authoritative: feature order, dtypes, sentinels
│   │   ├── transformer.py  # FeatureTransformer — stateless, no .fit()
│   │   ├── offline.py      # Polars rolling aggregations with strict time cutoff
│   │   ├── online.py       # Redis async fetch (15ms deadline)
│   │   └── contract.py     # FeatureContractValidator (sampling mode)
│   ├── inference/
│   │   ├── engine.py       # Stateless: transform → predict → calibrate
│   │   ├── decision.py     # ALLOW/CHALLENGE/BLOCK logic
│   │   └── state.py        # ModelState + hot-swap
│   ├── models/
│   │   ├── trainer.py      # Full 5-split training pipeline
│   │   ├── calibrator.py   # IsotonicRegression on val_calibration only
│   │   ├── evaluator.py    # PR-AUC gate, ECE gate, multi-seed stability
│   │   └── registry.py     # File + PostgreSQL model lifecycle
│   ├── monitoring/
│   │   ├── latency.py      # Ring buffer P50/P95/P99
│   │   ├── payload.py      # Monitoring push payload builders
│   │   └── score_stability.py
│   └── database/           # Dual-backend: PostgreSQL (ato_models, feedback) +
│                           # ClickHouse (ato_production_log)
├── data/
│   ├── generator/          # Synthetic ATO event generator
│   └── loaders/            # IEEE-CIS mapping + unified loader
├── services/
│   ├── inference_service.py
│   ├── training_service.py
│   └── retraining_service.py
├── cli/main.py             # Typer CLI: train, deploy, rollback, status
├── scripts/seed.py         # Bootstrap: generate → train → deploy
├── config/
│   ├── training.yaml       # Gates, seeds, XGBoost hyperparams
│   └── inference.yaml      # Thresholds, feature contract settings
├── models/                 # Model artifacts (gitignored)
├── tests/
└── main.py                 # FastAPI app
```

## Training Pipeline

The training pipeline uses a 5-way temporal split to prevent data leakage. Feature computation happens independently within each partition — never on the full dataset.

```
raw events (sorted by timestamp)
  ├── train           70%  → XGBoost.fit()
  ├── val_stop         5%  → early stopping signal only
  ├── val_calibration  5%  → IsotonicRegression.fit()
  ├── threshold_set    5%  → threshold tuning (ALLOW/CHALLENGE/BLOCK boundaries)
  └── test            15%  → final evaluation, read once
```

Gates that must pass before a model is promoted:
- **ECE ≤ 0.10** on `val_calibration` (hard fail — uncalibrated model makes thresholds meaningless)
- **PR-AUC improvement ≥ 2%** over the current production model on the test set
- **Multi-seed stability**: 3 fixed seeds (42, 137, 2024), all must individually pass the PR-AUC gate
- **Rolling gate**: challenger must beat the rolling average of the last 3 promoted models

Run a training cycle:
```bash
poetry run python -m cli.main train
```

Deploy the output:
```bash
poetry run python -m cli.main deploy <model-id>
```

## Decision Logic

```
prob >= threshold_block         →  BLOCK  (probability only, context rules never produce BLOCK)
prob >= threshold_challenge     →  CHALLENGE
rule_triggered                  →  CHALLENGE  (max escalation: ALLOW → CHALLENGE)
default                         →  ALLOW
```

Rule triggers: unknown device (`device_seen_flag=0`), large geo jump (`geo_distance_delta > geo_anomaly_km`), long dormancy (`last_login_gap_h > dormancy_hours`). Rules escalate at most one level — they never produce `BLOCK`.

All thresholds are tuned on `threshold_set` during training and written to `metadata.json`. Runtime values are read from `config/inference.yaml`, which is overwritten at deploy time.

## CI/CD

Tag-triggered via GitHub Actions:

```bash
git tag deploy.ml.$(date +%Y%m%d) && git push origin --tags
```

The workflow:
1. Runs the test suite (excluding adversarial)
2. SSHes into the Oracle VM
3. Pulls the tagged commit, rebuilds the Docker image from vigilant-api's compose stack
4. Auto-seeds (`make seed`) on first deploy if no model is loaded

CI uses Poetry 2.3.4 — the same version pinned in the Dockerfile.

## Testing

```bash
make test           # full suite (includes adversarial)
make test-fast      # excludes adversarial (faster CI)
```

Tests use `FakeDatabase` (SQLite in-memory) and `FakeRedis` stubs — no real databases required. The full suite runs in under 1 second.

Key test files:
- `tests/test_features.py` — transformer correctness, time cutoff, Redis dropout
- `tests/test_inference.py` — warm steady-state latency, degraded mode
- `tests/test_decision.py` — context rule escalation cap, threshold edge cases
- `tests/test_adversarial.py` — poisoned features, Redis outage, sentinel dominance
- `tests/test_training.py` — PR-AUC gate, ECE gate, partition identity checks

## Integration with vigilant-api

vigilant-detect integrates with vigilant-api in two directions:

**Incoming** — login events arrive via vigilant-api's ingestion endpoint:
```
POST https://vigilant-api.duckdns.org/api/v1/events/login
```
vigilant-api proxies the request to `http://vigilant-detect:8001/predict` and logs the result to ClickHouse.

**Outgoing** — vigilant-detect pushes monitoring data to vigilant-api every 500 events:
- `POST /api/v1/reporter/evaluate-model` — `y_true` / `y_pred` pairs for model performance tracking
- `POST /api/v1/reporter/evaluate-drift` — feature distribution statistics for drift detection

Push failures are logged and retried but never block inference.
