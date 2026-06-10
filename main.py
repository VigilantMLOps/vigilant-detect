"""vigilant-detect — FastAPI application entry point. Port 8001."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.database import db
from core.logger import get_logger
from api.v1 import predict, explain, feedback, model as model_router

_logger = get_logger("vigilant-detect.app")

# Module-level singletons accessed by route dependencies
inference_service = None


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        _logger.debug(
            "{} {} → {} ({:.1f}ms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service

    _logger.info("vigilant-detect starting up (port 8001) ...")
    db.startup()

    # Load inference config
    cfg_path = Path("config/inference.yaml")
    with open(cfg_path) as f:
        inf_cfg = yaml.safe_load(f)

    from core.inference.decision import DecisionRules
    rules = DecisionRules.from_config(inf_cfg)

    # Redis (optional)
    redis = None
    try:
        import redis.asyncio as aioredis
        redis = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=False,
        )
        await redis.ping()
        _logger.info("Redis connected.")
    except Exception as e:
        _logger.warning("Redis unavailable (degraded mode): {}", e)
        redis = None

    from core.monitoring.latency import LatencyTracker
    from core.monitoring.score_stability import ScoreStabilityMonitor
    from services.inference_service import InferenceService

    vigilant_api_url = os.getenv("VIGILANT_API_URL", inf_cfg.get("vigilant_api_url", "http://localhost:8000"))

    inference_service = InferenceService(
        rules=rules,
        db=db,
        redis=redis,
        vigilant_api_url=vigilant_api_url,
        latency_tracker=LatencyTracker(),
        stability_monitor=ScoreStabilityMonitor(),
    )

    # Load production model if one exists
    from core.models.registry import get_production_model_id
    from core.inference.state import load_and_validate_model, swap_model, ModelLoadError
    prod_id = get_production_model_id(db)
    if prod_id:
        try:
            hot_path_rate = inf_cfg.get("feature_contract", {}).get("hot_path_validation_rate", 0.05)
            state = load_and_validate_model(prod_id, hot_path_validation_rate=hot_path_rate)
            swap_model(state)
            _logger.info("Loaded production model: {}", prod_id)
        except ModelLoadError as e:
            _logger.error("Failed to load production model {}: {}", prod_id, e)
    else:
        _logger.warning("No production model found. Deploy one with: vigilant-detect deploy <model_id>")

    # APScheduler for weekly retraining cron
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            _weekly_retrain,
            trigger="cron",
            day_of_week="sun",
            hour=2,
            minute=0,
            id="weekly_retrain",
        )
        scheduler.start()
        _logger.info("APScheduler weekly retraining cron scheduled (Sunday 02:00 UTC).")
    except Exception as e:
        _logger.warning("APScheduler failed to start: {}", e)

    yield

    db.shutdown()
    if redis:
        await redis.aclose()
    _logger.info("vigilant-detect shut down.")


async def _weekly_retrain():
    from services.retraining_service import RetrainingService
    _logger.info("Weekly retraining triggered.")
    svc = RetrainingService(db=db)
    model_id = svc.run()
    if model_id:
        from core.inference.state import load_and_validate_model, swap_model
        state = load_and_validate_model(model_id)
        swap_model(state)
        _logger.info("Weekly retrain complete. New model: {}", model_id)


app = FastAPI(
    title="vigilant-detect",
    description="Real-time Account Takeover detection — inference + training lifecycle.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(LatencyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(feedback.router)
app.include_router(model_router.router)
