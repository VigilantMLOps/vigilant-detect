#!/bin/bash
set -e

PORT="${PORT:-8001}"

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --loop asyncio
