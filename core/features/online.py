"""Online Redis feature fetch — hard 15ms deadline, no retry, sentinels on any failure."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

# Sentinel values for online features — match schema.yaml exactly.
SENTINEL_LAST_LOGIN_GAP: float = -1.0
SENTINEL_GEO_DISTANCE: float = -1.0
SENTINEL_DEVICE_SEEN: float = 0.0

SENTINEL_DICT: dict[str, float] = {
    "last_login_gap_h":    SENTINEL_LAST_LOGIN_GAP,
    "geo_distance_delta":  SENTINEL_GEO_DISTANCE,
    "device_seen_flag":    SENTINEL_DEVICE_SEEN,
}

_REDIS_TIMEOUT = 0.015  # 15ms hard wall-clock deadline


async def fetch_redis_features(
    uid: str,
    device_fp: str,
    redis,
    current_ts: datetime | None = None,
    current_lat: float | None = None,
    current_lon: float | None = None,
) -> tuple[dict[str, float], bool]:
    """
    Fetch 3 Redis keys with a hard 15ms deadline.

    Returns (features, redis_ok) where redis_ok=False means Redis failed
    (timeout, connection error, etc.) — NOT that the user has no prior state.
    A new user with no Redis keys returns redis_ok=True with sentinel values.

    Rules:
    1. asyncio.timeout(0.015) enforces wall-clock deadline — not socket timeout.
    2. On ANY exception: return (SENTINEL_DICT, False). Zero retries.
    3. This function NEVER raises. Caller always receives a valid tuple.
    4. Degraded path (sentinels) is pure dict construction — < 0.1ms.
    """
    try:
        async with asyncio.timeout(_REDIS_TIMEOUT):
            pipe = redis.pipeline()
            pipe.get(f"ato:{uid}:last_ts")
            pipe.get(f"ato:{uid}:last_loc")
            pipe.sismember(f"ato:{uid}:devices", device_fp)
            results = await pipe.execute()
        return _parse_redis_results(results, current_ts, current_lat, current_lon), True
    except Exception:
        return dict(SENTINEL_DICT), False


def _parse_redis_results(
    results: list,
    current_ts: datetime | None,
    current_lat: float | None,
    current_lon: float | None,
) -> dict[str, float]:
    """Parse raw Redis pipeline results into feature dict. Returns sentinels on parse error."""
    try:
        last_ts_raw, last_loc_raw, device_member = results

        # last_login_gap_h
        if last_ts_raw is not None and current_ts is not None:
            last_ts = datetime.fromisoformat(last_ts_raw.decode() if isinstance(last_ts_raw, bytes) else last_ts_raw)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            gap_h = (current_ts - last_ts).total_seconds() / 3600.0
            last_login_gap_h = max(0.0, gap_h)
        else:
            last_login_gap_h = SENTINEL_LAST_LOGIN_GAP

        # geo_distance_delta
        if last_loc_raw is not None and current_lat is not None and current_lon is not None:
            loc_str = last_loc_raw.decode() if isinstance(last_loc_raw, bytes) else last_loc_raw
            prev_lat, prev_lon = (float(x) for x in loc_str.split(","))
            geo_distance_delta = _haversine_km(prev_lat, prev_lon, current_lat, current_lon)
        else:
            geo_distance_delta = SENTINEL_GEO_DISTANCE

        # device_seen_flag
        device_seen_flag = 1.0 if device_member else 0.0

        return {
            "last_login_gap_h":   last_login_gap_h,
            "geo_distance_delta": geo_distance_delta,
            "device_seen_flag":   device_seen_flag,
        }
    except Exception:
        return dict(SENTINEL_DICT)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


async def update_redis_state(
    uid: str,
    device_fp: str,
    timestamp: datetime,
    lat: float | None,
    lon: float | None,
    redis,
) -> None:
    """Update Redis state after a successful login. Runs in BackgroundTask — never blocks response."""
    try:
        async with asyncio.timeout(_REDIS_TIMEOUT):
            pipe = redis.pipeline()
            pipe.set(f"ato:{uid}:last_ts", timestamp.isoformat())
            if lat is not None and lon is not None:
                pipe.set(f"ato:{uid}:last_loc", f"{lat},{lon}")
            pipe.sadd(f"ato:{uid}:devices", device_fp)
            await pipe.execute()
    except Exception:
        pass  # State update failure is silent — not on critical path
