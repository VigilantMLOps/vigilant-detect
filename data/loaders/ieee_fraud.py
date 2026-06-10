"""IEEE-CIS fraud dataset loader — explicit mapping contract to login schema.

IEEE-CIS is not structurally equivalent to login events. It is used as a
pretraining signal source (scale + base-rate signal) through this defined
mapping layer. Fields without plausible mappings are dropped.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

# Reference epoch for TransactionDT (seconds since 2017-11-30 based on competition data)
_REFERENCE_EPOCH = datetime(2017, 11, 30, tzinfo=timezone.utc)

# IEEE-CIS DeviceType → UA family bucket
_DEVICE_TYPE_MAP = {
    "mobile": "mobile",
    "desktop": "desktop",
}

# card4 (card network) → approximate geo_country proxy
_CARD4_COUNTRY_MAP = {
    "visa":       "US",
    "mastercard": "US",
    "american express": "US",
    "discover":   "US",
}


def load_ieee_fraud(
    transaction_path: str | Path,
    identity_path: str | Path | None = None,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """
    Load IEEE-CIS fraud CSV and map to login schema fields.

    Returns a DataFrame with columns compatible with the login event schema.
    Carries inherent proxy noise — intentional. IEEE-CIS contributes scale
    and base rate signal; the synthetic generator provides ATO pattern signal.
    """
    df = pl.read_csv(str(transaction_path), n_rows=max_rows, infer_schema_length=10000)

    if identity_path is not None:
        id_df = pl.read_csv(str(identity_path), n_rows=max_rows, infer_schema_length=10000)
        df = df.join(id_df, on="TransactionID", how="left")

    rows = []
    for row in df.iter_rows(named=True):
        mapped = _map_row(row)
        if mapped is not None:
            rows.append(mapped)

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("timestamp")


def _map_row(row: dict) -> dict | None:
    """Map a single IEEE-CIS row to login schema. Returns None if row is invalid."""
    try:
        # timestamp: reference_epoch + TransactionDT seconds
        tx_dt = row.get("TransactionDT")
        if tx_dt is None:
            return None
        timestamp = _REFERENCE_EPOCH + timedelta(seconds=float(tx_dt))

        # user_id: hash of card1+card2+card3 (proxy for account identity)
        card_key = f"{row.get('card1', '')}_{row.get('card2', '')}_{row.get('card3', '')}"
        user_id = f"ieee_{abs(hash(card_key)) % 1_000_000:06d}"

        # geo_country: from card4 (card network → rough region proxy)
        card4 = str(row.get("card4", "")).lower()
        geo_country = _CARD4_COUNTRY_MAP.get(card4, "US")

        # login_success: inverse of isFraud (fraud = failed legitimate login)
        is_fraud = int(row.get("isFraud", 0))
        login_success = is_fraud == 0

        # mfa_method proxy: if P_emaildomain present → email; else none
        p_email = row.get("P_emaildomain")
        mfa_method = "email" if p_email and str(p_email) != "nan" else "none"

        # device_fingerprint: DeviceInfo proxy
        device_info = str(row.get("DeviceInfo", "unknown"))
        device_fp = f"ieee_device_{abs(hash(device_info)) % 10000:04d}"

        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "user_id": user_id,
            "session_id": str(uuid.uuid4()),
            "ip_address": f"ieee_{row.get('addr1', '0')}.{row.get('addr2', '0')}",
            "geo_country": geo_country,
            "geo_lat": float("nan"),   # IEEE addr1/addr2 are zip codes, not lat/lon
            "geo_lon": float("nan"),
            "login_success": login_success,
            "mfa_used": mfa_method != "none",
            "mfa_method": mfa_method,
            "login_duration_ms": float(row.get("TransactionAmt", 1000) or 1000),
            "device_fingerprint": device_fp,
            "label": is_fraud,
        }
    except Exception:
        return None
