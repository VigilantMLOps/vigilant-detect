"""File-based model registry + PostgreSQL lifecycle management."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from core.logger import get_logger

_logger = get_logger("vigilant-detect.registry")

MODELS_DIR = Path("models")


def model_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id


def save_model_artifacts(
    model_id: str,
    xgb_model: XGBClassifier,
    calibrator: IsotonicRegression,
    transformer,
    metadata: dict,
    schema_yaml_path: str | Path = "core/features/schema.yaml",
) -> Path:
    """Save all model artifacts to models/{model_id}/. Returns artifact directory."""
    d = model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_model, d / "model.pkl")
    joblib.dump(calibrator, d / "calibrator.pkl")
    joblib.dump(transformer, d / "transformer.pkl")

    # Snapshot schema.yaml (not a symlink — prevents schema drift)
    schema_src = Path(schema_yaml_path)
    if schema_src.exists():
        (d / "schema.yaml").write_text(schema_src.read_text())
        schema_hash = hashlib.sha256(schema_src.read_bytes()).hexdigest()
    else:
        schema_hash = ""

    metadata["schema_hash"] = schema_hash
    metadata["model_id"] = model_id
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()

    with open(d / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    _logger.info("Saved model artifacts to {}", d)
    return d


def load_model_artifacts(model_id: str) -> tuple:
    """Load (xgb, calibrator, transformer, metadata) from models/{model_id}/."""
    d = model_dir(model_id)
    if not d.exists():
        raise FileNotFoundError(f"Model artifacts not found: {d}")

    xgb_model = joblib.load(d / "model.pkl")
    calibrator = joblib.load(d / "calibrator.pkl")
    transformer = joblib.load(d / "transformer.pkl")
    with open(d / "metadata.json") as f:
        metadata = json.load(f)

    return xgb_model, calibrator, transformer, metadata


def compute_schema_hash(schema_yaml_path: str | Path = "core/features/schema.yaml") -> str:
    """SHA256 of schema.yaml — stored in metadata.json at training time."""
    with open(schema_yaml_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_model_id() -> str:
    return f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def register_model(db, model_id: str, metadata: dict, status: str = "staging") -> None:
    """Insert model row into PostgreSQL ato_models table."""
    db.execute(
        """INSERT INTO ato_models
           (model_id, status, pr_auc, ece, f1, n_rows, n_pos, n_neg,
            trained_at, git_sha, schema_hash, feature_names_version,
            thresholds, context_rules, seeds_pr_auc,
            val_timestamp_range, test_timestamp_range, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (model_id) DO UPDATE SET
             status = EXCLUDED.status,
             updated_at = NOW()
        """,
        [
            model_id,
            status,
            metadata.get("pr_auc"),
            metadata.get("ece"),
            metadata.get("f1"),
            metadata.get("n_rows"),
            metadata.get("n_pos"),
            metadata.get("n_neg"),
            metadata.get("trained_at"),
            metadata.get("git_sha"),
            metadata.get("schema_hash"),
            metadata.get("feature_names_version"),
            json.dumps(metadata.get("thresholds", {})),
            json.dumps(metadata.get("context_rules", {})),
            json.dumps(metadata.get("seeds_pr_auc", {})),
            json.dumps(metadata.get("val_timestamp_range", {})),
            json.dumps(metadata.get("test_timestamp_range", {})),
            json.dumps(metadata),
        ],
    )
    _logger.info("Registered model {} as {}", model_id, status)


def promote_model(db, model_id: str) -> None:
    """Atomically set model_id to production and archive the previous production model."""
    # Archive current production model
    db.execute(
        "UPDATE ato_models SET status = 'archived' WHERE status = 'production'",
    )
    # Promote new model
    db.execute(
        "UPDATE ato_models SET status = 'production' WHERE model_id = ?",
        [model_id],
    )
    _logger.info("Promoted model {} to production", model_id)


def get_production_model_id(db) -> str | None:
    row = db.fetchone("SELECT model_id FROM ato_models WHERE status = 'production'")
    return row["model_id"] if row else None


def get_model_pr_auc_history(db, n: int = 3) -> list[float]:
    """Return PR-AUC of last n promoted (production/archived) models."""
    rows = db.fetchall(
        "SELECT pr_auc FROM ato_models WHERE status IN ('production', 'archived') "
        "AND pr_auc IS NOT NULL ORDER BY trained_at DESC LIMIT ?",
        [n],
    )
    return [r["pr_auc"] for r in rows]


def get_model_metadata(db, model_id: str) -> dict | None:
    row = db.fetchone("SELECT metadata FROM ato_models WHERE model_id = ?", [model_id])
    if row is None:
        return None
    return json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
