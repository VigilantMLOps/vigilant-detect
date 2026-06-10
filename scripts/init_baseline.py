"""Compute and push feature baselines to vigilant-api.

Run once after the first model is deployed to establish drift detection baselines.
"""
from __future__ import annotations

import json
from pathlib import Path


def main():
    from core.database import db
    db.startup()

    from core.models.registry import get_production_model_id, load_model_artifacts
    model_id = get_production_model_id(db)
    if model_id is None:
        print("No production model found. Run 'python scripts/seed.py' first.")
        return

    xgb, calibrator, transformer, metadata = load_model_artifacts(model_id)

    # Fetch recent production log for baseline computation
    rows = db.fetchall(
        "SELECT feature_vector FROM ato_production_log "
        "ORDER BY predicted_at DESC LIMIT 10000"
    )
    if not rows:
        print("No production log data. Generate some predictions first.")
        return

    import numpy as np
    feature_names = transformer.feature_names_out
    feature_stats = {name: {"values": []} for name in feature_names}

    for row in rows:
        try:
            fv = json.loads(row["feature_vector"])
            vec = transformer.transform(fv)
            for i, name in enumerate(feature_names):
                feature_stats[name]["values"].append(float(vec[i]))
        except Exception:
            continue

    baseline = {}
    for name, data in feature_stats.items():
        vals = np.array(data["values"])
        if len(vals) > 0:
            baseline[name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p5": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
                "n": len(vals),
            }

    # Push baseline to vigilant-api
    import httpx, yaml, hashlib
    schema_path = Path("core/features/schema.yaml")
    schema_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()

    with open("config/inference.yaml") as f:
        inf_cfg = yaml.safe_load(f)
    vigilant_api_url = inf_cfg.get("vigilant_api_url", "http://localhost:8000")

    payload = {
        "schema_hash": schema_hash,
        "feature_names_version": metadata.get("feature_names_version", "v1"),
        "model_version": model_id,
        "baseline": baseline,
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{vigilant_api_url}/api/v1/reporter/evaluate-drift",
                json={**payload, "window_start": None, "window_end": None,
                      "event_count": len(rows), "feature_stats": baseline},
            )
            print(f"Baseline pushed: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Could not push baseline to vigilant-api: {e}")

    db.shutdown()
    print("Baseline initialization complete.")


if __name__ == "__main__":
    main()
