"""Push a PRE_PROD report for an already-trained model to vigilant-api.

Loads the model, generates a small test dataset, runs predictions, and calls
/api/v1/reporter/ingest-metrics so the model shows up in the dashboard dropdown.

Usage:
    python scripts/push_model_report.py [model_id]

If model_id is omitted, uses the currently-deployed (production) model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from core.database import db
    db.startup()

    from core.models.registry import get_production_model_id
    model_id = sys.argv[1] if len(sys.argv) > 1 else get_production_model_id(db)
    if model_id is None:
        print("No production model found. Run seed.py first.")
        sys.exit(1)

    print(f"Loading model: {model_id}")
    from core.inference.state import load_and_validate_model
    state = load_and_validate_model(model_id)

    print("Generating test events ...")
    from data.generator.ato_simulator import generate_events
    from data.loaders.unified import load_dataset

    test_df = generate_events(n_total=10_000, seed=99)
    df = load_dataset(synthetic_df=test_df)

    import numpy as np
    from core.features.offline import compute_training_features
    feat_df = compute_training_features(df)
    rows = feat_df.to_dicts()
    X = np.array([state.transformer.transform(r) for r in rows], dtype=np.float32)
    y = feat_df["label"].to_numpy().astype(int)

    print(f"  {len(X):,} events ({int(y.sum()):,} positives) — running predictions ...")
    raw_proba = state.xgb.predict_proba(X)[:, 1]
    calibrated = state.calibrator.predict(raw_proba)

    y_true = y.tolist()
    y_pred_proba = calibrated.tolist()

    import httpx
    vigilant_api_url = os.getenv("VIGILANT_API_URL", "http://localhost:8000")
    print(f"Pushing report to {vigilant_api_url} ...")
    resp = httpx.post(
        f"{vigilant_api_url}/api/v1/reporter/ingest-metrics",
        json={
            "y_true": y_true,
            "y_pred_proba": y_pred_proba,
            "model_version": model_id,
            "schema_hash": state.metadata.get("schema_hash", ""),
            "display_name": "ATO Detector",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    print(
        f"  Done. report_id={data['report_id']}"
        f"  F1={data['f1']:.4f}  ROC-AUC={data['roc_auc']:.4f}"
        f"  avg_precision(PR-AUC)={data['avg_precision']:.4f}"
    )
    db.shutdown()


if __name__ == "__main__":
    main()
