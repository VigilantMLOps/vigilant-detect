"""Bootstrap script: generate → train → deploy → push initial report to vigilant-api."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from core.database import db
    db.startup()

    print("Step 1: Loading data ...")
    from data.generator.ato_simulator import generate_events
    from data.loaders.unified import load_dataset

    # Synthetic-only: features (geo, device, gap) are fully populated for synthetic data.
    # IEEE-CIS addr1/addr2 are zip codes, not lat/lon — geo_distance_delta is meaningless
    # for IEEE rows, contaminating the strongest feature signal.
    synthetic_df = generate_events(n_total=200_000, seed=42)
    print(f"  Synthetic: {len(synthetic_df):,} events ({int((synthetic_df['label']==1).sum()):,} ATO)")

    df = load_dataset(synthetic_df=synthetic_df)
    print(f"  Total: {len(df):,} events ({int((df['label']==1).sum()):,} ATO)")

    print("Step 2: Training model ...")
    from services.training_service import TrainingService
    svc = TrainingService(db=db)
    result = svc.run(df=df)
    print(f"  Trained model_id={result.model_id} PR-AUC={result.pr_auc:.4f}")

    print("Step 3: Deploying model ...")
    from core.models.registry import promote_model
    promote_model(db, result.model_id)

    from core.inference.state import load_and_validate_model, swap_model
    state = load_and_validate_model(result.model_id)
    swap_model(state)
    print(f"  Deployed {result.model_id}")

    print("Step 4: Pushing initial evaluation report to vigilant-api ...")
    _push_initial_report(result)

    print("Seed complete.")
    db.shutdown()


def _push_initial_report(result) -> None:
    import httpx

    vigilant_api_url = os.getenv("VIGILANT_API_URL", "http://localhost:8000")

    if result.y_true_test is None or result.y_pred_proba_test is None:
        print("  Skipping push — test metrics not available in TrainingResult.")
        return

    try:
        resp = httpx.post(
            f"{vigilant_api_url}/api/v1/reporter/ingest-metrics",
            json={
                "y_true": result.y_true_test,
                "y_pred_proba": result.y_pred_proba_test,
                "model_version": result.model_id,
                "schema_hash": result.metadata.get("schema_hash", ""),
                "display_name": "ATO Detector",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        print(
            f"  Report pushed: report_id={data['report_id']}"
            f"  F1={data['f1']:.4f}  ROC-AUC={data['roc_auc']:.4f}"
            f"  PR-AUC(train)={result.pr_auc:.4f}"
        )
    except Exception as exc:
        print(f"  Warning: could not push report to vigilant-api ({exc}). Dashboard may be empty.")


if __name__ == "__main__":
    main()
