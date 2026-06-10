"""Bootstrap script: generate → train → deploy → push baselines to vigilant-api."""
from __future__ import annotations

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

    print("Seed complete.")
    db.shutdown()


if __name__ == "__main__":
    main()
