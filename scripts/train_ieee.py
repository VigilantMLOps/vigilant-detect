"""Train model on synthetic ATO data — no database required.

Uses synthetic-only data (200k events). IEEE-CIS is intentionally excluded:
addr1/addr2 are billing zip codes, not lat/lon, so geo_distance_delta is
meaningless for all 590k IEEE rows and contaminates the strongest feature.

Saves artifacts to models/{model_id}/ and prints the model_id on completion.
Run: poetry run python scripts/train_ieee.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from data.generator.ato_simulator import generate_events
    from data.loaders.unified import load_dataset
    from core.models.trainer import train
    import yaml

    # ── Load data ──────────────────────────────────────────────────────────
    print("Generating 200k synthetic events ...")
    synthetic_df = generate_events(n_total=200_000, seed=42)
    df = load_dataset(synthetic_df=synthetic_df)
    print(f"  Total: {len(df):,} rows  ({int((df['label']==1).sum()):,} positives  "
          f"({100*int((df['label']==1).sum())/len(df):.1f}% rate))")

    # ── Train ──────────────────────────────────────────────────────────────
    with open("config/training.yaml") as f:
        cfg = yaml.safe_load(f)

    print("\nTraining ... (this takes ~30s)")
    result = train(df=df, config=cfg, db=None)  # db=None: skip DB, save to disk only

    print(f"\nDone.")
    print(f"  model_id : {result.model_id}")
    print(f"  PR-AUC   : {result.pr_auc:.4f}")
    print(f"  ECE      : {result.ece:.4f}")
    print(f"  Artifacts: models/{result.model_id}/")
    print(f"\nTo deploy: poetry run python -m cli.main deploy {result.model_id}")


if __name__ == "__main__":
    main()
