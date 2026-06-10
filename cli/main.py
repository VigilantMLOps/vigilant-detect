"""vigilant-detect CLI — train, evaluate, deploy, rollback, generate, retrain, status."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="vigilant-detect", help="ATO detection ML service CLI")


def _get_db():
    from core.database import db
    db.startup()
    return db


@app.command()
def train(
    data_path: str = typer.Option(None, help="Path to Parquet training data. Uses synthetic if omitted."),
    config: str = typer.Option("config/training.yaml", help="Training config YAML path."),
):
    """Run the full training pipeline (5-way split, gates, artifact save)."""
    db = _get_db()
    from services.training_service import TrainingService
    svc = TrainingService(db=db, config_path=config)
    result = svc.run(data_path=data_path)
    typer.echo(f"Training complete. model_id={result.model_id} PR-AUC={result.pr_auc:.4f} ECE={result.ece:.4f}")


@app.command()
def evaluate(model_id: str = typer.Argument(..., help="Model ID to evaluate.")):
    """Evaluate a staged model on its held-out test set."""
    db = _get_db()
    row = db.fetchone("SELECT metadata FROM ato_models WHERE model_id = ?", [model_id])
    if row is None:
        typer.echo(f"Model {model_id} not found.", err=True)
        raise typer.Exit(1)
    import json
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    typer.echo(f"model_id: {model_id}")
    typer.echo(f"pr_auc:   {meta.get('pr_auc')}")
    typer.echo(f"ece:      {meta.get('ece')}")
    typer.echo(f"status:   {row.get('status', 'unknown')}")


@app.command()
def deploy(
    model_id: str = typer.Argument(..., help="Model ID to deploy."),
    force: bool = typer.Option(False, "--force", help="Skip shadow replay gate."),
):
    """
    Deploy a staged model to production.
    Runs mandatory shadow replay unless --force.
    Writes tuned thresholds to config/inference.yaml.
    """
    db = _get_db()
    from core.models.registry import (
        load_model_artifacts, promote_model, get_model_metadata,
    )

    row = db.fetchone("SELECT status FROM ato_models WHERE model_id = ?", [model_id])
    if row is None:
        typer.echo(f"Model {model_id} not found.", err=True)
        raise typer.Exit(1)

    if not force:
        from services.retraining_service import RetrainingService
        svc = RetrainingService(db=db)
        if not svc._shadow_replay(model_id):
            typer.echo("Shadow replay failed. Use --force to override.", err=True)
            raise typer.Exit(1)

    promote_model(db, model_id)

    # Write tuned thresholds to config/inference.yaml
    meta = get_model_metadata(db, model_id)
    if meta:
        _update_inference_config(meta)

    # Load model into memory
    from core.inference.state import load_and_validate_model, swap_model
    try:
        state = load_and_validate_model(model_id)
        swap_model(state)
        typer.echo(f"Model {model_id} deployed and loaded into memory.")
    except Exception as e:
        typer.echo(f"Warning: model deployed to DB but failed to load in memory: {e}", err=True)


@app.command()
def rollback(model_id: str = typer.Argument(..., help="Model ID to roll back to.")):
    """Roll back to a previously archived model."""
    db = _get_db()
    from core.models.registry import promote_model
    promote_model(db, model_id)
    typer.echo(f"Rolled back to model {model_id}.")


@app.command()
def generate(
    n_total: int = typer.Option(50_000, help="Total events to generate."),
    output: str = typer.Option("data/raw/synthetic.parquet", help="Output path."),
    seed: int = typer.Option(42, help="Random seed."),
):
    """Generate synthetic ATO training data."""
    from data.generator.ato_simulator import generate_events
    from data.loaders.unified import load_dataset
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_events(n_total=n_total, seed=seed)
    df = load_dataset(synthetic_df=df)
    df.write_parquet(output)
    n_pos = int((df["label"] == 1).sum())
    typer.echo(f"Generated {len(df)} events ({n_pos} ATO) → {output}")


@app.command()
def retrain():
    """Run the retraining pipeline (adaptive delay, dedup, decay weighting)."""
    db = _get_db()
    from services.retraining_service import RetrainingService
    svc = RetrainingService(db=db)
    model_id = svc.run()
    if model_id:
        typer.echo(f"Retraining complete. Promoted model_id={model_id}")
    else:
        typer.echo("Retraining did not produce a new model (gate failed or insufficient data).")


@app.command()
def status():
    """Show current model status."""
    db = _get_db()
    rows = db.fetchall(
        "SELECT model_id, status, pr_auc, ece, trained_at FROM ato_models "
        "ORDER BY trained_at DESC LIMIT 5"
    )
    if not rows:
        typer.echo("No models registered.")
        return
    for r in rows:
        typer.echo(
            f"  {r['status']:12s}  {r['model_id'][:40]}  PR-AUC={r.get('pr_auc', 'N/A')}  "
            f"ECE={r.get('ece', 'N/A')}  trained={r.get('trained_at', 'N/A')}"
        )


def _update_inference_config(metadata: dict) -> None:
    """Write tuned thresholds from training to config/inference.yaml."""
    import yaml
    cfg_path = Path("config/inference.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    thresholds = metadata.get("thresholds", {})
    rules = metadata.get("context_rules", {})

    cfg["decision"]["threshold_challenge"] = thresholds.get("challenge", cfg["decision"]["threshold_challenge"])
    cfg["decision"]["threshold_block"] = thresholds.get("block", cfg["decision"]["threshold_block"])
    cfg["decision"]["rules"]["geo_anomaly_km"] = rules.get("geo_anomaly_km", cfg["decision"]["rules"]["geo_anomaly_km"])
    cfg["decision"]["rules"]["dormancy_hours"] = rules.get("dormancy_hours", cfg["decision"]["rules"]["dormancy_hours"])

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


if __name__ == "__main__":
    app()
