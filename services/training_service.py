"""Training service — orchestrates the full training pipeline."""
from __future__ import annotations

import yaml
from pathlib import Path

from core.logger import get_logger

_logger = get_logger("vigilant-detect.training")

_DEFAULT_CONFIG = Path("config/training.yaml")


class TrainingService:
    def __init__(self, db, config_path: str | Path | None = None) -> None:
        self.db = db
        self.config_path = Path(config_path or _DEFAULT_CONFIG)

    def load_config(self, overrides: dict | None = None) -> dict:
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        if overrides:
            cfg.update(overrides)
        return cfg

    def run(
        self,
        df=None,
        data_path: str | Path | None = None,
        config_overrides: dict | None = None,
    ):
        """Run the full training pipeline. Returns TrainingResult."""
        from core.models.trainer import train
        from data.loaders.unified import load_dataset
        from data.generator.ato_simulator import generate_events

        cfg = self.load_config(config_overrides)

        if df is None:
            if data_path is not None:
                import polars as pl
                df = pl.read_parquet(str(data_path))
            else:
                _logger.info("No data provided — generating synthetic dataset ...")
                df = generate_events(
                    n_total=cfg.get("synthetic", {}).get("n_total", 50_000),
                    seed=cfg.get("synthetic", {}).get("seed", 42),
                )
                df = load_dataset(synthetic_df=df)

        _logger.info("Starting training on {} rows ...", len(df))
        result = train(df=df, config=cfg, db=self.db)
        _logger.info(
            "Training complete: model_id={} PR-AUC={:.4f} ECE={:.4f}",
            result.model_id, result.pr_auc, result.ece,
        )
        return result
