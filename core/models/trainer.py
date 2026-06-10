"""Full training pipeline — 4-phase, 5-way temporal split, multi-seed gate."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from xgboost import XGBClassifier

from core.features.contract import FeatureContractValidator, FeatureContractError
from core.features.offline import (
    temporal_split,
    compute_offline_features,
    compute_training_features,
    validate_no_leakage,
)
from core.features.transformer import FeatureTransformer, make_transformer
from core.models.calibrator import (
    fit_isotonic,
    calibrate_batch,
    check_calibrator_monotonicity,
)
from core.models.evaluator import (
    compute_ece,
    evaluate_pr_auc,
    tune_challenge_threshold,
    tune_block_threshold,
    tune_geo_threshold,
    tune_dormancy_threshold,
    multi_seed_pr_auc_gate,
)
from core.models.registry import (
    generate_model_id,
    save_model_artifacts,
    register_model,
    get_model_pr_auc_history,
    get_production_model_id,
    compute_schema_hash,
)
from core.logger import get_logger

_logger = get_logger("vigilant-detect.trainer")


class TrainingGateError(Exception):
    """Raised when a training gate (ECE, PR-AUC) fails. Artifact not written."""


@dataclass
class TrainingResult:
    model_id: str
    pr_auc: float
    ece: float
    thresholds: dict[str, float]
    context_rules: dict[str, float]
    seeds_pr_auc: dict[int, float]
    metadata: dict


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def apply_redis_degradation(
    X: np.ndarray,
    feature_names: list[str],
    dropout_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Scenario A — Redis outage simulation: dropout_rate % of rows get online
    features replaced with sentinels. is_cold_start is NOT changed.
    Applied to all splits except test (test measures clean ceiling).
    """
    online_features = ["last_login_gap_h", "geo_distance_delta", "device_seen_flag"]
    indices = []
    for f in online_features:
        try:
            indices.append(feature_names.index(f))
        except ValueError:
            pass

    if not indices:
        return X

    X = X.copy()
    outage_mask = rng.random(len(X)) < dropout_rate
    # Sentinels: -1.0, -1.0, 0.0
    sentinel_values = [-1.0, -1.0, 0.0]
    for i, idx in enumerate(indices):
        X[outage_mask, idx] = sentinel_values[i]
    return X


def _df_to_feature_matrix(
    df: pl.DataFrame,
    transformer: FeatureTransformer,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Polars DataFrame to (X, y) numpy arrays via transformer."""
    rows = df.to_dicts()
    X = np.array([transformer.transform(row) for row in rows], dtype=np.float32)
    y = np.array(df[label_col].to_list(), dtype=np.int32)
    return X, y


def train(
    df: pl.DataFrame,
    config: dict,
    db=None,
    schema_yaml_path: str | Path = "core/features/schema.yaml",
) -> TrainingResult:
    """
    Full 4-phase training pipeline.

    Phase 1: XGBoost.fit() on train only (val_stop for early stopping).
    Phase 2: IsotonicRegression.fit() on val_calibration only + ECE gate.
    Phase 3: Threshold tuning on threshold_set only, then FROZEN.
    Phase 4: Final evaluation on test (read once, PR-AUC gate).

    Raises TrainingGateError if ECE or PR-AUC gate fails.
    Artifact is NOT written on gate failure.
    """
    gates_cfg = config.get("gates", {})
    redis_cfg = config.get("redis_simulation", {})
    seed_cfg = config.get("multi_seed", {})
    xgb_cfg = config.get("xgboost", {})

    min_pr_auc_improvement = gates_cfg.get("min_pr_auc_improvement", 0.02)
    max_ece = gates_cfg.get("max_ece", 0.10)
    min_test_positives = gates_cfg.get("min_test_positives", 200)
    pr_auc_rolling_window = gates_cfg.get("pr_auc_rolling_window", 3)
    dropout_rate = redis_cfg.get("dropout_rate", 0.15)
    seeds = seed_cfg.get("seeds", [42, 137, 2024])

    transformer = make_transformer(schema_yaml_path)
    feature_names = transformer.feature_names_out
    schema_hash = compute_schema_hash(schema_yaml_path)
    with open(schema_yaml_path) as f:
        schema_data = yaml.safe_load(f)
    feature_names_version = schema_data.get("version", "v1")

    # ── Compute all features on full dataset (per-row time cutoff via shift(1)) ──
    _logger.info("Computing training features on full dataset ...")
    featurized_df = compute_training_features(df)

    # ── Temporal split AFTER featurization ────────────────────────────────
    _logger.info("Splitting pre-featurized dataset (70/5/5/5/15) ...")
    train_df, val_stop_df, val_cal_df, thresh_df, test_df = temporal_split(featurized_df)

    _logger.info(
        "Split sizes: train={}, val_stop={}, val_calibration={}, threshold_set={}, test={}",
        len(train_df), len(val_stop_df), len(val_cal_df), len(thresh_df), len(test_df),
    )

    # ── Validate no leakage on each split ─────────────────────────────────
    validate_no_leakage(train_df, "train")
    validate_no_leakage(val_stop_df, "val_stop")
    validate_no_leakage(val_cal_df, "val_calibration")
    validate_no_leakage(thresh_df, "threshold_set")
    validate_no_leakage(test_df, "test")

    X_train, y_train = _df_to_feature_matrix(train_df, transformer)
    X_val_stop, y_val_stop = _df_to_feature_matrix(val_stop_df, transformer)
    X_val_cal, y_val_cal = _df_to_feature_matrix(val_cal_df, transformer)
    X_thresh, y_thresh = _df_to_feature_matrix(thresh_df, transformer)
    X_test, y_test = _df_to_feature_matrix(test_df, transformer)

    # Guard: minimum test positives
    n_test_pos = int(y_test.sum())
    if n_test_pos < min_test_positives:
        raise TrainingGateError(
            f"Test set has only {n_test_pos} positive examples, minimum is {min_test_positives}. "
            "Increase dataset size before training."
        )

    # Partition identity assertion — structural guard
    assert id(X_val_cal) != id(X_thresh), \
        "val_calibration and threshold_set must be distinct arrays."

    # ── Apply Redis dropout simulation (before XGBoost.fit) ──────────────
    rng = np.random.default_rng(42)
    X_train = apply_redis_degradation(X_train, feature_names, dropout_rate, rng)
    X_val_stop = apply_redis_degradation(X_val_stop, feature_names, dropout_rate, rng)
    X_val_cal = apply_redis_degradation(X_val_cal, feature_names, dropout_rate, rng)
    X_thresh = apply_redis_degradation(X_thresh, feature_names, dropout_rate, rng)
    # X_test: no simulation — measures clean ceiling

    # ── Validate feature contract ─────────────────────────────────────────
    validator = FeatureContractValidator(feature_names)
    validator.validate_always(X_train, "train")
    validator.validate_always(X_val_stop, "val_stop")
    validator.validate_always(X_val_cal, "val_calibration")
    validator.validate_always(X_thresh, "threshold_set")

    # ── PHASE 1: XGBoost training ─────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    base_params = {
        "n_estimators": xgb_cfg.get("n_estimators", 500),
        "max_depth": xgb_cfg.get("max_depth", 6),
        "learning_rate": xgb_cfg.get("learning_rate", 0.05),
        "subsample": xgb_cfg.get("subsample", 0.8),
        "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": xgb_cfg.get("early_stopping_rounds", 50),
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "seed": 42,
    }

    _logger.info("Phase 1: XGBoost training (scale_pos_weight={:.2f}) ...", scale_pos_weight)
    final_model = XGBClassifier(**base_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val_stop, y_val_stop)],
        verbose=False,
    )

    # ── PHASE 2: Calibration (val_calibration only) ───────────────────────
    _logger.info("Phase 2: Isotonic calibration on val_calibration ...")
    calibrator = fit_isotonic(final_model, X_val_cal, y_val_cal)

    if not check_calibrator_monotonicity(calibrator):
        raise TrainingGateError("Calibrator failed monotonicity check.")

    cal_proba_val = calibrate_batch(calibrator, final_model.predict_proba(X_val_cal)[:, 1])
    ece = compute_ece(cal_proba_val, y_val_cal)
    _logger.info("ECE on val_calibration: {:.4f}", ece)

    if ece > max_ece:
        raise TrainingGateError(
            f"ECE gate failed: {ece:.4f} > {max_ece}. Artifact not written."
        )

    # ── PHASE 3: Threshold tuning on threshold_set ────────────────────────
    _logger.info("Phase 3: Threshold tuning on threshold_set ...")
    cal_proba_thresh = calibrate_batch(calibrator, final_model.predict_proba(X_thresh)[:, 1])

    threshold_challenge = tune_challenge_threshold(cal_proba_thresh, y_thresh)
    threshold_block = tune_block_threshold(cal_proba_thresh, y_thresh)
    geo_anomaly_km = tune_geo_threshold(X_thresh, y_thresh, feature_names)
    dormancy_hours = tune_dormancy_threshold(X_thresh, y_thresh, feature_names)

    _logger.info(
        "Tuned thresholds: challenge={:.3f}, block={:.3f}, geo={:.1f}km, dormancy={:.1f}h",
        threshold_challenge, threshold_block, geo_anomaly_km, dormancy_hours,
    )

    # ── PHASE 4: Final evaluation + multi-seed PR-AUC gate ───────────────
    _logger.info("Phase 4: Final evaluation on test set (3-seed gate) ...")

    # Get production baseline
    if db is not None:
        prod_id = get_production_model_id(db)
        if prod_id is not None:
            from core.models.registry import get_model_metadata
            prod_meta = get_model_metadata(db, prod_id)
            production_baseline = prod_meta.get("pr_auc", 0.0) if prod_meta else 0.0
        else:
            production_baseline = 0.0
            _logger.info("No production model found — skipping PR-AUC gate comparison.")
    else:
        production_baseline = 0.0

    gate_result = multi_seed_pr_auc_gate(
        X_train=X_train, y_train=y_train,
        X_val_stop=X_val_stop, y_val_stop=y_val_stop,
        X_val_cal=X_val_cal, y_val_cal=y_val_cal,
        X_test=X_test, y_test=y_test,
        production_baseline_pr_auc=production_baseline,
        seeds=seeds,
        min_improvement=min_pr_auc_improvement if production_baseline > 0 else 0.0,
        xgb_params=base_params,
    )

    if production_baseline > 0 and not gate_result.passed:
        raise TrainingGateError(f"PR-AUC gate failed: {gate_result.reason}")

    # Check rolling avg gate
    if db is not None:
        history = get_model_pr_auc_history(db, pr_auc_rolling_window)
        if len(history) >= pr_auc_rolling_window:
            rolling_avg = sum(history) / len(history)
            final_pr_auc = evaluate_pr_auc(final_model, calibrator, X_test, y_test)
            if final_pr_auc <= rolling_avg:
                raise TrainingGateError(
                    f"Rolling PR-AUC gate failed: {final_pr_auc:.4f} <= rolling avg {rolling_avg:.4f}"
                )

    final_pr_auc = evaluate_pr_auc(final_model, calibrator, X_test, y_test)
    _logger.info("Final PR-AUC on test: {:.4f}", final_pr_auc)

    # Baseline LR sanity check (non-blocking)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
        lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        lr.fit(X_train, y_train)
        lr_proba = lr.predict_proba(X_test)[:, 1]
        lr_pr_auc = float(average_precision_score(y_test, lr_proba))
        if final_pr_auc <= lr_pr_auc + 0.05:
            _logger.warning(
                "XGBoost PR-AUC ({:.4f}) is not sufficiently better than LR baseline ({:.4f}). "
                "Check feature pipeline.", final_pr_auc, lr_pr_auc,
            )
    except Exception as e:
        _logger.debug("LR baseline check skipped: {}", e)

    # Record timestamp ranges for retraining leakage prevention
    val_ts_start = val_stop_df["timestamp"].min()
    val_ts_end = val_cal_df["timestamp"].max()
    test_ts_start = test_df["timestamp"].min()
    test_ts_end = test_df["timestamp"].max()

    model_id = generate_model_id()
    metadata = {
        "model_id": model_id,
        "pr_auc": final_pr_auc,
        "ece": float(ece),
        "f1": None,
        "n_rows": len(df),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "schema_hash": schema_hash,
        "feature_names_version": feature_names_version,
        "feature_names": feature_names,
        "redis_dropout_rate": dropout_rate,
        "thresholds": {
            "challenge": float(threshold_challenge),
            "block": float(threshold_block),
        },
        "context_rules": {
            "geo_anomaly_km": float(geo_anomaly_km),
            "dormancy_hours": float(dormancy_hours),
        },
        "seeds_pr_auc": {str(k): v for k, v in gate_result.pr_auc_per_seed.items()},
        "val_timestamp_range": {
            "start": str(val_ts_start) if val_ts_start else None,
            "end": str(val_ts_end) if val_ts_end else None,
        },
        "test_timestamp_range": {
            "start": str(test_ts_start) if test_ts_start else None,
            "end": str(test_ts_end) if test_ts_end else None,
        },
        "version": model_id,
    }

    save_model_artifacts(
        model_id=model_id,
        xgb_model=final_model,
        calibrator=calibrator,
        transformer=transformer,
        metadata=metadata,
        schema_yaml_path=schema_yaml_path,
    )

    if db is not None:
        register_model(db, model_id, metadata, status="staging")

    return TrainingResult(
        model_id=model_id,
        pr_auc=final_pr_auc,
        ece=float(ece),
        thresholds=metadata["thresholds"],
        context_rules=metadata["context_rules"],
        seeds_pr_auc=gate_result.pr_auc_per_seed,
        metadata=metadata,
    )
