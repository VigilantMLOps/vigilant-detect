"""Model evaluation — PR-AUC gate, ECE gate, threshold tuning, multi-seed gate."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier


@dataclass
class GateResult:
    passed: bool
    pr_auc_per_seed: dict[int, float]
    baseline: float
    min_improvement: float
    rolling_avg: float | None = None
    reason: str = ""


def evaluate_pr_auc(
    model: XGBClassifier,
    calibrator: IsotonicRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    raw = model.predict_proba(X)[:, 1]
    cal = calibrator.predict(raw)
    return float(average_precision_score(y, cal))


def compute_ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — measures probability reliability."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        bin_prob = proba[mask].mean()
        bin_acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(bin_prob - bin_acc)
    return float(ece)


def tune_challenge_threshold(cal_proba: np.ndarray, y: np.ndarray) -> float:
    """
    Find threshold_challenge that maximises F1(ALLOW vs CHALLENGE+BLOCK).
    Scans candidate thresholds from PR curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y, cal_proba)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.35


def tune_block_threshold(cal_proba: np.ndarray, y: np.ndarray, target_recall: float = 0.80) -> float:
    """
    Find threshold_block that maximises Precision at >= target_recall for BLOCK.
    """
    precisions, recalls, thresholds = precision_recall_curve(y, cal_proba)
    valid = recalls[:-1] >= target_recall
    if not valid.any():
        return 0.75
    best_idx = int(np.argmax(precisions[:-1][valid]))
    valid_thresholds = thresholds[valid]
    return float(valid_thresholds[best_idx]) if len(valid_thresholds) > 0 else 0.75


def tune_geo_threshold(
    X_threshold: np.ndarray,
    y_threshold: np.ndarray,
    feature_names: list[str],
    candidates: list[float] | None = None,
) -> float:
    """
    Tune geo_anomaly_km on threshold_set by scanning candidates and maximising
    F1 on rule-escalation decisions (where geo is the distinguishing signal).
    Falls back to 500.0 if insufficient positive examples.
    """
    if candidates is None:
        candidates = [100.0, 200.0, 300.0, 500.0, 750.0, 1000.0]

    try:
        geo_idx = feature_names.index("geo_distance_delta")
    except ValueError:
        return 500.0

    geo_vals = X_threshold[:, geo_idx]
    best_threshold = 500.0
    best_f1 = -1.0

    for cand in candidates:
        rule_triggered = (geo_vals > cand).astype(int)
        if rule_triggered.sum() == 0:
            continue
        tp = int(((rule_triggered == 1) & (y_threshold == 1)).sum())
        fp = int(((rule_triggered == 1) & (y_threshold == 0)).sum())
        fn = int(((rule_triggered == 0) & (y_threshold == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = cand

    return best_threshold


def tune_dormancy_threshold(
    X_threshold: np.ndarray,
    y_threshold: np.ndarray,
    feature_names: list[str],
    candidates: list[float] | None = None,
) -> float:
    """Tune dormancy_hours on threshold_set. Falls back to 720.0."""
    if candidates is None:
        candidates = [168.0, 336.0, 480.0, 720.0, 1440.0, 2160.0]

    try:
        gap_idx = feature_names.index("last_login_gap_h")
    except ValueError:
        return 720.0

    gap_vals = X_threshold[:, gap_idx]
    best_threshold = 720.0
    best_f1 = -1.0

    for cand in candidates:
        rule_triggered = (gap_vals > cand).astype(int)
        if rule_triggered.sum() == 0:
            continue
        tp = int(((rule_triggered == 1) & (y_threshold == 1)).sum())
        fp = int(((rule_triggered == 1) & (y_threshold == 0)).sum())
        fn = int(((rule_triggered == 0) & (y_threshold == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = cand

    return best_threshold


def multi_seed_pr_auc_gate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_stop: np.ndarray,
    y_val_stop: np.ndarray,
    X_val_cal: np.ndarray,
    y_val_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    production_baseline_pr_auc: float,
    seeds: list[int],
    min_improvement: float,
    xgb_params: dict,
) -> GateResult:
    """
    All seeds must individually exceed production baseline + min_improvement.
    Any seed failing = gate fails. Fixed seeds = reproducible failures.
    """
    from core.models.calibrator import fit_isotonic

    pr_auc_results: dict[int, float] = {}

    for seed in seeds:
        params = dict(xgb_params)
        params["seed"] = seed
        params["random_state"] = seed

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val_stop, y_val_stop)],
            verbose=False,
        )
        calibrator = fit_isotonic(model, X_val_cal, y_val_cal)
        pr_auc = evaluate_pr_auc(model, calibrator, X_test, y_test)
        pr_auc_results[seed] = pr_auc

    all_pass = all(
        v > production_baseline_pr_auc + min_improvement
        for v in pr_auc_results.values()
    )
    failing = [s for s, v in pr_auc_results.items() if v <= production_baseline_pr_auc + min_improvement]
    reason = "" if all_pass else (
        f"Seeds {failing} did not clear baseline {production_baseline_pr_auc:.4f} + {min_improvement}: "
        + ", ".join(f"seed {s}={pr_auc_results[s]:.4f}" for s in failing)
    )
    return GateResult(
        passed=all_pass,
        pr_auc_per_seed=pr_auc_results,
        baseline=production_baseline_pr_auc,
        min_improvement=min_improvement,
        reason=reason,
    )
