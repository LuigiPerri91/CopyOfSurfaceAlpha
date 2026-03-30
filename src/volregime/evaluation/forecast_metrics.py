"""
Statistical forecast accuracy metrics for volatility and regime predictions.

Volatility metrics (operate on log(RV) space):
    qlike       — proper scoring rule; gold standard for vol forecast comparison
    ql_loss     — MSE in log space (symmetric, scale-free)
    mape        — mean absolute % error on raw RV
    r2_score    — fraction of RV variance explained
    bias        — mean signed error (positive = model over-predicts)
    rmse        — root mean squared error on raw RV

Classification metrics:
    brier_score       — proper scoring rule for probabilities
    tail_auc_roc      — area under ROC for tail-risk classification
    regime_accuracy   — fraction of dates where regime_pred == regime_true
    regime_calibration— mean |mean(p_k) - freq(true==k)| across regimes
"""

import numpy as np 
from sklearn.metrics import roc_auc_score

# Volatility metrics

def qlike(log_rv_pred: np.ndarray, log_rv_true:np.ndarray) -> float:
    """
    Quasi-Likelihood (QLIKE) loss — proper scoring rule for variance forecasts.

    QLIKE = E[ log(h) + RV/h ]   where h = predicted variance, RV = true variance.

    In log-RV space (model predicts log(σ), truth is log(σ_true)):
        h   = exp(log_rv_pred)^2  (predicted variance)
        RV  = exp(log_rv_true)^2  (true realized variance)

        QLIKE = 2*log_rv_pred + exp(2*(log_rv_true - log_rv_pred))

    Lower is better.
    """
    p = np.asarray(log_rv_pred, dtype= np.float64)
    t = np.asarray(log_rv_true, dtype = np.float64)
    return float(np.mean(2.0 * p + np.exp(2.0 * (t-p))))

def ql_loss(log_rv_pred: np.ndarray, log_rv_true:np.ndarray) -> float:
    """MSE in log space (QL / log-MSE). Symmetric and scale-free"""
    d = np.asarray(log_rv_pred) - np.asarray(log_rv_true)
    return float(np.mean(d**2))

def mape(rv_pred: np.ndarray, rv_true: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error on raw (non-log) realized vol"""
    p = np.asarray(rv_pred, dtype=np.float64)
    t = np.asarray(rv_true, dtype=np.float64)
    return float(np.mean(np.abs(p-t) / (np.abs(t) + eps)))

def r2_score(rv_pred: np.ndarray, rv_true: np.ndarray) -> float:
    """Coefficient of determination on raw realized vol."""
    p = np.asarray(rv_pred, dtype=np.float64)
    t = np.asarray(rv_true, dtype=np.float64)
    ss_res = np.sum((t-p)**2)
    ss_tot = np.sum((t-t.mean())**2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))

def bias(rv_pred: np.ndarray, rv_true: np.ndarray) -> float:
    """Mean signed error. Positive -> model over-predicts vol."""
    return float(np.mean(np.asarray(rv_pred) - np.asarray(rv_true)))

def rmse(rv_pred: np.ndarray, rv_true: np.ndarray) -> float:
    """Root mean squared error on raw realized vol."""
    d = np.asarray(rv_pred) - np.asarray(rv_true)
    return float(np.sqrt(np.mean(d ** 2)))

def hit_rate(rv_pred: np.ndarray, rv_true: np.ndarray, rv_prev: np.ndarray) -> float:
    """
    Fraction of dates where model correctly predicts direction of RV change.
    Requires prev-period RV as the baseline level.
    """
    pred_up = np.asarray(rv_pred) > np.asarray(rv_prev)
    true_up = np.asarray(rv_true) > np.asarray(rv_prev)
    return float(np.mean(pred_up == true_up))

def compute_vol_metrics(log_rv_pred: np.ndarray, log_rv_true: np.ndarray, rv_prev: np.ndarray | None = None) -> dict[str,float]:
    """
    Compute all volatility forecast metrics.

    Args:
        log_rv_pred: model predictions in log(RV) space, shape (N,)
        log_rv_true: true targets   in log(RV) space, shape (N,)
        rv_prev:     prior-period raw RV for hit-rate (optional)

    Returns:
        dict of metric_name → scalar value
    """
    rv_pred_raw = np.exp(log_rv_pred)
    rv_true_raw = np.exp(log_rv_true)

    metrics = {
        "qlike": qlike(log_rv_pred, log_rv_true),
        "ql":    ql_loss(log_rv_pred, log_rv_true),
        "mape":  mape(rv_pred_raw, rv_true_raw),
        "r2":    r2_score(rv_pred_raw, rv_true_raw),
        "bias":  bias(rv_pred_raw, rv_true_raw),
        "rmse":  rmse(rv_pred_raw, rv_true_raw),
        "n":     int(len(log_rv_pred)),
    }
    if rv_prev is not None:
        metrics['hit_rate'] = hit_rate(rv_pred_raw, rv_true_raw, rv_prev)
    return metrics


# classification metrics
def brier_score(tail_prob: np.ndarray, tail_true: np.ndarray) -> float:
    """Proper scoring rule for binary probabilities. Lower is better."""
    p = np.asarray(tail_prob, dtype=np.float64)
    t = np.asarray(tail_true, dtype=np.float64)
    return float(np.mean((p-t) ** 2))

def regime_accuracy(regime_pred: np.ndarray, regime_true: np.ndarray) -> float:
    """Fraction of dates where predicted regime matches ground truth."""
    return float(np.mean(np.asarray(regime_pred) == np.asarray(regime_true)))

def regime_calibration(regime_probs: np.ndarray, regime_true: np.ndarray, num_regimes: int = 6) -> float:
    """
    Mean absolute calibration error across all regimes.
    For each regime k: |mean_predicted_probability - true_frequency|
    """
    probs = np.asarray(regime_probs)
    labels = np.asarray(regime_true)
    errors = []
    for k in range(num_regimes):
        errors.append(abs(float(probs[:, k].mean()) - float((labels == k).mean())))
    return float(np.mean(errors))

def compute_classification_metrics(
    tail_prob: np.ndarray,
    tail_true: np.ndarray,
    regime_probs: np.ndarray,
    regime_true: np.ndarray,
    num_regimes: int = 6
) -> dict[str, float]:
    """Compute all classification metrics."""
    regime_pred = regime_probs.argmax(axis=1) if regime_probs.ndim == 2 else regime_probs

    metrics = {
        'brier_score': brier_score(tail_prob, tail_true),
        'regime_accuracy': regime_accuracy(regime_pred, regime_true),
        'regime_calibration': regime_calibration(regime_probs, regime_true, num_regimes)
    }

    metrics['tail_auc'] = float(roc_auc_score(tail_true, tail_prob))

    # per-regime accuracy
    regime_names = ["bull_quiet", "bull_volatile", "bear_quiet","bear_volatile", "sideways_quiet", "sideways_volatile"]
    for k, name in enumerate(regime_names[:num_regimes]):
        mask = np.asarray(regime_true) == k
        if mask.sum() >= 3:
            metrics[f'regime_acc_{name}'] = float(
                (np.asarray(regime_pred)[mask] == k).mean()
            )
            metrics[f"n_{name}"] = int(mask.sum())

    return metrics

def compute_per_regime_vol_metrics(
    log_rv_pred: np.ndarray,
    log_rv_true: np.ndarray,
    regime_true: np.ndarray,
    num_regimes: int = 6
) -> dict[str, dict]:
    """
    Compute vol metrics broken down by the true regime label.
    Useful for understanding where the model succeeds / fails.
    """
    regime_names = ["bull_quiet", "bull_volatile", "bear_quiet","bear_volatile", "sideways_quiet", "sideways_volatile"]
    results = {}
    for k in range(num_regimes):
        mask = np.asarray(regime_true) == k
        name = regime_names[k] if k < len(regime_names) else str(k)
        if mask.sum() < 5:
            results[name] = {'n': int(mask.sum()), 'qlike': float('nan')}
            continue
        results[name] = compute_vol_metrics(log_rv_pred[mask], log_rv_true[mask])
        
    return results