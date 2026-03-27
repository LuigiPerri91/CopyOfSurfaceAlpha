"""
Statistical tests for forecast comparison.

diebold_mariano:
    Tests H0 = equal predictive accuracy between model A and model B.
    A negative DM statistic with p < 0.05 means A significantly beats B.
    Uses Newey-West HAC variance to account for autocorrelated loss differentials
    (important when forecast horizon h > 1 day, as here with h=21).

mincer_zarnowitz:
    OLS regression of true_rv on (1, predicted_rv).
    Efficient forecast: intercept ~ 0, slope ~ 1, R² as high as possible.
    Returns slope, intercept, R², and F-stat for H0: (intercept=0, slope=1).
"""

from volregime.evaluation.forecast_metrics import qlike
import numpy as np
import statsmodels.api as sm


def diebold_mariano(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 21,
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Args:
        losses_a: per-sample loss values for model A (e.g. QLIKE per date)
        losses_b: per-sample loss values for model B
        h:        forecast horizon in periods (21 for 21-day ahead RV).
                  Used to set Newey-West lag order to correct for MA(h-1)
                  autocorrelation in the loss differential series.

    Returns:
        dm_stat:  test statistic (negative → A better than B)
        p_value:  two-sided p-value (reject H0 of equal accuracy if p < 0.05)

    Interpretation:
        dm_stat < 0, p < 0.05  → model A is significantly better than B
        dm_stat > 0, p < 0.05  → model B is significantly better than A

    Implementation note:
        We fit a constants-only OLS on the loss differential d = losses_a - losses_b,
        then extract the Newey-West HAC standard error for the intercept (which equals
        the mean of d). This is the standard approach and gives us the correct
        t-statistic with proper autocorrelation-robust inference.
    """
    a = np.asarray(losses_a, dtype=np.float64)
    b = np.asarray(losses_b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Loss arrays must have the same length: {len(a)} vs {len(b)}")
    if len(a) < h + 1:
        raise ValueError(f"Need at least h+1={h+1} observations, got {len(a)}")

    d = a - b  # loss differential series

    # OLS of d on a constant -> intercept = mean(d), residuals = d - mean(d)
    # Using HAC (Newey-West) covariance with nlags = h-1, as per the original paper.
    # statsmodels uses the Bartlett kernel by default.
    ones = np.ones((len(d), 1))
    model = sm.OLS(d, ones)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags': h - 1, 'use_correction': True})

    dm_stat = float(res.tvalues[0])
    p_value = float(res.pvalues[0])   # two-sided by default in statsmodels

    return dm_stat, p_value


def mincer_zarnowitz(
    rv_pred: np.ndarray,
    rv_true: np.ndarray,
) -> dict[str, float]:
    """
    Mincer-Zarnowitz (1969) regression for forecast efficiency.

    Runs OLS:  rv_true = α + β * rv_pred + ε

    An efficient, unbiased forecast satisfies: α = 0 and β = 1.
    The F-statistic tests this joint restriction.

    Args:
        rv_pred: model predictions (can be in log or raw space, but must be consistent)
        rv_true: realised values (same space as rv_pred)

    Returns:
        dict with:
            intercept:  α
            slope:      β  (should be ~1 for an efficient forecast)
            r2:         R² of the regression
            f_stat:     F-statistic for joint H0: α=0, β=1
            f_pvalue:   p-value of the F-test (small → forecast is biased/inefficient)
            n:          number of observations
    """
    p = np.asarray(rv_pred, dtype=np.float64)
    t = np.asarray(rv_true, dtype=np.float64)

    if len(p) != len(t):
        raise ValueError(f"Arrays must have the same length: {len(p)} vs {len(t)}")

    X = sm.add_constant(p)   # shape (N, 2): [1, rv_pred]
    res = sm.OLS(t, X).fit()

    intercept = float(res.params[0])
    slope = float(res.params[1])
    r2 = float(res.rsquared)

    # F-test for joint restriction: intercept=0, slope=1
    # R_matrix selects [intercept, slope], q_vector is [0, 1]
    hypotheses = '(const = 0), (x1 = 1)'
    f_test = res.f_test(hypotheses)
    f_stat = float(f_test.fvalue.item() if hasattr(f_test.fvalue, 'item') else f_test.fvalue)
    f_pvalue = float(f_test.pvalue.item() if hasattr(f_test.pvalue, 'item') else f_test.pvalue)

    return {
        'intercept': round(intercept, 6),
        'slope': round(slope, 6),
        'r2': round(r2, 6),
        'f_stat': round(f_stat, 4),
        'f_pvalue': round(f_pvalue, 6),
        'n': int(len(p)),
    }

def compare_all_models(
    predictions: dict[str, np.ndarray],
    log_rv_true: np.ndarray,
    h: int = 21
) -> dict[str, dict]:
    """
    Run DM tests between every model and the 'surface_alpha' model.

    Args:
        predictions:  {model_name: log_rv_pred array}
        log_rv_true:  true log(RV) values

    Returns:
        {baseline_name: {"dm_stat": float, "p_value": float, "significant": bool}}
    """
    if 'surface_alpha' not in predictions:
        raise ValueError("predictions must contain a 'surface_alpha' key.")

    sa_pred = predictions['surface_alpha']
    sa_losses = np.array([
        qlike(np.array([sa_pred[i]]), np.array([log_rv_true[i]])) for i in range(len(sa_pred))
    ])

    results = {}
    for name, pred in predictions.items():
        if name == 'surface_alpha':
            continue
        b_losses = np.array([
            qlike(np.array([pred[i]]), np.array([log_rv_true[i]])) for i in range(len(pred))
        ])
        dm_stat, p_val = diebold_mariano(sa_losses, b_losses, h=h)
        results[name] = {
            'dm_stat': round(dm_stat,4),
            'p_value': round(p_val, 6),
            'significant': p_val < 0.05,
            'sa_better': dm_stat < 0 and p_val < 0.05
        }

    return results