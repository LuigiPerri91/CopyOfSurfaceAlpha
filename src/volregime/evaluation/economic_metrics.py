"""
Financial / economic performance metrics.

All metrics operate on the backtest equity curve (a time series of portfolio values
or daily returns). These complement the statistical forecast metrics — a model can
have good QLIKE but poor Sharpe if the overlay fails to translate forecasts into
good positions.

Primary function:
    compute_economic_metrics(returns_series, cfg) -> dict

Benchmark strategies (for comparison in backtest):
    buy_and_hold:        w = 1.0 always
    inverse_vol:         w = sigma_target / trailing_rv  (no model, raw RV)
    constant_vol_target: w = sigma_target / rolling_std (simple vol-targeting)
"""

import numpy as np
import pandas as pd

# Individual metrics

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, ann_factor: int = 252) -> float:
    """Annualized Sharpe ratio. Risk-free rate should be annualized."""
    r = np.asarray(returns, dtype=np.float64)
    exc = r - risk_free_rate / ann_factor
    std = exc.std()
    return float(exc.mean()/ std * np.sqrt(ann_factor)) if std >0 else 0.0

def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, ann_factor: int =252) -> float:
    """Annualized sortino ratio (downside deviation only)"""
    r = np.asarray(returns, dtype=np.float64)
    exc = r - risk_free_rate / ann_factor
    neg = exc[exc < 0]
    downside = neg.std() * np.sqrt(ann_factor) if len(neg) > 1 else 1e-10
    return float(exc.mean() * ann_factor / downside)

def max_drawdown(equity: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown (as a negative fraction)
    e.g. -0.25 means a 25% drawdown at worst
    """
    eq = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq-peak) / peak
    return float(dd.min())

def calmar_ratio(returns: np.ndarray, equity: np.ndarray, ann_factor: int = 252) -> float:
    """Annualized return divided by absolute max drawdown"""
    ann_ret = float(np.asarray(returns).mean() * ann_factor)
    # use a small floor for MDD to avoid division by zero while maintaining sign
    mdd = max(abs(max_drawdown(equity)), 1e-6)
    return ann_ret / mdd

def turnover(weights: np.ndarray, ann_factor: int =252) -> float:
    """
    Annualized turnover = mean |Δw| * ann_factor.
    Measures how much the position changes on average per year.
    """
    dw = np.abs(np.diff(np.asarray(weights, dtype=np.float64)))
    return float(dw.mean() * ann_factor)

def vol_target_tracking(returns: np.ndarray, sigma_target: float, ann_factor: int = 252) -> float:
    """
    Ratio of realised annualised portfolio vol to sigma_target.
    Values near 1.0 mean the strategy is running at the intended risk level.
    Values < 1 mean under-invested; > 1 means over-invested.
    """
    r = np.asarray(returns, dtype=np.float64)
    realised_vol = float(r.std() * np.sqrt(ann_factor))
    return round(realised_vol / sigma_target, 4)

# combined metrics

def compute_economic_metrics(
    returns: np.ndarray | pd.Series,
    equity: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
    sigma_hat: np.ndarray | pd.Series | None = None,
    sigma_target: float = 0.10,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
) -> dict[str, float]:
    """
    Compute all financial performance metrics.

    Args:
        returns:        daily strategy net returns (after costs)
        equity:         cumulative portfolio value (starts at 1.0)
        weights:        daily position sizes (for turnover, optional)
        sigma_hat:      model's predicted annualised vol (for tracking, optional)
        sigma_target:   annualised vol target from backtest config
        risk_free_rate: annualised risk-free rate
        ann_factor:     trading days per year

    Returns:
        dict of metric_name -> scalar value
    """
    r = np.asarray(returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)

    ann_ret = float(r.mean() * ann_factor)
    ann_vol = float(r.std() * np.sqrt(ann_factor))
    mdd = max_drawdown(eq)

    metrics = {
        'ann_return_pct': round(ann_ret * 100, 3),
        'ann_vol_pct': round(ann_vol * 100, 3),
        'sharpe': round(sharpe_ratio(r, risk_free_rate, ann_factor),4),
        'sortino': round(sortino_ratio(r, risk_free_rate, ann_factor),4),
        'max_drawdown_pct': round(mdd * 100, 3),
        'calmar': round(calmar_ratio(r,eq, ann_factor),4),
        'total_return_pct': round(float(eq[-1] - 1.0) * 100, 3),
        'n_trading_days': int(len(r)),
        'positive_days_pct': round(float((r>0).mean()) * 100, 2)
    }

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        metrics['turnover_ann'] = round(turnover(w, ann_factor), 3)
        metrics['avg_weight'] = round(float(w.mean()), 4)
        metrics['weight_std'] = round(float(w.std()), 4)

    metrics['vol_target_tracking'] = round(
        vol_target_tracking(r, sigma_target, ann_factor), 4
    )
    
    return metrics

def compute_benchmark_metrics(
    underlying_returns: np.ndarray,
    sigma_target: float = 0.10,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
) -> dict[str, dict]:
    """
    Compute metrics for benchmark strategies.

    Args:
        underlying_returns: daily log returns of the underlying (e.g. SPY)

    Returns:
        dict of benchmark_name -> metrics_dict
    """
    r = np.asarray(underlying_returns, dtype=np.float64)

    results = {}

    # buy and hold
    bh_returns = r
    bh_equity  = np.cumprod(1 + bh_returns)
    results['buy_and_hold'] = compute_economic_metrics(
        bh_returns, bh_equity, risk_free_rate=risk_free_rate,ann_factor=ann_factor
    )

    # inverse vol (trailing RV vol-targeting, no model)
    window = 21
    trail_rv = pd.Series(r).rolling(window).std().bfill().values
    trail_rv_ann = trail_rv * np.sqrt(ann_factor)
    iv_weights = np.clip(sigma_target / np.maximum(trail_rv_ann, 1e-4), 0.0, 1.5)
    iv_returns = np.roll(iv_weights, 1) * r
    iv_returns[0] = 0.0
    iv_equity = np.cumprod(1 + iv_returns)
    results['inverse_vol'] = compute_economic_metrics(
        iv_returns, iv_equity, weights=iv_weights, risk_free_rate=risk_free_rate, ann_factor=ann_factor
    )

    return results 