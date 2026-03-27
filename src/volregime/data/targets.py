import pandas as pd
import numpy as np 
import logging

logger = logging.getLogger(__name__)

def compute_forward_rv(future_log_returns, horizon):
    """
    Forward realized volatility: RV_{t→t+h} = sqrt(sum_{i=1}^{h} r_{t+i}^2)

    Args:
        future_log_returns: array of log returns for dates [t+1 .. t+h]
        horizon: h (must equal len(future_log_returns))

    Returns:
        rv: float (the realized vol)
        log_rv: float (log of realized vol — the actual training target)
    """
    assert len(future_log_returns) == horizon, f"Expected {horizon} future returns, got {len(future_log_returns)}"

    rv = np.sqrt(np.sum(future_log_returns ** 2))

    # guard against log(0)
    log_rv = np.log(max(rv, 1e-10))

    return rv, log_rv

def compute_tail_indicator(future_log_returns, rv, method,  threshold_value, historical_rvs = None):
    """
    Binary tail-risk indicator.

    Args:
        future_log_returns: array of forward returns
        rv: forward realized vol (already computed)
        method: "percentile" or "sigma"
        threshold_value: 90 (for percentile) or N (for sigma multiplier)
        historical_rvs: array of all past RV values (needed for percentile method)

    Returns:
        tail_indicator: 0 or 1
    """
    if method == 'percentile':
        if historical_rvs is None or len(historical_rvs) == 0:
            return 0
        threshold = np.percentile(historical_rvs, threshold_value)
        return int(rv > threshold)
    elif method == 'sigma':
        # tail if max absolute daily return > N * historical return std
        # Note: historical_rvs here is past RV values (not returns), so we
        # use the forward returns themselves as the baseline for comparison.
        # A better sigma baseline would be trailing return std, but since we
        # only have historical_rvs available here, use rv as the scale proxy.
        max_abs_ret = np.max(np.abs(future_log_returns))
        if historical_rvs is not None and len(historical_rvs) > 0:
            # Approximate: typical daily move ≈ trailing RV / sqrt(h)
            h = len(future_log_returns)
            typical_daily_rv = float(np.median(historical_rvs)) / np.sqrt(max(h, 1))
            return int(max_abs_ret > threshold_value * typical_daily_rv)
        return 0
    else:
        raise ValueError(f"Unknown tail method: {method}")

def compute_regime_label(underlying_window, market_state_row, regime_config):
    """
    Rule-based regime classification from the PDR.
    Uses price vs 200 MA + ADX + ATR ratio.

    Args:
        underlying_window: DataFrame of underlying OHLCV for trailing ~200+ days
            must have columns: [adj_close, high, low, close]
        market_state_row: dict/Series with at least 'vix' value
        regime_config: the backtest.yaml regime_identification section

    Returns:
        regime_label: integer 0-5
            0 = bull_quiet
            1 = bull_volatile
            2 = bear_quiet
            3 = bear_volatile
            4 = sideways_quiet
            5 = sideways_volatile
    """
    prices = underlying_window['adj_close'].values
    highs = underlying_window['high'].values
    lows = underlying_window['low'].values
    closes = underlying_window['close'].values

    # trend direction: price vs 200 MA
    ma_period = regime_config.get('ma_period',200)
    if len(prices) < ma_period:
        direction = 'sideways'
    else:
        ma_200 = np.mean(prices[-ma_period:])
        current_price = prices[-1]
        pct_from_ma = (current_price - ma_200) / ma_200

        if pct_from_ma > 0.02:
            direction = 'bull'
        elif pct_from_ma < -0.02:
            direction = 'bear'
        else:
            direction = 'sideways'

    # trend strength : ADX
    adx_period = regime_config.get('adx_period',14)
    adx = compute_adx(highs, lows, closes, adx_period)
    adx_thresholds = regime_config.get('adx_thresholds', {'no_trend':20})

    # override direction to sideways of ADX says no trend
    if adx < adx_thresholds.get("no_trend",20):
        direction = 'sideways'

    # volatility state: VIX hard override → ATR ratio tiebreaker
    vix_thresholds = regime_config.get('vix_thresholds', {'low': 15, 'elevated': 25})
    vix = float(market_state_row.get('vix', float('nan')) if hasattr(market_state_row, 'get') else market_state_row['vix'])

    if not np.isnan(vix) and vix >= vix_thresholds.get('elevated', 25):
        # VIX at or above elevated threshold → hard override to volatile
        volatility = 'volatile'
    elif not np.isnan(vix) and vix <= vix_thresholds.get('low', 15):
        # VIX at or below low threshold → hard override to quiet
        volatility = 'quiet'
    else:
        # VIX in normal range → fall through to ATR ratio
        atr_short_period = regime_config.get('atr_short_period', 10)
        atr_long_period = regime_config.get('atr_long_period', 50)
        atr_ratio_thresholds = regime_config.get('atr_ratio_thresholds', {'quiet': 0.75, 'volatile': 1.25})
        atr_short = compute_atr(highs, lows, closes, atr_short_period)
        atr_long = compute_atr(highs, lows, closes, atr_long_period)

        atr_ratio = atr_short / atr_long if atr_long > 0 else 1.0

        if atr_ratio < atr_ratio_thresholds.get('quiet', 0.75):
            volatility = 'quiet'
        elif atr_ratio > atr_ratio_thresholds.get('volatile', 1.25):
            volatility = 'volatile'
        else:
            volatility = 'quiet'  # normal range → default to quiet
    
    regime_map = {
        ("bull", "quiet"): 0,
        ("bull", "volatile"): 1,
        ("bear", "quiet"): 2,
        ("bear", "volatile"): 3,
        ("sideways", "quiet"): 4,
        ("sideways", "volatile"): 5,
    }

    return regime_map[(direction, volatility)]

def compute_adx(highs, lows, closes, period=14):
    """Compute Average Directional Index. Returns the latest ADX value."""
    n = len(highs)
    # need at least 2*period bars: one for DI smoothing, one for ADX smoothing
    if n < 2 * period + 1:
        return 0.0

    # true range
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))

    # directional movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((up_move < down_move) & (down_move > 0), down_move, 0.0)

    # first Wilder pass: smooth TR, +DM, -DM
    atr_s = wilder_smooth(tr, period)
    plus_dm_s = wilder_smooth(plus_dm, period)
    minus_dm_s = wilder_smooth(minus_dm, period)

    # +DI / -DI (valid from index period-1 onward; NaN before)
    plus_di = 100.0 * plus_dm_s / np.where(atr_s > 0, atr_s, 1.0)
    minus_di = 100.0 * minus_dm_s / np.where(atr_s > 0, atr_s, 1.0)

    # DX — still has NaN in warm-up positions 0..period-2
    di_sum = plus_di + minus_di
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where(di_sum > 0, di_sum, 1.0)

    # Strip the NaN warm-up prefix before the second Wilder pass (ADX).
    # Without this, wilder_smooth seeds with np.mean([NaN, ...]) = NaN
    # and the entire ADX result is NaN, so the regime override never fires.
    dx_valid = dx[period - 1:]  # first valid DX is at index period-1
    if len(dx_valid) < period:
        return 0.0

    # second Wilder pass: smooth DX -> ADX
    adx_values = wilder_smooth(dx_valid, period)
    valid = adx_values[~np.isnan(adx_values)]
    return float(valid[-1]) if len(valid) > 0 else 0.0

def compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range. Returns the latest ATR value."""
    n = len(highs)
    if n < 2:
        return 0.0
    
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))

    if len(tr) < period:
        return float(np.mean(tr))
    
    atr_values = wilder_smooth(tr, period)
    valid = atr_values[~np.isnan(atr_values)]
    return float(valid[-1]) if len(valid) > 0 else 0.0

def wilder_smooth(values, period):
    """Wilder's exponential smoothing (used in ATR, ADX)."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)

    result = np.zeros(len(values), dtype=np.float64)
    # seed: first valid value is the SMA of the first `period` bars
    result[period - 1] = np.mean(values[:period])

    for i in range(period, len(values)):
        result[i] = (result[i - 1] * (period - 1) + values[i]) / period

    # mark the warm-up prefix as NaN so callers know these are invalid
    result[:period - 1] = np.nan

    return result
