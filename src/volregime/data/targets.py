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
        # tail if max absolute daily return > N* rolling std
        max_abs_ret = np.max(np.abs(future_log_returns))
        if historical_rvs is not None and len(historical_rvs) > 0:
            rolling_std  = np.std(historical_rvs)
            return int(max_abs_ret > threshold_value * rolling_std)
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

    # volatility state: ATR ratio
    atr_short_period = regime_config.get('atr_short_period', 10)
    atr_long_period = regime_config.get('atr_long_period',50)
    atr_ratio_thresholds = regime_config.get('atr_ratio_thresholds', {"quiet": 0.75, "volatile": 1.25})
    atr_short = compute_atr(highs, lows, closes, atr_short_period)
    atr_long = compute_atr(highs, lows, closes, atr_long_period)    

    if atr_long > 0:
        atr_ratio = atr_short / atr_long 
    else:
        atr_ratio = 1.0

    if atr_ratio < atr_ratio_thresholds.get("quiet", 0.75):
        volatility = 'quiet'
    elif atr_ratio > atr_ratio_thresholds.get('volatile',1.25):
        volatility = 'volatile'
    else:
        volatility = 'quiet' # normal -> default to quiet
    
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
    if n < period + 1:
        return 0.0
    
    # true range
    tr = np.max(highs[1:] - lows[1:], np.max(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))

    # directional movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move>0), up_move, 0.0)
    minus_dm = np.where((up_move < down_move) & (down_move>0), down_move, 0.0)

    # smoothed averages (Wilder's smoothing)
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / np.where(atr >0, atr, 1)
    minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr >0, atr, 1)

    # dx
    di_sum = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / np.where(di_sum >0 , di_sum, 1)

    # adx 
    adx_values = wilder_smooth(dx, period)

    return float(adx_values[-1]) if len(adx_values) > 0 else 0.0

def compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range. Returns the latest ATR value."""
    n = len(highs)
    if n < 2:
        return 0.0
    
    tr = np.max(highs[1:] - lows[1:], np.max(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))

    if len(tr) < period:
        return float(np.mean(tr))
    
    atr_values = wilder_smooth(tr, period)
    return float(atr_values[-1]) if len(atr_values) >0 else 0.0

def wilder_smooth(values, period):
    """Wilder's exponential smoothing (used in ATR, ADX)."""
    result = np.zeros_like(values)
    result[:period] = np.nan
    result[period-1] = np.mean(values[:period]) # seed with SMA
    
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period

    return result
