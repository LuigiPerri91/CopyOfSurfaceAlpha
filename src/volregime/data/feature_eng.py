import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
import yaml

load_dotenv()

with open('../configs/default.yaml','r') as f:
    default = yaml.safe_load(f)

with open('../configs/data.yaml','r') as f:
    data = yaml.safe_load(f)

logger = logging.getLogger(__name__)

def build_returns_tensor(underlying_df, returns_config):
    """
    Build the returns feature tensor for one (date, symbol)
    
    underlying_df: DataFrame of underlying OHLCV for the TRAILING L days,
            sorted ascending by date. Must have columns:
            [date, adj_close, log_return, ...]
            Must already have log_return computed.
        returns_config: the data.yaml returns section

    Returns:
        numpy array of shape (L, F_ret) where F_ret = number of enabled features
    """
    L = returns_config['lookback_window'] # 60
    features_config = returns_config['features']

    if len(underlying_df) < L:
        raise ValueError(f"underlying_df must have at least {L} rows")
    
    # take last L rows
    window = underlying_df.tail(L).copy().reset_index(drop=True)
    
    columns = []

    # log returns
    if features_config.get("log_returns", True):
        columns.append(window['log_return'].values)

    # realized volatility
    rv_windows = features_config.get('realized_vol_windows', [5,10,21])
    for rv_w in rv_windows:
        rv = window['log_return'].rolling(rv_w).apply(
            lambda x: np.sqrt((x**2).sum()), raw=True
        ).values
        # fill leading NaNs with 0 (begining of window)
        rv = np.nan_to_num(rv, nan=0.0)
        columns.append(rv)

    # jump indicator
    if features_config.get('jump_indicator', True):
        threshold_sigma = features_config.get('jump_threshold_sigma', 2.5)
        rolling_std = window['log_return'].rolling(21).std().values
        rolling_std = np.nan_to_num(rolling_std, nan=999.0) # no jumps if insufficient data
        jump = (np.abs(window['log_return'].values) > threshold_sigma * rolling_std).astype(np.float32)
        columns.append(jump)

    # rolling beta to SPY
    if features_config.get('rolling_beta', False):
        # this requires SPY returns to be pre-joined to the underlying_df
        # if 'spy_return' column exists, compute rolling OLS beta
        beta_window = features_config.get('rolling_beta_window',60)
        if 'spy_return' in window.columns:
            betas = []
            for i in range(len(window)):
                if i < beta_window - 1:
                    betas.append(0.0)
                else:
                    y = window['log_return'].iloc[i - beta_window + 1:i + 1].values
                    x = window['spy_return'].iloc[i - beta_window + 1:i + 1].values
                    # simple OLS: beta = cov(x,y) / var(x)
                    if np.var(x) > 0:
                        beta = np.cov(y,x)[0,1] / np.var(x)
                    else:
                        beta = 0.0
                    betas.append(beta)
            columns.append(np.array(betas, dtype=np.float32))
        else:
            # no SPY data available, fill with zeros
            columns.append(np.zeros(L, dtype=np.float32))
    
    # stack into (L, F_ret)
    tensor = np.stack(columns, axis=1).astype(np.float32)
    
    return tensor

def build_vol_history_vector(vol_history_row):
    """
    Extract the vol-history feature vector for one (date, symbol).

    Args:
        vol_history_row: single row (Series or dict) from the vol_history features DataFrame.
            Must contain the computed feature columns:
            iv_rank, hv_rank, vol_risk_premium,
            iv_momentum_short, iv_momentum_medium,
            hv_momentum_short, hv_momentum_medium,
            days_since_iv_year_high, days_since_iv_year_low,
            days_since_hv_year_high, days_since_hv_year_low
    
    Returns:
        numpy array of shape (F_vh,) - about 11 features
    """
    feature_cols = [
        "iv_rank", "hv_rank", "vol_risk_premium",
        "iv_momentum_short", "iv_momentum_medium",
        "hv_momentum_short", "hv_momentum_medium",
        "days_since_iv_year_high", "days_since_iv_year_low",
        "days_since_hv_year_high", "days_since_hv_year_low",
    ]

    values = []
    for col in feature_cols:
        val = vol_history_row.get(col, np.nan)
        values.append(float(val) if val is not None else np.nan)

    return np.array(values, dtype=np.float32) # shape: (11,)

def build_market_state_vector(market_state_row, config_market_state):
    """
    Extract market state features for one date.

    Args:
        market_state_row: single row from market_state DataFrame
        config_market_state: market_state section from data.yaml

    Returns:
        numpy array of shape (F_mkt,) — 2-3 features
    """
    values = []
    if config_market_state.get('vix', True):
        values.append(float(market_state_row.get('vix',0.0)))
    if config_market_state.get('spy_return', True):
        values.append(float(market_state_row.get('spy_return',0.0)))
    if config_market_state.get('risk_free_rate', False):
        values.append(float(market_state_row.get('risk_free_rate',0.0)))
    
    return np.array(values, dtype=np.float32)

    
