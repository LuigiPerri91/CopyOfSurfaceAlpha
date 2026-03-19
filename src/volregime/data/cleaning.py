import pandas as pd
import logging

logger = logging.getLogger(__name__)

def standardize_call_put(df):
    """Normalize call_put column to 'C' / 'P' ."""
    df = df.copy()
    df['call_put'] = df['call_put'].str.strip().str.upper()
    # handle both "CALL"/"PUT" and "C"/"P" formats
    df['call_put'] = df['call_put'].map(lambda x: 'C' if x.startswith('C') else 'P')
    
    # drop rows that didn't map
    bad_rows = ~df['call_put'].isin(['C', 'P'])
    if bad_rows.any():
        logger.warning("Dropped %d rows with invalid call_put values", bad_rows.sum())
        df = df[~bad_rows]
    return df

def rename_iv_column(df):
    """Rename IV column to iv"""
    if 'vol' in df.columns and "iv" not in df.columns:
        df = df.rename(columns={'vol': 'iv'})
    return df
    
def filter_quality(df, filters_config):
    """
    Apply all quality/liquidity filters from data.yaml filters section.

    Args:
        df: DataFrame with columns [bid, ask, iv, delta,...]
        filters_config: dict with keys min_bid, max_spread_pct, min_iv, max_iv, min_delta_abs, max_delta_abs

    Returns:
        filtered DataFrame
    """
    n_before = len(df)

    # bid must be positive and above minimum
    df = df[df['bid'] >= filters_config['min_bid']]

    # ask must be >= bid
    df = df[df['ask'] >= df['bid']]

    # iv bounds
    df = df[df['iv'] >= filters_config['min_iv']]
    df = df[df['iv'] <= filters_config['max_iv']]

    # bid-ask spread: (ask-bid)/midpoint
    midpoint = (df['ask'] + df['bid']) / 2
    spread_pct = (df['ask'] - df['bid']) / midpoint
    df = df[spread_pct <= filters_config['max_spread_pct']]

    # delta bounds (absolute value)
    df = df[df['delta'].abs() >= filters_config['min_delta_abs']]
    df = df[df['delta'].abs() <= filters_config['max_delta_abs']]

    n_after =len(df)
    logger.info("Quality filter: %d -> %d rows (droped %d, %.1f%%)",
                n_before, n_after, n_before-n_after, 100 * (n_before- n_after) / max(n_before, 1))
    return df

def filter_maturity(df, min_days, max_days):
    """
    Filter by time to maturity
    requires 'tau' column already computed (expiration - date)/365.
    """
    min_tau = min_days / 365
    max_tau = max_days / 365

    n_before = len(df)
    df = df[(df['tau'] >= min_tau) & (df['tau'] <= max_tau)]
    logger.info("Maturity filter [%d,%d] days: %d -> %d rows", min_days, max_days, n_before, len(df))
    return df

def filter_moneyness(df, moneyness_min, moneyness_max):
    """
    Filter by moneyness = strike / spot.
    Requires 'moneyness' column already computed (after underlying join).
    """
    n_before = len(df)
    df = df[(df["moneyness"] >= moneyness_min) & (df["moneyness"] <= moneyness_max)]
    logger.info("Moneyness filter [%.2f, %.2f]: %d → %d rows",
                moneyness_min, moneyness_max, n_before, len(df))
    return df

def detect_obs_frequency(dates_series):
    """
    Given a sorted series of dates for a single symbol, classify observation frequency.

    Returns: "daily", "mwf", or "weekly"
    """
    if len(dates_series) <2:
        return "unknown"
    
    dates = pd.to_datetime(pd.Series(sorted(dates_series.unique())))
    gaps = dates.diff().dt.days.dropna()
    median_gap = gaps.median()

    if median_gap <= 1.5:
        return 'daily'
    elif median_gap <= 3.0:
        return 'mwf'
    else:
        return 'weekly'

def detect_gaps(dates_for_symbol, full_trading_calendar, max_gap_days, policy):
    """
    Find missing dates for a symbol

    Args:
        dates_for_symbol: set of dates where data exists
        full_tading_calendar: lsit of all trading days in the period
        max_gap_days: gaps longer than this are marked 'drop' not 'fill'
        policy: 'skip', 'forward_fill', or 'mark_stale'

    Returns:
        list of dicts: [{date, action: "keep"/"fill"/"drop", gaps_days: int}]
    """
    result = []
    last_obs_date = None

    for trade_date in full_trading_calendar:
        if trade_date in dates_for_symbol:
            result.append({'date':trade_date, 'action':'keep','gap_days': 0})
            last_obs_date = trade_date
        else:
            if last_obs_date is None:
                #no data yet for this symbol
                result.append({'date':trade_date, 'action':'drop','gap_days': 0})
                continue

            gap = (trade_date - last_obs_date).days
            if gap > max_gap_days:
                result.append({'date':trade_date, 'action':'drop','gap_days': gap})
            elif policy == 'skip':
                result.append({'date':trade_date, 'action':'drop','gap_days': gap})
            else:
                #forward_fill or mark_stale
                result.append({'date':trade_date, 'action':'fill','gap_days': gap})
                
    return result
