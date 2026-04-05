import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def fetch_market_state(start, end, config_market_state):
    """
    Fetch VIX, SPY returns, and optional risk-free rate.

    Args:
        start: padded start date (252 days before config start for indicator warmup)
        end: end date
        config_market_state; the market_state section in data.yaml

    Returns:
        DataFrame keyeed with columns [vix, spy_return, risk_free_rate]
        (columns only present if enabled in config)
    """
    frames = {}

    ## VIX
    if config_market_state.get('vix', True):
        logger.info("Fetching VIX...")
        vix_raw = yf.download('^VIX', start=start, end=end)
        vix = vix_raw[['Close']].copy()
        vix.columns = ['vix']
        vix = vix.reset_index()
        vix = vix.rename(columns={'Date':'date'})
        vix['date'] = pd.to_datetime(vix['date']).dt.date
        frames['vix'] = vix.set_index('date')

    ## SPY returns + derived macro indicators
    _spy_features = (
        config_market_state.get('spy_return', True)
        or config_market_state.get('spy_pct_from_ma200', False)
        or config_market_state.get('spy_adx14', False)
        or config_market_state.get('spy_atr_ratio', False)
    )
    if _spy_features:
        logger.info("Fetching SPY OHLCV for market features...")
        spy_raw = yf.download('SPY', start=start, end=end, auto_adjust=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = [c[0] for c in spy_raw.columns]

        spy_close = spy_raw['Adj Close']
        spy_high  = spy_raw['High']
        spy_low   = spy_raw['Low']

        spy_cols = {}

        if config_market_state.get('spy_return', True):
            spy_cols['spy_return'] = np.log(spy_close / spy_close.shift(1))

        if config_market_state.get('spy_pct_from_ma200', False):
            ma200 = spy_close.rolling(200).mean()
            spy_cols['spy_pct_from_ma200'] = (spy_close - ma200) / ma200

        if config_market_state.get('spy_adx14', False) or config_market_state.get('spy_atr_ratio', False):
            prev_close = spy_close.shift(1)
            tr = pd.concat([
                spy_high - spy_low,
                (spy_high - prev_close).abs(),
                (spy_low  - prev_close).abs(),
            ], axis=1).max(axis=1)

            if config_market_state.get('spy_adx14', False):
                period = 14
                prev_high = spy_high.shift(1)
                prev_low  = spy_low.shift(1)
                plus_dm  = (spy_high - prev_high).clip(lower=0)
                minus_dm = (prev_low  - spy_low ).clip(lower=0)
                plus_dm  = plus_dm.where(plus_dm  >= minus_dm, 0.0)
                minus_dm = minus_dm.where(minus_dm >  plus_dm,  0.0)
                alpha = 1.0 / period
                atr14     = tr.ewm(alpha=alpha, adjust=False).mean()
                plus_di   = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean()  / atr14
                minus_di  = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr14
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
                spy_cols['spy_adx14'] = dx.ewm(alpha=alpha, adjust=False).mean()

            if config_market_state.get('spy_atr_ratio', False):
                atr_short = tr.rolling(10).mean()
                atr_long  = tr.rolling(50).mean()
                spy_cols['spy_atr_ratio'] = atr_short / atr_long.replace(0, np.nan)

        spy_df = pd.DataFrame(spy_cols)
        spy_df = spy_df.reset_index().rename(columns={'Date': 'date'})
        spy_df['date'] = pd.to_datetime(spy_df['date']).dt.date
        frames['spy'] = spy_df.set_index('date')

    ## risk free rate
    if config_market_state.get('risk_free_rate', True):
        logger.info("Fetching risk-free rate proxy (^IRX)...")
        irx_raw = yf.download('^IRX', start=start, end=end)
        irx = irx_raw[['Close']].copy()
        irx.columns = ['risk_free_rate_annualized']
        irx['risk_free_rate'] = irx['risk_free_rate_annualized'] / 100  # annualized fraction (0.01–0.05)
        irx = irx[['risk_free_rate']].copy()
        irx = irx.reset_index()
        irx = irx.rename(columns={'Date':'date'})
        irx['date'] = pd.to_datetime(irx['date']).dt.date
        frames['irx'] = irx.set_index('date')

    ## merge all on date
    if not frames:
        raise ValueError("No market state features enabled in config")
    
    result = frames[list(frames.keys())[0]]
    for key in list(frames.keys())[1:]:
        result = result.join(frames[key], how='outer')
    
    result = result.ffill(limit=3)

    result = result.dropna()

    result = result.reset_index()
    result = result.rename(columns= {'index': 'date'})

    logger.info("Market state: %d rows, columns: %s", len(result), list(result.columns))
    return result