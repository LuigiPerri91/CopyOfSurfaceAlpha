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

    ## SPY returns
    if config_market_state.get('spy_return', True):
        logger.info("Fetching SPY for market returns...")
        spy_raw = yf.download('SPY', start=start, end=end, auto_adjust=False)
        spy = spy_raw[['Adj Close']].copy()
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = ['Adj Close']
        spy.columns = ['spy_adj_close']
        spy['spy_return'] = np.log(spy['spy_adj_close'] / spy['spy_adj_close'].shift(1))
        spy = spy[['spy_return']].copy()
        spy = spy.reset_index()
        spy = spy.rename(columns={'Date':'date'})
        spy['date'] = pd.to_datetime(spy['date']).dt.date
        frames['spy'] = spy.set_index('date')

    ## risk free rate
    if config_market_state.get('risk_free_rate', True):
        logger.info("Fetching risk-free rate proxy (^IRX)...")
        irx_raw = yf.download('^IRX', start=start, end=end)
        irx = irx_raw[['Close']].copy()
        irx.columns = ['risk_free_rate_annualized']
        irx['risk_free_rate'] = (irx['risk_free_rate_annualized'] / 100) / 252
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