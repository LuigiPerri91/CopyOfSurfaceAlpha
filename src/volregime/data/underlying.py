import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def fetch_underlying(tickers, start, end, cache_dir=None):
    """
    Download daily OHLCV + Adj close for all tickers via yfinance

    Args:
        tickers: list of yfinance tickers
        start: start date string 'YYYY-MM-DD' (should be padded by ~252 days for MA warmup)
        end: end date string 'YYYY-MM-DD'
        cache_dir: if provided, check foir cached parquets before downloading

    Returns:
        dict of {ticker: DataFrame} with columns [open, high, low, close, adj_close, volume]
    """
    results = {}
    failed =[]

    if cache_dir:
        cache_dir = Path(cache_dir)
        for ticker in tickers:
            cache_path = cache_dir / f"{ticker}.parquet"
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                if df['date'].min() <= pd.Timestamp(start).date() and df['date'].max() >= pd.Timestamp(end).date():
                    logger.info('Cache hit for %s', ticker)
                    results[ticker] = df

    tickers_to_fetch = [t for t in tickers if t not in results]

    if tickers_to_fetch:
        logger.info("Downloading %d tickers form yfinance: %s", len(tickers_to_fetch), tickers_to_fetch)

        raw = yf.download(tickers_to_fetch, start=start, end=end, group_by='ticker', auto_adjust=False)
        for ticker in tickers_to_fetch:
            try:
                if len(tickers_to_fetch) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()

                df.columns = [c.lower().replace(" ", "_") for c in df.columns]

                if 'adj_close' not in df.columns and 'adj close' in df.columns:
                    df = df.rename(columns={"adj close": 'adj_close'})

                df =df.reset_index()
                df = df.rename(columns={'Date':'date','date':'date'})
                df['date'] = pd.to_datetime(df['date']).dt.date

                df = df.dropna(subset=['close'])

                if len(df) == 0:
                    raise ValueError(f"No data returned for {ticker}")
                
                date_series = pd.to_datetime(pd.Series([d for d in df['date']]))
                gaps = date_series.diff().dt.date
                max_gap = gaps.max()
                if max_gap > 7:
                    logger.warning("%s has a %d-gap in price data", ticker, max_gap)
                
                results[ticker] = df

                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_dir / f'{ticker}.parquet', index=False)

            except Exception as e:
                logger.error("Failed to process %s: %s", ticker, e)
                failed.append(ticker)

    logger.info("Underlying fetch complete: %d succeded, %d failed %s", len(results), len(failed), failed if failed else "")

    return results, failed

def compute_log_returns(df):
    """
    Given an underlying DataFrame with 'adj_close' column, add a 'log_return' column
    """
    df = df.copy()
    df['log_return'] = np.log(df['adj_close']/ df['adj_close'].shift(1))
    return df