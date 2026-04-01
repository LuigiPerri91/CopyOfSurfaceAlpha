"""
Historical options client backed by philippdubach/options-data.

Covers 2008-01-02 through 2025-12-16 for 104 US symbols with pre-computed
IV + Greeks and real bid/ask prices. No API key required — parquet files are
downloaded once from the CDN and cached locally.

Source:  https://github.com/philippdubach/options-data
Files:   https://static.philippdubach.com/data/options/{ticker}/options.parquet
         https://static.philippdubach.com/data/options/{ticker}/underlying.parquet

Mirrors the DoltClient interface (query_option_chain / query_vol_history /
get_provenance) so it acts as a drop-in source for dates before DoltHub
coverage begins (2019-02-09).

Usage:
    client = DubachClient(cache_dir="./data/raw/dubach_cache")
    df_chain = client.query_option_chain(["SPY"], "2015-01-01", "2019-02-08")
    df_vol   = client.query_vol_history(["SPY"], "2015-01-01", "2019-02-08")
    client.save_provenance("./data/raw/dubach_provenance.json")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_BASE_URL = "https://static.philippdubach.com/data/options"

# 104 tickers available in the dataset (lowercase, matching CDN URL paths)
_AVAILABLE = {
    "aapl", "abbv", "abt", "acn", "adbe", "aig", "amd", "amgn", "amt", "amzn",
    "avgo", "axp", "ba", "bac", "bk", "bkng", "blk", "bmy", "brk.b", "c",
    "cat", "cl", "cmcsa", "cof", "cop", "cost", "crm", "csco", "cvs", "cvx",
    "de", "dhr", "dis", "duk", "emr", "fdx", "gd", "ge", "gild", "gm",
    "goog", "googl", "gs", "hd", "hon", "ibm", "intu", "isrg", "iwm", "jnj",
    "jpm", "ko", "lin", "lly", "lmt", "low", "ma", "mcd", "mdlz", "mdt",
    "met", "meta", "mmm", "mo", "mrk", "ms", "msft", "nee", "nflx", "nke",
    "now", "nvda", "orcl", "pep", "pfe", "pg", "pltr", "pm", "pypl", "qcom",
    "qqq", "rtx", "sbux", "schw", "so", "spg", "spy", "t", "tgt", "tmo",
    "tmus", "tsla", "txn", "uber", "unh", "unp", "ups", "usb", "v", "vix",
    "vz", "wfc", "wmt", "xom",
}


class DubachClient:
    """
    Historical options client backed by philippdubach/options-data.

    Exposes the same interface as DoltClient:
        query_option_chain()  →  date, act_symbol, expiration, strike, call_put,
                                  bid, ask, vol, delta, gamma, theta, vega, rho
        query_vol_history()   →  date, act_symbol, iv_current/week_ago/month_ago/
                                  year_high/year_low, hv_current/...
        get_provenance()
        save_provenance()
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        for sub in ("options", "underlying", "underlying_yf", "vol_history"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    # Public interface
    def query_option_chain(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Return options chain data filtered to [start, end], schema matching DoltClient."""
        frames = [df for s in symbols if not (df := self._load_options(s, start, end)).empty]

        if not frames:
            logger.warning("No option chain data returned for symbols=%s", symbols)
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "act_symbol", "expiration", "strike", "call_put"]
        ).reset_index(drop=True)
        logger.info(
            "option_chain: %d rows, %d symbols, dates %s → %s",
            len(out), out["act_symbol"].nunique(), out["date"].min(), out["date"].max(),
        )
        return out

    def query_vol_history(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Derive volatility_history rows matching the DoltHub schema.

        Both chain and underlying are padded 400 calendar days before start so
        rolling stats (iv_week_ago, iv_year_high, etc.) are accurate from day one.
        """
        # ~252 trading days of warm-up before the first output date
        padded_start = (pd.Timestamp(start) - pd.DateOffset(days=400)).strftime("%Y-%m-%d")

        frames = []
        for symbol in symbols:
            cache_path = self.cache_dir / "vol_history" / f"{symbol.lower()}_{start}_{end}.parquet"
            if cache_path.exists():
                logger.info("Vol history cache hit for %s [%s → %s]", symbol, start, end)
                frames.append(pd.read_parquet(cache_path))
                continue

            chain_df = self._load_options(symbol, padded_start, end)
            underlying_df = self._load_underlying(symbol, padded_start, end)
            df = self._compute_vol_history(chain_df, underlying_df, symbol, start, end)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
                frames.append(df)

        if not frames:
            logger.warning("No vol_history data for symbols=%s", symbols)
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "act_symbol"]
        ).reset_index(drop=True)
        logger.info(
            "volatility_history: %d rows, %d symbols, dates %s → %s",
            len(out), out["act_symbol"].nunique(), out["date"].min(), out["date"].max(),
        )
        return out

    def get_provenance(self) -> dict:
        return {
            "source": "philippdubach/options-data",
            "url": "https://github.com/philippdubach/options-data",
            "data_url": _BASE_URL,
            "extracted_at": datetime.utcnow().isoformat() + "Z",
            "cache_dir": str(self.cache_dir),
        }

    def save_provenance(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_provenance(), f, indent=2)
        logger.info("Provenance saved to %s", path)

    # Data loading
    def _load_options(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download (once) and return options chain for symbol filtered to [start, end]."""
        ticker = symbol.lower()
        if ticker not in _AVAILABLE:
            logger.warning("'%s' not available in philippdubach/options-data — skipping.", symbol)
            return pd.DataFrame()

        local = self._download("options", ticker)
        if local is None:
            return pd.DataFrame()

        df = pd.read_parquet(local)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        start_dt, end_dt = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"symbol": "act_symbol", "type": "call_put", "implied_volatility": "vol"})
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        df["act_symbol"] = df["act_symbol"].str.upper()

        keep = ["date", "act_symbol", "expiration", "strike", "call_put",
                "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    def _load_underlying(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Load underlying daily adjusted close prices.

        Tries philippdubach underlying.parquet first; falls back to yfinance
        if the file is unavailable or missing an adjusted_close column.
        """
        ticker = symbol.lower()
        if ticker in _AVAILABLE:
            local = self._download("underlying", ticker)
            if local is not None:
                try:
                    raw = pd.read_parquet(local)
                    raw["date"] = pd.to_datetime(raw["date"]).dt.date
                    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
                    if "adj_close" in raw.columns:
                        raw = raw.rename(columns={"adj_close": "adjusted_close"})
                    if "adjusted_close" in raw.columns:
                        start_dt, end_dt = pd.Timestamp(start).date(), pd.Timestamp(end).date()
                        df = raw[(raw["date"] >= start_dt) & (raw["date"] <= end_dt)]
                        if not df.empty:
                            return df.reset_index(drop=True)
                except Exception as exc:
                    logger.warning("Could not read philippdubach underlying for %s: %s", symbol, exc)

        logger.info("Falling back to yfinance for %s underlying prices", symbol)
        return self._fetch_underlying_yfinance(symbol, start, end)

    def _fetch_underlying_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch adjusted close prices via yfinance, cached locally."""
        cache_path = self.cache_dir / "underlying_yf" / f"{symbol.lower()}_{start}_{end}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        try:
            hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                return pd.DataFrame()
            hist.index = hist.index.tz_localize(None).date
            df = pd.DataFrame({"date": hist.index, "adjusted_close": hist["Close"].values})
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as exc:
            logger.error("yfinance fallback failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def _download(self, file_type: str, ticker: str) -> Optional[Path]:
        """Download a CDN parquet file to local cache. Returns path or None on failure."""
        local = self.cache_dir / file_type / f"{ticker}.parquet"
        if local.exists():
            return local

        url = f"{_BASE_URL}/{ticker}/{file_type}.parquet"
        logger.info("Downloading %s", url)
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(local, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
            return local
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            if local.exists():
                local.unlink()
            return None

    # Vol history computation
    def _compute_vol_history(
        self,
        chain_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Build volatility_history rows from ATM 30-day IV series + realised HV."""
        if chain_df.empty or underlying_df.empty:
            return pd.DataFrame()

        chain_df = chain_df.copy()
        chain_df["date"] = pd.to_datetime(chain_df["date"]).dt.date
        chain_df["expiration"] = pd.to_datetime(chain_df["expiration"]).dt.date

        underlying_df = underlying_df.sort_values("date")
        prices = pd.Series(
            underlying_df["adjusted_close"].values,
            index=underlying_df["date"].values,
            name=symbol,
        )

        iv_by_date: dict[date, float] = {}
        for trade_date, group in chain_df.groupby("date"):
            S = prices.get(trade_date)
            if S is None or pd.isna(S):
                continue
            group = group.copy()
            group["dte"] = [(e - trade_date).days for e in group["expiration"]]
            group["moneyness_dist"] = (group["strike"] - S).abs()

            candidates = group[(group["dte"] >= 15) & (group["dte"] <= 60)]
            if candidates.empty:
                candidates = group[group["dte"] > 0]
            if candidates.empty:
                continue

            near30 = candidates[(candidates["dte"] - 30).abs() == (candidates["dte"] - 30).abs().min()]
            atm_row = near30.loc[near30["moneyness_dist"].idxmin()]
            iv = float(atm_row["vol"])
            if not pd.isna(iv) and iv > 0:
                iv_by_date[trade_date] = iv

        if not iv_by_date:
            return pd.DataFrame()

        iv_series = pd.Series(iv_by_date, name="iv_current").sort_index()
        hv_series = (
            np.log(prices / prices.shift(1)).dropna().rolling(21).std() * np.sqrt(252)
        ).rename("hv_current")

        start_dt, end_dt = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        lags = {"week_ago": 5, "month_ago": 21}

        rows = []
        for d in (d for d in iv_series.index if start_dt <= d <= end_dt):
            iv_s = self._rolling_stats(iv_series, d, lags)
            hv_s = self._rolling_stats(hv_series, d, lags)
            rows.append({
                "date": d, "act_symbol": symbol,
                "hv_current": hv_s["current"], "hv_week_ago": hv_s["week_ago"],
                "hv_month_ago": hv_s["month_ago"], "hv_year_high": hv_s["year_high"],
                "hv_year_high_date": hv_s["year_high_date"], "hv_year_low": hv_s["year_low"],
                "hv_year_low_date": hv_s["year_low_date"], "iv_current": iv_s["current"],
                "iv_week_ago": iv_s["week_ago"], "iv_month_ago": iv_s["month_ago"],
                "iv_year_high": iv_s["year_high"], "iv_year_high_date": iv_s["year_high_date"],
                "iv_year_low": iv_s["year_low"], "iv_year_low_date": iv_s["year_low_date"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _rolling_stats(series: pd.Series, as_of: date, lags: dict[str, int]) -> dict:
        past = series[series.index <= as_of]
        current = float(past.iloc[-1]) if not past.empty else np.nan
        result: dict = {"current": round(current, 4) if not pd.isna(current) else np.nan}

        for name, n in lags.items():
            lagged = past.iloc[:-n] if len(past) > n else pd.Series(dtype=float)
            result[name] = round(float(lagged.iloc[-1]), 4) if not lagged.empty else np.nan

        window = past.iloc[-252:] if len(past) >= 252 else past
        if not window.empty:
            result.update({
                "year_high": round(float(window.max()), 4),
                "year_high_date": window.idxmax(),
                "year_low": round(float(window.min()), 4),
                "year_low_date": window.idxmin(),
            })
        else:
            result.update({"year_high": np.nan, "year_high_date": None,
                            "year_low": np.nan, "year_low_date": None})
        return result

    def __repr__(self) -> str:
        return f"DubachClient(cache_dir='{self.cache_dir}')"
