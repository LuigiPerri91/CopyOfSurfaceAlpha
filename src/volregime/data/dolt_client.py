"""
DoltHub client for the post-no-preference/options repository.

Supports two access modes (set via configs/data.yaml dolthub.access_method):

  "api"   — HTTP SQL API (default, recommended).
            Queries DoltHub directly over HTTPS. No local clone required.
            Fast for small date windows (e.g. the recent 3-month gap).

  "clone" — Local Dolt clone (legacy).
            Clones the full repo locally then queries via SQL.
            Slow on first run (repo is multi-GB), fast after caching.
            Use only if you need offline access or the full history.

Usage:
    client = DoltClient(repo="post-no-preference/options", access_method="api")
    df = client.query_option_chain(["SPY"], "2025-12-17", "2026-03-20")
    df = client.query_vol_history(["SPY"], "2025-12-17", "2026-03-20")
    meta = client.get_provenance()
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_DOLTHUB_API = "https://www.dolthub.com/api/v1alpha1"


class DoltClient:
    """
    DoltHub options client.

    Defaults to the HTTP API mode — no clone, no pull, just fast SQL over HTTPS.
    Pass access_method="clone" to use the legacy local-clone approach.
    """

    def __init__(
        self,
        repo: str,
        access_method: str = "api",
        clone_dir: str | Path | None = None,
        branch: str = "master",
    ) -> None:
        """
        Args:
            repo:          DoltHub repo path, e.g. "post-no-preference/options".
            access_method: "api" (HTTP SQL, fast) or "clone" (local clone, slow).
            clone_dir:     Required when access_method="clone".
            branch:        DoltHub branch to query (default "main").
        """
        self.repo = repo
        self.access_method = access_method
        self.branch = branch
        self.clone_dir = Path(clone_dir) if clone_dir else None

        # API key (optional — increases rate limits and unlocks private repos)
        api_key = os.environ.get("DOLTHUB_TOKEN", "") or os.environ.get("DOLTHUB_API_KEY", "")
        self._api_headers: dict[str, str] = (
            {"authorization": api_key} if api_key else {}
        )

        # clone-mode internals
        self._dolt = None
        self._commit_hash: Optional[str] = None

    # ──────────────────────────────────────────────
    # Connection (clone mode only)
    # ──────────────────────────────────────────────

    def connect(self) -> None:
        """Clone mode: clone the repo if needed, otherwise open the existing clone."""
        if self.access_method == "api":
            return  # no connection needed for API mode

        from doltcli import Dolt
        assert self.clone_dir is not None, "clone_dir is required for access_method='clone'"
        dolt_meta_dir = self.clone_dir / ".dolt"
        if dolt_meta_dir.is_dir():
            logger.info("Opening existing Dolt clone at %s", self.clone_dir)
            self._dolt = Dolt(str(self.clone_dir))
        else:
            logger.info("Cloning %s → %s (this may take a while on first run)", self.repo, self.clone_dir)
            self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            self._dolt = Dolt.clone(self.repo, str(self.clone_dir))
        self._commit_hash = self._dolt.head
        logger.info("Connected — commit %s, branch %s", self._commit_hash[:12], self._dolt.active_branch)

    def pull(self) -> None:
        """Clone mode: pull latest changes from DoltHub remote."""
        if self.access_method == "api":
            return  # API mode always queries latest
        self._ensure_clone_connected()
        old_hash = self._commit_hash
        self._dolt.pull("origin")
        self._commit_hash = self._dolt.head
        if self._commit_hash != old_hash:
            logger.info("Pulled new data: %s → %s", old_hash[:12], self._commit_hash[:12])
        else:
            logger.info("Already up to date at %s", self._commit_hash[:12])

    # Queries
    def query_option_chain(
        self,
        symbols: list[str],
        start: str,
        end: str,
        batch_size: int = 50_000,
    ) -> pd.DataFrame:
        """
        Query the option_chain table for the given symbols and date range.

        Schema:
            date, act_symbol, expiration, strike, call_put,
            bid, ask, vol, delta, gamma, theta, vega, rho
        """
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            logger.info("Querying option_chain for %s [%s → %s]", symbol, start, end)
            if self.access_method == "api":
                # Range queries hit a server-side deadline; query one date at a time instead.
                df = self._query_api_by_date(
                    table="option_chain",
                    cols="date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho",
                    symbol=symbol,
                    start=start,
                    end=end,
                )
            else:
                sql = (
                    f"SELECT date, act_symbol, expiration, strike, call_put, "
                    f"bid, ask, vol, delta, gamma, theta, vega, rho "
                    f"FROM option_chain "
                    f"WHERE act_symbol = '{symbol}' "
                    f"  AND date >= '{start}' AND date <= '{end}'"
                )
                df = self._query(sql, batch_size)
            if not df.empty:
                frames.append(df)

        if not frames:
            logger.warning("No option_chain data returned for symbols=%s", symbols)
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = self._cast_option_chain_dtypes(df)
        logger.info(
            "option_chain: %d rows, %d symbols, dates %s → %s",
            len(df), df["act_symbol"].nunique(), df["date"].min(), df["date"].max(),
        )
        return df

    def query_vol_history(
        self,
        symbols: list[str],
        start: str,
        end: str,
        batch_size: int = 50_000,
    ) -> pd.DataFrame:
        """
        Query the volatility_history table for the given symbols and date range.

        Schema:
            date, act_symbol,
            hv_current/week_ago/month_ago/year_high/year_high_date/year_low/year_low_date,
            iv_current/week_ago/month_ago/year_high/year_high_date/year_low/year_low_date
        """
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            logger.info("Querying volatility_history for %s [%s → %s]", symbol, start, end)
            if self.access_method == "api":
                df = self._query_api_by_date(
                    table="volatility_history",
                    cols="*",
                    symbol=symbol,
                    start=start,
                    end=end,
                )
            else:
                sql = (
                    f"SELECT * FROM volatility_history "
                    f"WHERE act_symbol = '{symbol}' "
                    f"  AND date >= '{start}' AND date <= '{end}'"
                )
                df = self._query(sql, batch_size)
            if not df.empty:
                frames.append(df)

        if not frames:
            logger.warning("No volatility_history data returned for symbols=%s", symbols)
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = self._cast_vol_history_dtypes(df)
        logger.info(
            "volatility_history: %d rows, %d symbols, dates %s → %s",
            len(df), df["act_symbol"].nunique(), df["date"].min(), df["date"].max(),
        )
        return df

    # Provenance
    def get_provenance(self) -> dict:
        meta: dict = {
            "dolthub_repo": self.repo,
            "access_method": self.access_method,
            "branch": self.branch,
            "extracted_at": datetime.utcnow().isoformat() + "Z",
        }
        if self.access_method == "clone" and self._commit_hash:
            meta["dolt_commit_hash"] = self._commit_hash
            meta["clone_dir"] = str(self.clone_dir)
        return meta

    def save_provenance(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_provenance(), f, indent=2)
        logger.info("Provenance saved to %s", path)

    @property
    def is_connected(self) -> bool:
        return self.access_method == "api" or self._dolt is not None

    # Internal: query dispatch
    def _query_api_by_date(
        self, table: str, cols: str, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        API mode: DoltHub range queries exceed the server deadline on large tables.
        Query one calendar date at a time using equality, which hits the index.
        """
        dates = pd.date_range(start=start, end=end, freq="D")
        frames = []
        owner, name = self.repo.split("/", 1)
        url = f"{_DOLTHUB_API}/{owner}/{name}/{self.branch}"
        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            sql = (
                f"SELECT {cols} FROM {table} "
                f"WHERE act_symbol = '{symbol}' AND date = '{date_str}'"
            )
            resp = requests.get(url, params={"q": sql}, headers=self._api_headers, timeout=60)
            resp.raise_for_status()
            rows = resp.json().get("rows", [])
            if rows:
                frames.append(pd.DataFrame(rows))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _query(self, sql: str, batch_size: int) -> pd.DataFrame:
        if self.access_method == "api":
            return self._query_api(sql, batch_size)
        return self._query_clone(sql, batch_size)

    def _query_api(self, sql: str, batch_size: int) -> pd.DataFrame:
        """Execute paginated SQL via the DoltHub HTTP API."""
        owner, name = self.repo.split("/", 1)
        url = f"{_DOLTHUB_API}/{owner}/{name}/{self.branch}"
        frames = []
        offset = 0

        while True:
            paginated = f"{sql} LIMIT {batch_size} OFFSET {offset}"
            resp = requests.get(url, params={"q": paginated}, headers=self._api_headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("rows", [])
            if not rows:
                break

            frames.append(pd.DataFrame(rows))
            if len(rows) < batch_size:
                break
            offset += batch_size

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _query_clone(self, sql: str, batch_size: int) -> pd.DataFrame:
        """Execute paginated SQL against the local Dolt clone."""
        self._ensure_clone_connected()
        frames = []
        offset = 0

        while True:
            paginated = f"{sql} LIMIT {batch_size} OFFSET {offset}"
            rows = self._dolt.sql(paginated, result_format="csv")
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            if len(rows) < batch_size:
                break
            offset += batch_size

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Internal: dtype casting
    @staticmethod
    def _cast_option_chain_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
        for col in ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["act_symbol"] = df["act_symbol"].astype(str).str.strip()
        df["call_put"] = df["call_put"].astype(str).str.strip()
        return df

    @staticmethod
    def _cast_vol_history_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        for col in ["date", "hv_year_high_date", "hv_year_low_date",
                    "iv_year_high_date", "iv_year_low_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        for col in ["hv_current", "hv_week_ago", "hv_month_ago", "hv_year_high", "hv_year_low",
                    "iv_current", "iv_week_ago", "iv_month_ago", "iv_year_high", "iv_year_low"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["act_symbol"] = df["act_symbol"].astype(str).str.strip()
        return df

    def _ensure_clone_connected(self) -> None:
        if self._dolt is None:
            raise RuntimeError("DoltClient not connected. Call .connect() first.")

    def __repr__(self) -> str:
        if self.access_method == "api":
            return f"DoltClient(repo='{self.repo}', mode=api)"
        status = f"commit={self._commit_hash[:12]}" if self._commit_hash else "not connected"
        return f"DoltClient(repo='{self.repo}', mode=clone, {status})"
