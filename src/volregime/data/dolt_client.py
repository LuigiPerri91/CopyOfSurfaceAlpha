"""
DoltHub client wrapper for the post-no-preference/options repository.

Handles cloning, querying option_chain and volatility_history tables,
and recording provenance metadata for reproducibility.

Usage:
    client = DoltClient(repo="post-no-preference/options", clone_dir="./data/raw/dolt_clone")
    client.connect()  # clones if needed, or opens existing

    df = client.query_option_chain(symbols=["SPY"], start="2020-01-01", end="2024-12-31")
    df = client.query_vol_history(symbols=["SPY"], start="2020-01-01", end="2024-12-31")

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
from doltpy.cli.dolt import Dolt

logger = logging.getLogger(__name__)


class DoltClient:
    """Wrapper around a local Dolt clone of the options repository."""

    def __init__(self, repo: str, clone_dir: str | Path) -> None:
        """
        Args:
            repo: DoltHub repository path, e.g. "post-no-preference/options".
            clone_dir: Local directory to clone into (or where an existing clone lives).
        """
        self.repo = repo
        self.clone_dir = Path(clone_dir)
        self._dolt: Optional[Dolt] = None
        self._commit_hash: Optional[str] = None

    # ──────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────

    def connect(self) -> None:
        """Clone the repo if it doesn't exist locally, otherwise open the existing clone."""
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
        """Pull latest changes from DoltHub remote."""
        self._ensure_connected()
        old_hash = self._commit_hash
        self._dolt.pull("origin")
        self._commit_hash = self._dolt.head
        if self._commit_hash != old_hash:
            logger.info("Pulled new data: %s → %s", old_hash[:12], self._commit_hash[:12])
        else:
            logger.info("Already up to date at %s", self._commit_hash[:12])

    # ──────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────

    def query_option_chain(
        self,
        symbols: list[str],
        start: str,
        end: str,
        batch_size: int = 50_000,
    ) -> pd.DataFrame:
        """
        Query the option_chain table for the given symbols and date range.

        DoltHub schema (all from decimal types, returned as strings by doltpy):
            date          DATE         (PK)
            act_symbol    VARCHAR(64)  (PK)
            expiration    DATE         (PK)
            strike        DECIMAL(7,2) (PK)
            call_put      VARCHAR(64)  (PK)
            bid           DECIMAL(7,2)
            ask           DECIMAL(7,2)
            vol           DECIMAL(5,4)  ← this is implied volatility
            delta         DECIMAL(5,4)
            gamma         DECIMAL(5,4)
            theta         DECIMAL(5,4)
            vega          DECIMAL(5,4)
            rho           DECIMAL(5,4)

        Args:
            symbols: List of act_symbol values, e.g. ["SPY", "AAPL"].
            start: Start date inclusive, ISO format "YYYY-MM-DD".
            end: End date inclusive, ISO format "YYYY-MM-DD".
            batch_size: Max rows per query (pagination for large pulls).

        Returns:
            DataFrame with all columns, types cast from strings to proper dtypes.
        """
        self._ensure_connected()

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            logger.info("Querying option_chain for %s [%s → %s]", symbol, start, end)
            offset = 0
            while True:
                query = (
                    f"SELECT date, act_symbol, expiration, strike, call_put, "
                    f"       bid, ask, vol, delta, gamma, theta, vega, rho "
                    f"FROM option_chain "
                    f"WHERE act_symbol = '{symbol}' "
                    f"  AND date >= '{start}' "
                    f"  AND date <= '{end}' "
                    f"ORDER BY date, expiration, strike, call_put "
                    f"LIMIT {batch_size} OFFSET {offset}"
                )
                rows = self._dolt.sql(query, result_format="csv")

                if not rows:
                    break

                df_batch = pd.DataFrame(rows)
                frames.append(df_batch)
                logger.debug("  %s: fetched %d rows (offset=%d)", symbol, len(df_batch), offset)

                if len(rows) < batch_size:
                    break
                offset += batch_size

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

        DoltHub schema:
            date                DATE         (PK)
            act_symbol          VARCHAR(64)  (PK)
            hv_current          DECIMAL(5,4)
            hv_week_ago         DECIMAL(5,4)
            hv_month_ago        DECIMAL(5,4)
            hv_year_high        DECIMAL(5,4)
            hv_year_high_date   DATE
            hv_year_low         DECIMAL(5,4)
            hv_year_low_date    DATE
            iv_current          DECIMAL(5,4)
            iv_week_ago         DECIMAL(5,4)
            iv_month_ago        DECIMAL(5,4)
            iv_year_high        DECIMAL(5,4)
            iv_year_high_date   DATE
            iv_year_low         DECIMAL(5,4)
            iv_year_low_date    DATE

        Args:
            symbols: List of act_symbol values.
            start: Start date inclusive, ISO format.
            end: End date inclusive, ISO format.
            batch_size: Max rows per query.

        Returns:
            DataFrame with proper dtypes.
        """
        self._ensure_connected()

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            logger.info("Querying volatility_history for %s [%s → %s]", symbol, start, end)
            offset = 0
            while True:
                query = (
                    f"SELECT * FROM volatility_history "
                    f"WHERE act_symbol = '{symbol}' "
                    f"  AND date >= '{start}' "
                    f"  AND date <= '{end}' "
                    f"ORDER BY date "
                    f"LIMIT {batch_size} OFFSET {offset}"
                )
                rows = self._dolt.sql(query, result_format="csv")

                if not rows:
                    break

                df_batch = pd.DataFrame(rows)
                frames.append(df_batch)
                logger.debug("  %s: fetched %d rows (offset=%d)", symbol, len(df_batch), offset)

                if len(rows) < batch_size:
                    break
                offset += batch_size

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

    # ──────────────────────────────────────────────
    # Provenance
    # ──────────────────────────────────────────────

    def get_provenance(self) -> dict:
        """
        Return a metadata dict for reproducibility. Save this alongside
        your raw data so anyone can reconstruct the exact same extraction.
        """
        self._ensure_connected()
        return {
            "dolthub_repo": self.repo,
            "dolt_commit_hash": self._commit_hash,
            "dolt_branch": self._dolt.active_branch,
            "clone_dir": str(self.clone_dir),
            "extracted_at": datetime.utcnow().isoformat() + "Z",
        }

    def save_provenance(self, path: str | Path) -> None:
        """Write provenance metadata to a JSON file."""
        meta = self.get_provenance()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Provenance saved to %s", path)

    # ──────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────

    @property
    def commit_hash(self) -> str:
        """Current Dolt HEAD commit hash."""
        self._ensure_connected()
        return self._commit_hash

    @property
    def is_connected(self) -> bool:
        return self._dolt is not None

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._dolt is None:
            raise RuntimeError("DoltClient is not connected. Call .connect() first.")

    @staticmethod
    def _cast_option_chain_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        doltpy returns all values as strings via csv.DictReader.
        Cast to proper Python/pandas types.
        """
        if df.empty:
            return df

        # Dates
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date

        # Numerics — decimal columns come back as strings like "0.2500"
        numeric_cols = ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # call_put stays as string
        df["act_symbol"] = df["act_symbol"].astype(str).str.strip()
        df["call_put"] = df["call_put"].astype(str).str.strip()

        return df

    @staticmethod
    def _cast_vol_history_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Cast volatility_history string values to proper types."""
        if df.empty:
            return df

        # Dates
        date_cols = ["date", "hv_year_high_date", "hv_year_low_date",
                     "iv_year_high_date", "iv_year_low_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        # Numerics
        numeric_cols = [
            "hv_current", "hv_week_ago", "hv_month_ago", "hv_year_high", "hv_year_low",
            "iv_current", "iv_week_ago", "iv_month_ago", "iv_year_high", "iv_year_low",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["act_symbol"] = df["act_symbol"].astype(str).str.strip()

        return df

    def __repr__(self) -> str:
        status = f"commit={self._commit_hash[:12]}" if self._commit_hash else "not connected"
        return f"DoltClient(repo='{self.repo}', {status})"
