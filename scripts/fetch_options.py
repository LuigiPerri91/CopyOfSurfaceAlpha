"""
Fetch options chain + volatility history data.

Routing:
  - 2008-01-02 → 2025-12-16:  philippdubach/options-data (no API key, pre-computed Greeks)
  - 2025-12-17 → present:     DoltHub (live/recent data)

Caching:
  - DubachClient caches raw per-symbol parquets + computed vol_history in dubach_cache/
  - DoltHub query results are cached in dolt_cache/ to avoid re-querying on every run
  - Final option_chain.parquet and volatility_history.parquet are checked independently
    so a missing vol_history doesn't force a full option chain re-fetch

Usage:
    python scripts/fetch_options.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volregime.data.dolt_client import DoltClient
from volregime.data.dubach_client import DubachClient
from volregime.utils.config import load_config

load_dotenv()

# philippdubach/options-data covers up to this date (inclusive)
DUBACH_END = "2025-12-16"

cfg = load_config()
active_symbols = cfg["active_symbols_list"]
start = cfg["data"]["date_range"]["start"]
end = cfg["data"]["date_range"]["end"]
repo_link = cfg["data"]["dolthub"]["repo"]

raw_dir = Path(cfg["paths"]["raw_dir"])
raw_dir.mkdir(parents=True, exist_ok=True)

dolt_cache_dir = raw_dir / "dolt_cache"
dolt_cache_dir.mkdir(exist_ok=True)

oc_path = raw_dir / "option_chain.parquet"
vh_path = raw_dir / "volatility_history.parquet"

need_oc = not oc_path.exists()
need_vh = not vh_path.exists()

if not need_oc and not need_vh:
    print("Both option_chain.parquet and volatility_history.parquet already exist — skipping fetch.")
    print("Run 'make clean-data' to force a full re-fetch.")
    sys.exit(0)

oc_frames = []
vh_frames = []

# philippdubach segment (2008-01-02 -> 2025-12-16)
dubach_end = min(pd.Timestamp(DUBACH_END), pd.Timestamp(end)).strftime("%Y-%m-%d")
if start <= dubach_end:
    print(f"Fetching philippdubach options: {start} → {dubach_end}")
    dubach = DubachClient(cache_dir=raw_dir / "dubach_cache")

    if need_oc:
        oc_dubach = dubach.query_option_chain(active_symbols, start, dubach_end)
        if not oc_dubach.empty:
            oc_frames.append(oc_dubach)

    if need_vh:
        vh_dubach = dubach.query_vol_history(active_symbols, start, dubach_end)
        if not vh_dubach.empty:
            vh_frames.append(vh_dubach)

    dubach.save_provenance(raw_dir / "fetch_dubach_meta.json")

# DoltHub segment (2025-12-17 -> present)
dolt_start = (pd.Timestamp(DUBACH_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
if end >= dolt_start:
    dolt_oc_cache = dolt_cache_dir / f"oc_{dolt_start}_{end}.parquet"
    dolt_vh_cache = dolt_cache_dir / f"vh_{dolt_start}_{end}.parquet"

    if need_oc and not dolt_oc_cache.exists() or need_vh and not dolt_vh_cache.exists():
        print(f"Fetching DoltHub options via API: {dolt_start} → {end}")
        client = DoltClient(repo=repo_link, access_method="api")

        if need_oc and not dolt_oc_cache.exists():
            oc_dolt = client.query_option_chain(active_symbols, start=dolt_start, end=end)
            if not oc_dolt.empty:
                oc_dolt.to_parquet(dolt_oc_cache, index=False)
        if need_vh and not dolt_vh_cache.exists():
            vh_dolt = client.query_vol_history(active_symbols, start=dolt_start, end=end)
            if not vh_dolt.empty:
                vh_dolt.to_parquet(dolt_vh_cache, index=False)

        client.save_provenance(raw_dir / "fetch_options_meta.json")
    else:
        print(f"DoltHub cache hit: {dolt_start} → {end}")

    if need_oc and dolt_oc_cache.exists():
        oc_frames.append(pd.read_parquet(dolt_oc_cache))
    if need_vh and dolt_vh_cache.exists():
        vh_frames.append(pd.read_parquet(dolt_vh_cache))

# Save outputs
if need_oc:
    if not oc_frames:
        print("ERROR: No option chain data fetched.")
        sys.exit(1)
    options_chain_df = pd.concat(oc_frames, ignore_index=True).sort_values(
        ["date", "act_symbol", "expiration", "strike", "call_put"]
    ).reset_index(drop=True)
    options_chain_df.to_parquet(oc_path)
    print(f"Saved {len(options_chain_df)} option chain rows -> {oc_path}")
    print(options_chain_df.head())

if need_vh and vh_frames:
    vol_history_df = pd.concat(vh_frames, ignore_index=True).sort_values(
        ["date", "act_symbol"]
    ).reset_index(drop=True)
    vol_history_df.to_parquet(vh_path)
    print(f"Saved {len(vol_history_df)} vol history rows -> {vh_path}")