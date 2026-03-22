import pandas as pd
import numpy as np 
from pathlib import Path
import yaml
import sys, os
import logging
import json
import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(project_root) 

from volregime.data.cleaning import (standardize_call_put, rename_iv_column,
                                     filter_quality, filter_moneyness, filter_maturity, detect_obs_frequency)
from volregime.data.vol_history import compute_vol_history_features
from volregime.data.underlying import compute_log_returns
from volregime.utils.io import save_parquet, save_json
from volregime.utils.config import load_config

config = load_config()

raw_dir = Path(config['default']['paths']['raw_dir'])
canonical_dir = Path(config['default']['paths']['canonical_dir'])
canonical_dir.mkdir(parents=True, exist_ok=True)

# load raw data
option_chain = pd.read_parquet(raw_dir / 'option_chain.parquet')
vol_history_raw = pd.read_parquet(raw_dir / 'volatility_history.parquet')
underlying_all = pd.read_parquet(raw_dir / 'underlying_all.parquet')
market_state = pd.read_parquet(raw_dir / 'market_state.parquet')

print(f'Raw: option_chain={len(option_chain)}, vol_history={len(vol_history_raw)}, underlying={len(underlying_all)}')

# step 1: clean option chain
option_chain = standardize_call_put(option_chain)
option_chain = rename_iv_column(option_chain)
option_chain = filter_quality(option_chain, config['data']['filters'])

# step 2: add log returns to underlying
underlying_frames = []
for symbol in underlying_all['symbol'].unique():
    sym_df = underlying_all[underlying_all['symbol'] == symbol].sort_values('date').copy()
    sym_df = compute_log_returns(sym_df)
    underlying_frames.append(sym_df)
underlying_all = pd.concat(underlying_frames, ignore_index=True)

# step 3: join option_chain to underlying
# need spot price for moneyness
# underlying has 'symbol' column, option_chain has 'act_symbol'
# use symbol_map to align them 
# create a lookup: (date, symbol) -> adj_close
underlying_lookup = underlying_all.set_index(['date','symbol'])['adj_close'].to_dict()

spot_prices = []
for _, row in option_chain.iterrows():
    key = (row['date'], row['act_symbol']) # assumes act_symbol == yf ticker for pilot
    spot_prices.append(underlying_lookup.get(key, np.nan))

option_chain['spot'] = spot_prices

# drop rows with no spot prices (underlying not found for that date)
n_before = len(option_chain)
option_chain = option_chain.dropna(subset=['spot'])
print(f"After underlying join: {n_before} -> {len(option_chain)} (dropped {n_before - len(option_chain)} with no spot)")

# step 4: compute moneyness and tau
option_chain['moneyness'] = option_chain['strike'] / option_chain['spot']
option_chain['log_moneyness'] = np.log(option_chain['moneyness'])
option_chain['tau'] = (pd.to_datetime(option_chain['expiration']) - pd.to_datetime(option_chain['date'])).dt.days / 365

# step 5: apply moneyess + maturity filters
option_chain = filter_moneyness(option_chain, config['data']['moneyness']['min'], config['data']['moneyness']['max'])
option_chain = filter_maturity(option_chain, config['data']['maturity']['min_days'], config['data']['maturity']['max_days'])

# step 6: observation frequency + gao detection
for symbol in option_chain['act_symbol'].unique():
    symbol_dates = option_chain[option_chain['act_symbol'] == symbol]['date']
    freq = detect_obs_frequency(symbol_dates)
    option_chain.loc[option_chain['act_symbol'] == symbol, "obs_frequency"] = freq
    print(f"   {symbol}: obs_frequency = {freq}")

# (gap detection deferred to build_surface.py where surfaces are constructed date-by-date)

# step 7: compute vol_history features
vol_history_features = compute_vol_history_features(vol_history_raw, config['data']['vol_history']['features'])

# step 8: save canonical tables
save_parquet(option_chain, canonical_dir / "options_canonical.parquet")
save_parquet(underlying_all, canonical_dir / "underlying_canonical.parquet")
save_parquet(market_state, canonical_dir / "market_state_canonical.parquet")
save_parquet(vol_history_features, canonical_dir / "vol_history_canonical.parquet")

save_json({
    "rows": {"options": len(option_chain), "underlying":len(underlying_all), "market_state": len(market_state), "vol_history": len(vol_history_features)},
    "symbols": list(option_chain['act_symbol'].unique()),
    "date_range": [str(option_chain['date'].min()), str(option_chain['date'].max())],
    "timestamp": datetime.datetime.utcnow().isoformat()
}, canonical_dir / "build_canonical_meta.json")

print("Canonical build complete.")