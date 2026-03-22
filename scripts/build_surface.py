import pandas as pd
import numpy as np 
import torch
from pathlib import Path
import yaml, sys, logging, json, datetime, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(project_root) 

from volregime.data.surface_builder import build_surface
from volregime.data.feature_eng import build_returns_tensor, build_vol_history_vector, build_market_state_vector
from volregime.data.targets import compute_forward_rv, compute_tail_indicator, compute_regime_label
from volregime.utils.io import save_tensor, save_json
from volregime.utils.config import load_config

config = load_config()

canonical_dir = Path(config['default']['paths']['canonical_dir'])
processed_dir = Path(config['default']['paths']['processed_dir'])

# load canonical data
options = pd.read_parquet(canonical_dir / "options_canonical.parquet")
underlying = pd.read_parquet(canonical_dir / "underlying_canonical.parquet")
market_state = pd.read_parquet(canonical_dir / "market_state_canonical.parquet")
vol_history = pd.read_parquet(canonical_dir / "vol_history_canonical.parquet")

L = config['data']['returns']['lookback_horizon'] # 60
h = config['data']['targets']['forward_horizon'] # 21

# precompute: merge SPY returns into underlying for beta calculation
spy_returns = market_state[['date','spy_return']].copy() if "spy_return" in market_state.columns else None

# build index for all (date, symbol) pairs
unique_pairs = options[['date','act_symbol']].drop_duplicates().sort_values('date')

sample_index = []
skipped = {"no_returns_history": 0, "no_forward_returns": 0, "no_market_state": 0}

for symbol in unique_pairs['act_symbol'].unique():
    symbol_pairs = unique_pairs[unique_pairs['act_symbol'] == symbol]
    symbol_underlying = underlying[underlying['symbol'] == symbol].sort_values('date')
    symbol_vol_history = vol_history[vol_history['act_symbol'] == symbol]

    # precompute historical RVs for tail indicator (expanding window, no future leakage)
    historical_rvs = []

    for _, pair_row in symbol_pairs.iterrows():
        date = pair_row['date']

        # returns tensor
        underlying_window = symbol_underlying[symbol_underlying['date'] < date].tail(L)
        if len(underlying_window) < L:
            skipped['no_returns_history'] += 1
            continue
        
        # merge SPY returns for data
        if spy_returns is not None:
            underlying_window = underlying_window.merge(spy_returns, on='date', how='left')
            underlying_window['spy_return'] = underlying_window['spy_return'].fillna(0)
        
        returns_tensor = build_returns_tensor(underlying_window, config['data']['returns'])

        # forward returns
        future_underlying = symbol_underlying[symbol_underlying['date'] > date].head(h)
        if len(future_underlying) < h:
            skipped['no_forward_returns'] += 1
            continue

        future_returns = future_underlying['log_return'].values
        rv, log_rv = compute_forward_rv(future_returns, h)

        tail = compute_tail_indicator(
            future_returns, rv,
            method = config['data']['targets']['tail_threshold_method'],
            threshold_value=config['data']['targets']['tail_threshold_value'],
            historical_rvs = np.array(historical_rvs) if historical_rvs else None
        )

        # update expanding RV history (for tail threshold computation)
        historical_rvs.append(rv)

        # regime label
        regime_underlying = symbol_underlying[symbol_underlying['date'] <= date].tail(250)
        mkt_row = market_state[market_state['date'] == date]
        if len(mkt_row) == 0:
            skipped['no_market_state'] += 1
            continue
        mkt_row = mkt_row.iloc[0]

        regime = compute_regime_label(regime_underlying, mkt_row, config['backtest']['regime_identification'])
        
        # surface tensor
        option_rows = options[(options['date'] == date) & (options['act_symbol']==symbol)]
        surface = build_surface(option_rows, config['data']['surface'])

        # vol history features
        vh_row = symbol_vol_history[symbol_vol_history['date'] == date]
        if len(vh_row) >0:
            vh_vector = build_vol_history_vector(vh_row.iloc[0])
        else:
            vh_vector = np.full(11, np.nan, dtype=np.float32)
        
        # market state vector
        mkt_vector = build_market_state_vector(mkt_row, config['data']['market_state'])

        # target tensor
        target_tensor = np.array([log_rv, float(tail), float(regime)], dtype=np.float32)

        # save all tensors
        date_str = str(date)
        base = f"{symbol}/{date_str}"

        save_tensor(torch.tensor(surface), processed_dir / f"surfaces/{base}.pt")
        save_tensor(torch.tensor(returns_tensor), processed_dir / f"returns/{base}.pt")
        save_tensor(torch.tensor(vh_vector), processed_dir / f"vol_history/{base}.pt")
        save_tensor(torch.tensor(mkt_vector), processed_dir / f"market_state/{base}.pt")
        save_tensor(torch.tensor(target_tensor), processed_dir / f"targets/{base}.pt")

        sample_index.append({
            "date": date_str,
            "symbol": symbol,
            "surface_path": f"surfaces/{base}.pt",
            "returns_path": f"returns/{base}.pt",
            "vh_path": f"vol_history/{base}.pt",
            "mkt_path": f"market_state/{base}.pt",
            "target_path": f"targets/{base}.pt",
            "obs_frequency": option_rows["obs_frequency"].iloc[0] if "obs_frequency" in option_rows.columns else "daily",
            "is_gap_filled": False,
        })

# save master index
index_df = pd.DataFrame(sample_index)
index_df.to_parquet(processed_dir / "sample_index.parquet", index=False)

save_json({
    "num_samples": len(sample_index),
    "symbols": list(unique_pairs['act_symbol'].unique()),
    "skipped": skipped,
    "timestamp": datetime.datetime.utcnow().isoformat()
}, processed_dir / "build_surface_meta.json")

print(f"Built {len(sample_index)} samples")
print(f"Skipped: {skipped}")