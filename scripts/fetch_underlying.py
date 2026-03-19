from volregime.data.underlying import fetch_underlying
from volregime.data.symbol_map import SymbolMap
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os, sys, time, yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(project_root) 

load_dotenv()

with open('../configs/default.yaml','r') as f:
    default = yaml.safe_load(f)

with open('../configs/symbols.yaml','r') as f:
    symbols = yaml.safe_load(f)

with open('../configs/data.yaml','r') as f:
    data = yaml.safe_load(f)

active_symbols = symbols[default['active_symbols']]
start  = data['date_range']['start']
end  = data['date_range']['end']
raw_dir = Path(default['paths']['raw_dir'])

#pad start by 300 calendar days for 200 MA + other indicator warmup
padded_start = (pd.Timestamp(start) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")

#resolve act_symbols to yfinance tickers
sym_map = SymbolMap(overrides_from_config=symbols.get("symbol_overrides"),exclude= symbols.get('exclude'))
ticker_mapping , skipped = sym_map.resolve_all(active_symbols)
yf_tickers = list(ticker_mapping.values())

print(f"Fetching underlying for {len(yf_tickers)} tickers, padded range {padded_start} -> {end}")

#download all via yfinance (batched)
cache_dir = raw_dir / 'underlying'
results, failed = fetch_underlying(yf_tickers, padded_start, end, cache_dir=cache_dir)

# save combined file for convenience
if results:
    all_frames = []
    for ticker, df in results.items():
        df_copy = df.copy()
        df_copy['symbol'] = ticker
        all_frames.append(df_copy)
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(raw_dir / "underlying_all.parquet", index=False)

# save metadata
metadata= {
    "tickers_requested": yf_tickers,
    "tickers_succeeded": results.keys(),
    "tickers_failed": failed,
    "padded_start": padded_start,
    "end": end,
    "timestamp": time.time()
}
with open(raw_dir/"fetch_underlying_meta.json", "w") as f:
    json.dump(metadata, f)