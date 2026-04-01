import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import yaml, sys, os, time, json
from volregime.data.market_state import fetch_market_state
from volregime.utils.config import load_config

load_dotenv()

cfg = load_config()
start = cfg['data']['date_range']['start']
end = cfg['data']['date_range']['end']
market_state_cfg = cfg['data']['market_state']
raw_dir = Path(cfg['paths']['raw_dir'])

# pad start for indicator warmup
padded_start = (pd.Timestamp(start) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")

market_state_path = raw_dir / "market_state.parquet"
if market_state_path.exists():
    print("market_state.parquet already exists — skipping fetch. Run 'make clean-data' to re-fetch.")
    import sys; sys.exit(0)

print(f"Fetching market state, padded range {padded_start} -> {end}")

market_df = fetch_market_state(padded_start, end, market_state_cfg)

raw_dir.mkdir(parents=True, exist_ok=True)
market_df.to_parquet(raw_dir / 'market_state.parquet', index=False)

metadata = {
    "sources_enabled": [k for k in market_state_cfg if market_state_cfg[k]],
    "padded_start": padded_start,
    "end": end,
    "num_rows": len(market_df),
    "timestamp": time.time()
}
with open(raw_dir/"fetch_market_meta.json","w") as f:
    json.dump(metadata,f)