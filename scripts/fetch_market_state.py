import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import yaml, sys, os
from volregime.data.market_state import fetch_market_state

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(project_root) 

load_dotenv()

with open('../configs/default.yaml','r') as f:
    default = yaml.safe_load(f)

with open('../configs/data.yaml','r') as f:
    data = yaml.safe_load(f)

start = data['date_range']['start']
end = data['date_range']['end']
raw_dir = Path(default['paths']['raw_dir'])

# pad start for indicator warmup
padded_start = (pd.Timestamp(start) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")

print(f"Fetching market state, padded range {padded_start} -> {end}")

market_df = fetch_market_state(padded_start, end, data['market_state'])

raw_dir.mkdir(parents=True, exist_ok=True)
market_df.to_parquet(raw_dir / 'market_state.parquet', index=False)

metadata = {
    "sources_enabled": [k for k in data['market_state'] if data['market_state'][k]],
    "padded_start": padded_start,
    "end": end,
    "num_rows": len(market_df),
    "timestamp": time.time()
}
with open(raw_dir/"fetch_market_meta.json","w") as f:
    json.dump(metadata,f)