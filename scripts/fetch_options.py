import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import yaml
from ..src.volregime.data.dolt_client import DoltClient

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
repo_link = data['dolthub']['repo']
raw_dir = Path(default['paths']['raw_dir'])
clone_dir = raw_dir / 'dolt_clone' 

client = DoltClient(repo=repo_link, clone_dir=clone_dir)
client.connect()
client.pull()

options_chain_df = client.query_option_chain(symbols=active_symbols, start=start, end= end)
vol_history_df = client.query_vol_history(symbols=active_symbols,start=start, end=end)

raw_dir.mkdir(parents=True, exist_ok=True)
options_chain_df.to_parquet(raw_dir / 'option_chain.parquet')
vol_history_df.to_parquet(raw_dir / 'volatility_history.parquet')

client.save_provenance(raw_dir / 'fetch_options_meta.json')

print(options_chain_df.head())
print(vol_history_df.head())
print(client.commit_hash()[:12])