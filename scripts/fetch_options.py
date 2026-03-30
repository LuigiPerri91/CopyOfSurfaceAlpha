import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import yaml
from volregime.data.dolt_client import DoltClient
from volregime.utils.config import load_config

load_dotenv()

cfg = load_config()

active_symbols = cfg['active_symbols_list']
start  = cfg['data']['date_range']['start']
end    = cfg['data']['date_range']['end']
repo_link = cfg['data']['dolthub']['repo']

raw_dir = Path(cfg['paths']['raw_dir'])
clone_dir = raw_dir / 'dolt_clone' 

oc_path = raw_dir / 'option_chain.parquet'
vh_path = raw_dir / 'volatility_history.parquet'

if oc_path.exists() and vh_path.exists():
    print("Using cached Dolt data. Skipping network pull.")
    options_chain_df = pd.read_parquet(oc_path)
    vol_history_df = pd.read_parquet(vh_path)
    print(options_chain_df.head())
    print(vol_history_df.head())
else:
    client = DoltClient(repo=repo_link, clone_dir=clone_dir)
    client.connect()
    client.pull()

    options_chain_df = client.query_option_chain(symbols=active_symbols, start=start, end=end)
    vol_history_df = client.query_vol_history(symbols=active_symbols,start=start, end=end)

    raw_dir.mkdir(parents=True, exist_ok=True)
    options_chain_df.to_parquet(oc_path)
    vol_history_df.to_parquet(vh_path)

    client.save_provenance(raw_dir / 'fetch_options_meta.json')

    print(options_chain_df.head())
    print(vol_history_df.head())
    print(client.commit_hash[:12])