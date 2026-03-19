import yaml, os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    """
    Load and merge all yaml configs. Returns a single nested dict.
    """
    load_dotenv()

    with open("../../configs/default.yaml") as f:
        default = yaml.safe_load(f)

    with open("../../configs/data.yaml") as f:
        data = yaml.safe_load(f)

    with open("../../configs/model.yaml") as f:
        model = yaml.safe_load(f)

    with open("../../configs/training.yaml") as f:
        training = yaml.safe_load(f)
        
    with open("../../configs/backtest.yaml") as f:
        backtest = yaml.safe_load(f)

    with open("../../configs/symbols.yaml") as f:
        symbols = yaml.safe_load(f)

    #merge into one dict
    config ={
        **default,
        "data" : data,
        "model": model,
        "training": training,
        "backtest": backtest,
        "symbols":symbols,
    }

    #resolve active_symbols from the pointer
    config['active_symbols_list'] = symbols[default['active_symbols']]

    return config

def get_project_root():
    """Walk up from this file until we find pyproject.toml"""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current  = current.parent
    raise FileNotFoundError