import yaml
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf

def load_config():
    """
    Load and merge all yaml configs. Returns a single nested dict.
    """
    load_dotenv()
    root = get_project_root()
    cfg_dir = root / 'configs'

    default = OmegaConf.load(cfg_dir / "default.yaml")
    data = OmegaConf.load(cfg_dir / "data.yaml")
    model = OmegaConf.load(cfg_dir / "model.yaml")
    training = OmegaConf.load(cfg_dir / "training.yaml")
    backtest = OmegaConf.load(cfg_dir / "backtest.yaml")
    symbols = OmegaConf.load(cfg_dir / "symbols.yaml")

    config = OmegaConf.create({
        **OmegaConf.to_container(default),
        "data" : OmegaConf.to_container(data),
        "model": OmegaConf.to_container(model),
        "training": OmegaConf.to_container(training),
        "backtest": OmegaConf.to_container(backtest),
        "symbols": OmegaConf.to_container(symbols),
    })

    OmegaConf.resolve(config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    #resolve active_symbols from the pointer
    config_dict['active_symbols_list'] = config_dict['symbols'][config_dict['active_symbols']]

    return config_dict

def get_project_root():
    """Walk up from this file until we find pyproject.toml"""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current  = current.parent
    raise FileNotFoundError