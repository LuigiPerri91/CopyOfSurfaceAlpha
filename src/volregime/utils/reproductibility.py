import random, subprocess, json
import numpy as np
import torch

def set_seed(seed=42):
    """set all random seeds for reproductibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

def get_git_hash():
    """return current git commit hash, or 'unknown' if not in a repo"""
    try:
        result = subprocess.run(['git','rev-parse','HEAD'],
        capture_output=True, text=True, cwd=project_root)
        return result.stdout.strip()
    except:
        return "unknown"

def record_provenance(config, dolt_commit_hash=None):
    """build a full provenance dict for experiment tracking"""
    return {
        'git_hash': get_git_hash(),
        'dolt_commit_hash': dolt_commit_hash,
        'seed': config.get('seed'),
        'torch_version' : torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'config_snapshot': config,
    }