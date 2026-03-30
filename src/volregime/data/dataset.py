import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np 
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SurfaceAlphaDataset(Dataset):
    """
    PyTorch Dataset that loads pre-built tensors from data/processed/.

    Reads from the master sample_index.parquet to find paths.
    """

    def __init__(self, sample_index_path, processed_dir):
        """
        Args:
            sample_index_path: path to sample_index.parquet
            processed_dir: root of data/processed/
        """
        self.processed_dir = Path(processed_dir)
        self.index = pd.read_parquet(sample_index_path)
        logger.info("Dataset loaded: %d samples", len(self.index))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]

        surface = torch.load(self.processed_dir / row['surface_path'], weights_only=True)
        returns = torch.load(self.processed_dir / row['returns_path'], weights_only=True)
        vol_history = torch.load(self.processed_dir / row['vh_path'], weights_only=True)
        market_state = torch.load(self.processed_dir / row['mkt_path'], weights_only=True)
        targets = torch.load(self.processed_dir / row['target_path'], weights_only=True)

        return {
            "surface": surface,             # (6, H, W) float32
            "returns": returns,             # (L, F_ret) float32
            "vol_history": vol_history,     # (F_vh,) float32
            "market_state": market_state,   # (F_mkt,) float32
            "target_rv": targets[0],        # scalar: log forward RV
            "target_tail": targets[1],      # scalar: 0 or 1
            "target_regime": targets[2],    # scalar: integer 0-5
            "meta": {
                "date": row["date"],
                "symbol": row["symbol"],
                "obs_frequency": row.get("obs_frequency", "daily"),
                "is_gap_filled": row.get("is_gap_filled", False),
            }
        }

    def get_subset(self, start_date, end_date):
        """Return a new dataset filtered to a date range (for walk-forward splits)."""
        mask = (self.index['date'] >= str(start_date)) & (self.index['date'] <= str(end_date))
        subset = SurfaceAlphaDataset.__new__(SurfaceAlphaDataset)
        subset.processed_dir = self.processed_dir
        subset.index = self.index[mask].reset_index(drop=True)
        return subset

    def get_dates(self):
        """Return sorted unique dates."""
        return sorted(self.index['date'].unique())

    def get_symbols(self):
        """Return unique symbols."""
        return list(self.index['symbol'].unique())