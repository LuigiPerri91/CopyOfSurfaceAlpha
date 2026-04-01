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
    All tensors are loaded into memory on first access via get_subset(),
    so DataLoader workers see no file I/O in the hot path.
    """

    def __init__(self, sample_index_path, processed_dir):
        """
        Args:
            sample_index_path: path to sample_index.parquet
            processed_dir: root of data/processed/
        """
        self.processed_dir = Path(processed_dir)
        self.index = pd.read_parquet(sample_index_path)
        self._cache = None  # populated lazily by _load_cache()
        logger.info("Dataset loaded: %d samples", len(self.index))

    def _load_cache(self):
        """Load all tensors into memory. Called once per subset."""
        if self._cache is not None:
            return
        logger.info("Caching %d samples into memory...", len(self.index))
        cache = []
        for _, row in self.index.iterrows():
            cache.append({
                "surface": torch.load(self.processed_dir / row['surface_path'], weights_only=True),
                "returns": torch.load(self.processed_dir / row['returns_path'], weights_only=True),
                "vol_history": torch.load(self.processed_dir / row['vh_path'], weights_only=True),
                "market_state": torch.load(self.processed_dir / row['mkt_path'], weights_only=True),
                "targets": torch.load(self.processed_dir / row['target_path'], weights_only=True),
            })
        self._cache = cache
        logger.info("Cache ready.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self._cache is None:
            self._load_cache()
        row = self.index.iloc[idx]
        c = self._cache[idx]
        targets = c["targets"]
        return {
            "surface": c["surface"],
            "returns": c["returns"],
            "vol_history": c["vol_history"],
            "market_state": c["market_state"],
            "target_rv": targets[0],
            "target_tail": targets[1],
            "target_regime": targets[2],
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
        subset._cache = None  # will be populated on first __getitem__
        return subset

    def get_dates(self):
        """Return sorted unique dates."""
        return sorted(self.index['date'].unique())

    def get_symbols(self):
        """Return unique symbols."""
        return list(self.index['symbol'].unique())

    def get_regime_weights(self, n_classes: int = 6) -> "torch.Tensor":
        """Compute inverse-frequency class weights for CrossEntropyLoss.

        Loads regime labels from target tensor files (index[2] of each target tensor).
        Returns a float32 tensor of shape (n_classes,) where weight[c] = N / (n_classes * count[c]).
        Classes absent from the training split get weight 0.0.
        """
        if self._cache is not None:
            counts = np.zeros(n_classes, dtype=np.float32)
            for c in self._cache:
                cls = int(c["targets"][2].item())
                if 0 <= cls < n_classes:
                    counts[cls] += 1
        else:
            counts = np.zeros(n_classes, dtype=np.float32)
            for _, row in self.index.iterrows():
                targets = torch.load(self.processed_dir / row['target_path'], weights_only=True)
                cls = int(targets[2].item())
                if 0 <= cls < n_classes:
                    counts[cls] += 1
        total = counts.sum()
        weights = np.where(counts > 0, total / (n_classes * counts), 0.0)
        return torch.tensor(weights, dtype=torch.float32)
