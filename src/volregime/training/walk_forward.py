"""
Walk-forward cross-validation orchestrator.

Computes fold boundaries from trading-day arrays (not calendar days), then for
each fold:
    1. Slices train/val/test subsets from the master SurfaceAlphaDataset
    2. Creates DataLoaders
    3. Instantiates a fresh SurfaceAlphaModel
    4. Trains with Trainer (early stopping, checkpointing)
    5. Runs inference on the test subset and saves per-fold predictions
    6. Optionally fits and evaluates all baselines on the same fold

Fold structure (expanding mode, window sizes in trading days):
    [train_window][embargo][val_window][test_window]
    Each fold advances the test window by step_days.

Config keys (cfg["training"]["walk_forward"]):
    num_folds, train_window_days, val_window_days, test_window_days,
    mode ("expanding"|"rolling"), embargo_days, step_days
"""

from torch.utils.data._utils.pin_memory import pin_memory
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import DataLoader

from ..models.full_model import SurfaceAlphaModel
from ..utils.io import save_json
from .losses import SurfaceAlphaLoss
from .trainer import Trainer

logger = logging.getLogger(__name__)

import wandb

@dataclass
class FoldSpec:
    """Date boundries for one walk-forward fold"""
    fold_idx:      int
    train_start:   str   # "YYYY-MM-DD"
    train_end:     str
    embargo_start: str
    embargo_end:   str
    val_start:     str
    val_end:       str
    test_start:    str
    test_end:      str

class WalkForwardOrchestrator:
    def __init__(self, cfg: dict, dataset, output_dir: str | None = None):
        """
        Args:
            cfg: full merged config dict from load_config()
            dataset: SurfaceAlphaDataset over the full date range
            output_dir: project root directory
        """
        self.cfg = cfg
        self.dataset = dataset
        self.cfg_train = cfg.get('training',{})
        self.root = Path(output_dir) if output_dir else Path('.')
        (self.root / "outputs" / "predictions").mkdir(parents=True, exist_ok=True)
        (self.root / "outputs" / "checkpoints").mkdir(parents=True, exist_ok=True)

    def compute_fold_specs(self) -> list[FoldSpec]:
        """
        Compute fold date boundaries from the dataset's available trading dates.

        Returns:
            list of FoldSpec, one per fold (may be fewer than num_folds if
            the dataset does not cover the full time range).
        """
        dates = self.dataset.get_dates()
        n = len(dates)

        wf = self.cfg_train.get('walk_forward', {})
        train_w = int(wf.get('train_window_days', 756))
        val_w = int(wf.get('val_window_days', 63))
        test_w = int(wf.get('test_window_days', 63))
        embargo = int(wf.get('embargo_days', 5))
        step = int(wf.get('step_days', 63))
        num_folds = int(wf.get('num_folds', 8))
        mode = wf.get('mode', 'expanding')

        base = train_w + embargo + val_w

        specs = []
        for k in range(num_folds):
            test_start_idx = base + k * step
            test_end_idx = test_start_idx + test_w -1

            if test_end_idx >= n:
                logger.warning(
                    'Fold %d: test end index %d exceeds dataset length %d - skipping.', k, test_end_idx, n
                )
                break

            val_end_idx = test_start_idx - 1
            val_start_idx = val_end_idx - val_w + 1
            embargo_end_idx = val_start_idx - 1
            emb_start_idx = embargo_end_idx - embargo + 1
            train_end_idx = emb_start_idx - 1

            if mode == 'expanding':
                train_start_idx = 0
            else: # rolling
                train_start_idx = max(0, train_end_idx - train_w + 1)

            specs.append(FoldSpec(
                fold_idx= k,
                train_start= str(dates[train_start_idx]),
                train_end= str(dates[train_end_idx]),
                embargo_start= str(dates[emb_start_idx]),
                embargo_end= str(dates[embargo_end_idx]),
                val_start= str(dates[val_start_idx]),
                val_end = str(dates[val_end_idx]),
                test_start= str(dates[test_start_idx]),
                test_end= str(dates[test_end_idx])
            ))  
        return specs

    def _make_loaders(self, train_ds, val_ds, test_ds) -> tuple:
        dl_cfg = self.cfg_train.get('dataloader', {})
        kwargs = dict(
            batch_size = int(self.cfg_train.get('batch_size', 64)),
            num_workers = int(dl_cfg.get('num_workers',4)),
            pin_memory = bool(dl_cfg.get('pin_memory', True)) 
        )
        pf = dl_cfg.get('prefetch_factor', 2)
        if kwargs['num_workers'] > 0:
            kwargs['prefetch_factor'] = int(pf)

        train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
        test_loader = DataLoader(test_ds, shuffle = False, **kwargs)
        return train_loader, val_loader, test_loader

    @torch.no_grad()
    def _collect_predictions(self, model, loader, device) -> pd.DataFrame:
        """Run model on loader and collect per-sample predictions."""
        model.eval()
        records = []
        for batch in loader:
            surface = batch['surface'].to(device)
            returns = batch['returns'].to(device)
            vol_history = batch['vol_history'].to(device)
            market_state = batch['market_state'].to(device)

            outputs = model(surface, returns, vol_history, market_state)

            rv_forecast = outputs['rv_forecast'].cpu().numpy()
            tail_prob = outputs['tail_prob'].cpu().numpy()
            regime_probs = outputs['regime_probs'].cpu().numpy()
            expert_preds = outputs['expert_preds'].cpu().numpy()

            target_rv = batch['target_rv'].numpy()
            target_tail = batch['target_tail'].numpy()
            target_regime = batch['target_regime'].numpy()
            dates = batch['meta']['date']
            symbols = batch['meta']['symbol']

            for i in range(len(rv_forecast)):
                row = {
                    'date': dates[i],
                    'symbol': symbols[i],
                    'rv_pred': float(rv_forecast[i]),
                    'tail_prob': float(tail_prob[i]),
                    'regime_pred': int(regime_probs[i].argmax()),
                    'target_rv': float(target_rv[i]),
                    'target_tail': int(target_tail[i]),
                    'target_regime': int(target_regime[i]),
                }
                for k in range(regime_probs.shape[1]):
                    row[f'p_regime_{k}'] = float(regime_probs[i,k])
                for k in range(expert_preds.shape[1]):
                    row[f'expert_{k}'] = float(expert_preds[i,k])
                records.append(row)

        return pd.DataFrame(records)

    def run(self, start_fold: int = 0, end_fold: int | None = None) -> list[dict]:
        """
        Execute walk-forward folds. Returns list of per-fold result dicts.

        Args:
            start_fold: first fold index to run (inclusive). Use to resume after failure.
            end_fold: last fold index to run (exclusive). None means run all remaining folds.

        Each result dict:
            {fold_idx, train_start, test_end, best_val_loss,
             n_test_samples, predictions_path}
        """
        fold_specs = self.compute_fold_specs()
        fold_specs = [s for s in fold_specs if s.fold_idx >= start_fold]
        if end_fold is not None:
            fold_specs = [s for s in fold_specs if s.fold_idx < end_fold]
        logger.info("Walk-forward: running folds %d-%s (%d total)",
                    start_fold, end_fold - 1 if end_fold is not None else "end", len(fold_specs))

        wandb.init(
            project=self.cfg.get('wandb', {}).get('project','surfacealpha'),
            config = self.cfg,
            reinit=True
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_results = []

        for spec in fold_specs:
            k = spec.fold_idx
            logger.info(
                'Fold %d | train %s -> %s | val %s -> %s | test %s -> %s',
                k, spec.train_start, spec.train_end,
                spec.val_start, spec.val_end,
                spec.test_start, spec.test_end
            )

            # slice dataset
            train_ds = self.dataset.get_subset(spec.train_start, spec.train_end)
            val_ds = self.dataset.get_subset(spec.val_start, spec.val_end)
            test_ds = self.dataset.get_subset(spec.test_start, spec.test_end)
            logger.info(
                "Fold %d | sizes - train=%d val=%d test=%d",
                k, len(train_ds), len(val_ds), len(test_ds),
            )

            # dataloaders 
            train_loader, val_loader, test_loader = self._make_loaders(train_ds, val_ds, test_ds)

            # fresh model (new weights for each fold)
            model = SurfaceAlphaModel(self.cfg)
            logger.info('Fold %d | params: %s', k , model.count_parameters())

            # train
            trainer = Trainer(
                cfg = self.cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                fold_idx=k,
                output_dir=str(self.root),
                model_type='full'
            )
            best_val_loss = trainer.fit()

            # collect test-set predictions
            preds_df = self._collect_predictions(model, test_loader, device)
            preds_path = (
                self.root / 'outputs' / 'predictions' / f'fodl_{k}_test_preds.parquet'
            )
            preds_df.to_parquet(preds_path, index=False)
            logger.info("Fold %d | saved %d predictions -> %s", k, len(preds_df), preds_path)

            result ={
                'fold_idx': k,
                'train_start': spec.train_start,
                'train_end': spec.train_end,
                'test_start': spec.test_start,
                'test_end': spec.test_end,
                'best_val_loss': best_val_loss,
                'n_test_samples': len(preds_df),
                'predictions_path': str(preds_path)
            }
            all_results.append(result)

            # save fold summary json
            summary_path = self.root / 'outputs' / f'fold_{k}_summary.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

        if wandb.run is not None:
            wandb.finish()
        
        logger.info('Walk forward complete. %d folds run.', len(all_results))
        return all_results

