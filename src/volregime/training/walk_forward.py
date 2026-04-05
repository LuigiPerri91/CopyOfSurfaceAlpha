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

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np 
import pandas as pd 
import torch

from ..models.full_model import SurfaceAlphaModel
from .losses import SingleTaskLoss
from .trainer import Trainer
from ..utils.io import save_checkpoint, load_checkpoint
from torch.utils.data import TensorDataset, DataLoader as _DL

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

        train_loader = _DL(train_ds, shuffle=True, **kwargs)
        val_loader = _DL(val_ds, shuffle=False, **kwargs)
        test_loader = _DL(test_ds, shuffle = False, **kwargs)
        return train_loader, val_loader, test_loader

    def _collect_baseline_data(self, dataset) -> dict:
        """Extract flat arrays from a dataset for baseline models.

        Returns dict with keys:
            trailing_log_rv  (N,)  — log(rv_21) at last timestep, same space as target_rv
            forward_rv       (N,)  — target_rv (log-space)
            log_returns      (N,)  — last-day log return (for GARCH series)
            returns_tensors  (N, L, F) — full returns feature tensor (for boosting)
            surfaces         (N, C, H, W)
            vol_histories    (N, vh_dim)
            market_states    (N, ms_dim)
            dates, symbols   lists of str
        """
        trailing_log_rv, forward_rv, log_returns_series = [], [], []
        returns_tensors, surfaces, vol_histories, market_states = [], [], [], []
        dates, symbols = [], []

        for i in range(len(dataset)):
            item = dataset[i]
            ret = item['returns'].numpy()  # (L, F): [log_ret, rv5, rv10, rv21, jump, beta]
            rv21 = float(ret[-1, 3])
            trailing_log_rv.append(np.log(max(rv21, 1e-10)))
            forward_rv.append(float(item['target_rv']))
            log_returns_series.append(float(ret[-1, 0]))
            returns_tensors.append(ret)
            surfaces.append(item['surface'].numpy())
            vol_histories.append(item['vol_history'].numpy())
            market_states.append(item['market_state'].numpy())
            dates.append(item['meta']['date'])
            symbols.append(item['meta']['symbol'])

        return {
            'trailing_log_rv': np.array(trailing_log_rv, dtype=np.float32),
            'forward_rv': np.array(forward_rv, dtype=np.float32),
            'log_returns': np.array(log_returns_series, dtype=np.float32),
            'returns_tensors': np.stack(returns_tensors, axis=0),
            'surfaces': np.stack(surfaces, axis=0),
            'vol_histories': np.stack(vol_histories, axis=0),
            'market_states': np.stack(market_states, axis=0),
            'dates': dates,
            'symbols': symbols,
        }

    def _train_deep_ts(
        self,
        model: "torch.nn.Module",
        name: str,
        train_returns: np.ndarray,
        train_targets: np.ndarray,
        fold_idx: int,
        device: "torch.device",
    ) -> None:
        """Train a deep_ts baseline model in-place using an inline loop.

        Saves best checkpoint to outputs/checkpoints/baselines/fold_{fold_idx}/{name}/best.pt
        and reloads best weights at the end. Does NOT use Trainer to avoid checkpoint
        path collision with the main SurfaceAlphaModel.
        """

        split = int(0.8 * len(train_returns))
        X_tr  = torch.from_numpy(train_returns[:split]).float()
        y_tr  = torch.from_numpy(train_targets[:split]).float()
        X_val = torch.from_numpy(train_returns[split:]).float()
        y_val = torch.from_numpy(train_targets[split:]).float()

        bs = int(self.cfg_train.get('batch_size', 64))
        train_loader = _DL(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
        val_loader = _DL(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)

        loss_fn = SingleTaskLoss(float(self.cfg_train.get('huber_delta', 1.0)))
        opt_cfg = self.cfg_train.get('optimizer', {})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg.get('learning_rate', 3e-4)),
            weight_decay=float(opt_cfg.get('weight_decay', 0.01)),
        )
        model = model.to(device)

        num_epochs = int(self.cfg_train.get('num_epochs', 1000))
        clip_norm  = float(self.cfg_train.get('gradient_clip_norm', 1.0))
        es_cfg = self.cfg_train.get('early_stopping', {})
        patience = int(es_cfg.get('patience', 100))
        min_delta = float(es_cfg.get('min_delta', 1e-4))

        ckpt_dir = (self.root / 'outputs' / 'checkpoints' / 'baselines'
                    / f'fold_{fold_idx}' / name)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / 'best.pt'

        best_val = float('inf')
        patience_ctr = 0

        for epoch in range(num_epochs):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss_fn(model(X_b), y_b).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    val_losses.append(loss_fn(model(X_b.to(device)), y_b.to(device)).item())
            val_loss = float(np.mean(val_losses))

            if val_loss < best_val - min_delta:
                best_val = val_loss
                patience_ctr = 0
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logger.info('Fold %d | %s early stop @ epoch %d (best=%.4f)',
                                fold_idx, name, epoch, best_val)
                    break

        if ckpt_path.exists():
            load_checkpoint(str(ckpt_path), model)

    @torch.no_grad()
    def _predict_deep_ts(
        self,
        model: "torch.nn.Module",
        test_returns: np.ndarray,
        device: "torch.device",
    ) -> np.ndarray:
        """Run a trained deep_ts model on test_returns and return log(RV) predictions."""
        model.eval()
        X = torch.from_numpy(test_returns).float().to(device)
        return model(X).cpu().numpy()

    def _run_baselines(self, train_ds, test_ds, fold_idx: int) -> None:
        """Fit all enabled baselines on train_ds, predict on test_ds, save parquet."""
        cfg_bl = self.cfg_train.get('baselines', {})
        enabled = [k for k in ('persistence', 'har_rv', 'garch', 'boosting', 'deep_ts')
                   if cfg_bl.get(k, False)]
        if not enabled:
            return

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        logger.info('Fold %d | fitting baselines: %s', fold_idx, enabled)
        train = self._collect_baseline_data(train_ds)
        test  = self._collect_baseline_data(test_ds)
        n_test = len(test['trailing_log_rv'])

        records: dict = {
            'date':      test['dates'],
            'symbol':    test['symbols'],
            'target_rv': test['forward_rv'].tolist(),
        }

        # Persistence: log(rv_21) carried forward
        if 'persistence' in enabled:
            from ..baselines.persistence import PersistanceBaseline
            bl = PersistanceBaseline()
            records['persistence'] = bl.predict(test['trailing_log_rv']).tolist()
            logger.info('Fold %d | persistence done', fold_idx)

        # HAR-RV: OLS on [log_rv_1d, log_rv_5d_avg, log_rv_21d_avg] → log(RV)
        if 'har_rv' in enabled:
            from ..baselines.har_rv import HARRVBaseline
            bl = HARRVBaseline()
            bl.fit(train['trailing_log_rv'], train['forward_rv'])
            X_test = bl._build_features(test['trailing_log_rv'])
            preds = bl.reg.predict(X_test).astype(np.float32)
            records['har_rv'] = preds.tolist()
            logger.info('Fold %d | HAR-RV done (train R²=%.3f)', fold_idx, bl.train_r2_ or 0.0)

        # GARCH(1,1): fit on training log-return series, predict h-step-ahead vol
        if 'garch' in enabled:
            from ..baselines.garch import GARCHBaseline
            horizon = (self.cfg.get('data', {})
                       .get('targets', {})
                       .get('forward_horizon', 21))
            bl = GARCHBaseline(horizon=horizon)
            try:
                bl.fit(train['log_returns'])
                forecast_rv = bl.predict()                  # raw RV units
                forecast_log_rv = float(np.log(max(forecast_rv, 1e-10)))
                records['garch'] = [forecast_log_rv] * n_test
                logger.info('Fold %d | GARCH done (log_rv=%.4f)', fold_idx, forecast_log_rv)
            except Exception as exc:
                logger.warning('Fold %d | GARCH failed: %s', fold_idx, exc)
                records['garch'] = [float('nan')] * n_test

        # Boosting: LightGBM on hand-engineered features
        if 'boosting' in enabled:
            from ..baselines.boosting import BoostingBaseline, build_boosting_features
            try:
                train_rows = [
                    build_boosting_features(
                        train['surfaces'][i],
                        train['returns_tensors'][i],
                        train['vol_histories'][i],
                        train['market_states'][i],
                    )
                    for i in range(len(train['forward_rv']))
                ]
                test_rows = [
                    build_boosting_features(
                        test['surfaces'][i],
                        test['returns_tensors'][i],
                        test['vol_histories'][i],
                        test['market_states'][i],
                    )
                    for i in range(n_test)
                ]
                train_feat_df = pd.DataFrame(train_rows)
                test_feat_df  = pd.DataFrame(test_rows)
                bl = BoostingBaseline()
                bl.fit(train_feat_df, train['forward_rv'])
                records['boosting'] = bl.predict(test_feat_df).tolist()
                logger.info('Fold %d | boosting done', fold_idx)
            except Exception as exc:
                logger.warning('Fold %d | boosting failed: %s', fold_idx, exc)
                records['boosting'] = [float('nan')] * n_test

        # deep_ts (LSTM, GRU, TCN) — trained per fold with early stopping
        if 'deep_ts' in enabled:
            from ..baselines.deep_ts import LSTMBaseline, GRUBaseline, TCNBaseline
            input_dim = int(train['returns_tensors'].shape[-1])
            _dt_flags = cfg_bl.get('deep_ts_models', {})
            _all_deep = {
                'deep_ts_lstm': LSTMBaseline,
                'deep_ts_gru': GRUBaseline,
                'deep_ts_tcn': TCNBaseline,
            }
            deep_models = {
                name: cls(input_dim=input_dim)
                for name, cls in _all_deep.items()
                if _dt_flags.get(name.replace('deep_ts_', ''), True)
            }
            for col_name, model in deep_models.items():
                try:
                    self._train_deep_ts(
                        model, col_name,
                        train['returns_tensors'], train['forward_rv'],
                        fold_idx, device,
                    )
                    records[col_name] = self._predict_deep_ts(
                        model, test['returns_tensors'], device,
                    ).tolist()
                    logger.info('Fold %d | %s done', fold_idx, col_name)
                except Exception as exc:
                    logger.warning('Fold %d | %s failed: %s', fold_idx, col_name, exc)
                    records[col_name] = [float('nan')] * n_test

        path = self.root / 'outputs' / 'predictions' / f'fold_{fold_idx}_baseline_preds.parquet'
        pd.DataFrame(records).to_parquet(path, index=False)
        logger.info('Fold %d | saved baseline predictions -> %s', fold_idx, path)

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

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
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

            # class weights for regime head (computed from training split)
            regime_weights = train_ds.get_regime_weights()
            logger.info('Fold %d | regime class weights: %s', k, regime_weights.tolist())

            # train
            trainer = Trainer(
                cfg = self.cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                fold_idx=k,
                output_dir=str(self.root),
                model_type='full',
                regime_class_weights=regime_weights,
            )
            best_val_loss = trainer.fit()

            # collect test-set predictions
            preds_df = self._collect_predictions(model, test_loader, device)
            preds_path = (
                self.root / 'outputs' / 'predictions' / f'fold_{k}_test_preds.parquet'
            )
            preds_df.to_parquet(preds_path, index=False)
            logger.info("Fold %d | saved %d predictions -> %s", k, len(preds_df), preds_path)

            # baselines
            self._run_baselines(train_ds, test_ds, k)

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

