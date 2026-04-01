"""
Training loop for one walk-forward fold.

Reads all hyperparameters from cfg["training"]:
    optimizer:     name, learning_rate, weight_decay, betas, eps
    scheduler:     name ("cosine"|"plateau"), cosine_t_max, cosine_eta_min,
                   plateau_factor, plateau_patience, warmup_epochs
    early_stopping: enabled, patience, min_delta
    gradient_clip_norm, num_epochs, batch_size

Supports:
    model_type="full"             → SurfaceAlphaModel (4 inputs, multi-task loss)
    model_type="deep_ts_baseline" → LSTM/GRU/TCN (returns only, single Huber loss)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from ..utils.io import save_checkpoint, load_checkpoint
from .losses import SurfaceAlphaLoss, SingleTaskLoss

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg: dict, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, fold_idx: int = 0, output_dir: str | None = None, model_type: str = 'full', regime_class_weights: torch.Tensor | None = None):
        """
        Args:
            cfg:          full merged config dict from load_config()
            model:        nn.Module to train
            train_loader: DataLoader for training split
            val_loader:   DataLoader for validation split
            fold_idx:     walk-forward fold index (0-4)
            output_dir:   project root; checkpoints go under outputs/checkpoints/
            model_type:   "full" | "deep_ts_baseline"
        """
        self.fold_idx = fold_idx
        self.model_type = model_type
        cfg_train = cfg.get('training',cfg)

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # optimizer 
        opt = cfg_train.get('optimizer', {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = float(opt.get('learning_rate', 3e-4)),
            weight_decay = float(opt.get('weight_decay', 0.01)),
            betas = tuple(opt.get('betas', [0.9,0.999])),
            eps=float(opt.get('eps', 1e-8))
        )

        # scheduler
        sched = cfg_train.get('scheduler',{})
        self.warmup = int(sched.get('warmup_epochs', 3))
        sched_name = sched.get('name','cosine')
        if sched_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max = int(sched.get('cosine_t_max', 50)),
                eta_min=float(sched.get('cosine_eta_min', 1e-6))
            )
        else: # plateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor = float(sched.get('plateau_factor', 0.5)),
                patience=float(sched.get('plateau_patience', 5))
            )
        
        # loss
        if model_type == 'full':
            weights = regime_class_weights.to(self.device) if regime_class_weights is not None else None
            self.loss_fn = SurfaceAlphaLoss(cfg_train, regime_class_weights=weights)
        else:
            self.loss_fn = SingleTaskLoss(float(cfg_train.get('huber_delta',1.0)))

        # training config
        self.num_epochs = int(cfg_train.get('num_epochs', 100))
        self.grad_clip = float(cfg_train.get('gradient_clip_norm', 1.0))

        es = cfg_train.get('early_stopping', {})
        self.use_es = bool(es.get('enabled', True))
        self.patience = int(es.get('patience', 10))
        self.min_delta = float(es.get('min_delta', 1e-4))

        # state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # checkpoint
        root = Path(output_dir) if output_dir else Path('.')
        ckpt_dir = root / 'outputs' / 'checkpoints' / f'fold_{fold_idx}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = ckpt_dir / "best.pt"

    def _to_device(self, batch:dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
    
    def _forward_and_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if self.model_type == 'full':
            outputs = self.model(
                batch['surface'],
                batch['returns'],
                batch['vol_history'],
                batch['market_state'],
            )
            return self.loss_fn(outputs, batch)
        else:
            pred = self.model(batch['returns'])
            loss = self.loss_fn(pred, batch['target_rv'])
            return loss, {'total': loss.item()}

    def _train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            loss, _ = self._forward_and_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n,1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss, _ = self._forward_and_loss(batch)
            total += loss.item()
            n += 1
        return total / max(n,1)

    def fit(self) -> float:
        """Train until early stopping or num_epochs. Returns best val loss."""
        logger.info(
            "Fold %d | type=%s | device=%s | epochs=%d | patience=%d",
            self.fold_idx, self.model_type, self.device,
            self.num_epochs, self.patience,
        )

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()

            # LR schedule (hold during warmup)
            if epoch >= self.warmup:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            
            if wandb.run is not None:
                wandb.log({
                    f"fold_{self.fold_idx}/train_loss": train_loss,
                    f"fold_{self.fold_idx}/val_loss": val_loss,
                    f"fold_{self.fold_idx}/lr": lr,
                    "epoch": epoch,
                })

            logger.info("Fold %d | E%3d | train=%.4f | val=%.4f | lr=%.2e", self.fold_idx, epoch, train_loss, val_loss, lr)

            # Early stopping + checkpoint
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, self.ckpt_path)
            elif self.use_es:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        "Fold %d | Early stop epoch=%d (best=%d val=%.4f)",
                        self.fold_idx, epoch, self.best_epoch, self.best_val_loss,
                    )
                    break
        
        # reload best weights
        if self.ckpt_path.exists():
            load_checkpoint(self.ckpt_path, self.model, self.optimizer)

        logger.info("Fold %d | Done. Best val=%.4f @ epoch %d",
                    self.fold_idx, self.best_val_loss, self.best_epoch)
        return self.best_val_loss
        