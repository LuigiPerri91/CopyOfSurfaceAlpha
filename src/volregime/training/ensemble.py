"""
Deep ensemble support (Phase 7 feature — disabled by default).

When ensemble.enabled: true in training.yaml, train N independent models
with different seeds and combine their predictions at inference time.

FoldEnsemble:
    - Loads all member checkpoints for a given fold
    - Computes mean (and optionally std) of rv_forecast across members
    - std is used for confidence_scaling in the portfolio overlay

Config keys (cfg["training"]["ensemble"]):
    enabled, num_members, seeds
"""

import logging
from pathlib import Path


import numpy as np 
import torch
import torch.nn as nn 

from ..models.full_model import SurfaceAlphaModel
from ..utils.io import load_checkpoint
from ..utils.reproductibility import set_seed

logger = logging.getLogger(__name__)

class FoldEnsemble:
    """
    Ensemble of independently-trained SurfaceAlphaModel instances.

    Usage:
        ensemble = FoldEnsemble(cfg, fold_idx=0, output_dir=".")
        ensemble.load_members()

        # Single-sample inference
        outputs = ensemble.predict(surface, returns, vol_history, market_state)
        # outputs["rv_forecast"] = ensemble mean
        # outputs["rv_forecast_std"] = ensemble std (uncertainty)
    """
    def __init__(self, cfg: dict, fold_idx: int =0, output_dir: str | None = None):
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.root = Path(output_dir) if output_dir else Path('.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.members: list[nn.Module] = []

        ens_cfg = cfg.get('training', {}).get('ensemble',{})
        self.enabled = bool(ens_cfg.get('enabled', False))
        self.num_members = int(ens_cfg.get('num_members', 5))
        self.seeds = list(ens_cfg.get('seeds', [42, 123, 456, 789, 1024]))

    def train_members(self, train_loader, val_loader) -> None:
        """
        Train each ensemble member with its own seed.
        Call this instead of the standard Trainer if ensemble is enabled.
        """
        from .trainer import Trainer

        if not self.enabled:
            logger.info("Ensemble disabled - skipping member training")
            return
        
        for m in range(self.num_members):
            seed = self.seeds[m] if m < len(self.seeds) else m * 100 + 42
            set_seed(seed)

            logger.info('Fold %d | Ensemble member %d (seed= %d)', self.fold_idx, m, seed)
            
            model = SurfaceAlphaModel(self.cfg).to(self.device)
            trainer = Trainer(
                cfg = self.cfg, 
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                fold_idx=self.fold_idx,
                output_dir= str(
                    self.root / 'outputs' / 'checkpoints' / f'fold_{self.fold_idx}' / f'member_{m}'
                ),
                model_type='full'
            ) 
            trainer.fit()
            self.members.append(model)

    def load_members(self) -> None:
        """Load all member checkpoints for this fold from disk."""
        for m in range(self.num_members):
            ckpt_path = (self.root / 'outputs' / 'checkpoints' / f'fold_{self.fold_idx}' / f'member_{m}')
            if not ckpt_path.exists():
                logger.warning("Member %d checkpoint not found: %s", m, ckpt_path)
                continue
            model = SurfaceAlphaModel(self.cfg).to(self.device)
            load_checkpoint(ckpt_path, model)
            model.eval()
            self.members.append(model)
        logger.info('Fold %d | Loaded %d ensemble members.', self.fold_idx, len(self.members))

    @torch.no_grad()
    def predict(self, surface: torch.Tensor, returns: torch.Tensor, vol_history: torch.Tensor, market_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Ensemble forward pass — average predictions across all members.

        Returns dict with same keys as SurfaceAlphaModel.forward(), plus:
            rv_forecast_std: (batch,) — std of rv_forecast across members
            tail_prob_std:   (batch,) — std of tail_prob across members
        """
        if not self.members:
            raise RuntimeError("No members loaded. Call load_members() first.")

        all_rv, all_tail, all_regime, all_expert = [], [], [], []

        for model in self.members:
            out = model(
                surface.to(self.device),
                returns.to(self.device),
                vol_history.to(self.device),
                market_state.to(self.device),
            )
            all_rv.append(out['rv_forecast'])
            all_tail.append(out['tail_prob'])
            all_regime.append(out['regime_probs'])
            all_expert.append(out['expert_preds'])

        rv_stack = torch.stack(all_rv, dim=0)
        tail_stack = torch.stack(all_tail, dim=0)
        regime_stack = torch.stack(all_regime, dim=0)
        expert_stack = torch.stack(all_expert, dim=0)

        return {
            'rv_forecast': rv_stack.mean(dim=0),
            'rv_forecast_std': rv_stack.std(dim=0),
            'tail_prob': tail_stack.mean(dim=0),
            'tail_prob_std': tail_stack.std(dim=0),
            'regime_probs': regime_stack.mean(dim=0),
            'expert_preds': expert_stack.mean(dim=0),
            'regime_logits': torch.zeros_like(regime_stack.mean(dim=0))
        }