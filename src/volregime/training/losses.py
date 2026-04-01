"""
Multi-task combined loss for SurfaceAlphaModel.

L = lambda_vol*L_vol + lambda_tail*L_tail + lambda_reg*L_reg + lambda_smooth*L_smooth

    L_vol:    Huber(rv_forecast, target_rv)               — primary regression
    L_tail:   BCE(tail_prob, target_tail)                 — tail-risk binary clf
    L_reg:    CrossEntropy(regime_logits, target_regime)  — regime clf
    L_smooth: Var(expert_preds, dim=experts).mean()       — MoE agreement regularizer
              Penalises expert disagreement → stabilises MoE training.

Config keys (cfg["training"]):
    loss_weights.lambda_vol / lambda_tail / lambda_reg / lambda_smooth
    huber_delta
"""

from torch import tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceAlphaLoss(nn.Module):
    def __init__(self, cfg: dict, regime_class_weights: "torch.Tensor | None" = None):
        """Args:
            cfg: cfg["training"] (the training sub-config).
            regime_class_weights: optional float32 tensor of shape (n_classes,) passed to
                CrossEntropyLoss to correct for class imbalance in the regime head.
        """
        super().__init__()
        w = cfg.get('loss_weights',{})
        self.lambda_vol = float(w.get('lambda_vol', 1.0))
        self.lambda_tail = float(w.get('lambda_tail', 0.3))
        self.lambda_reg = float(w.get('lambda_reg', 0.5))
        self.lambda_smooth = float(w.get('lambda_smooth', 0.05))

        self.huber = nn.HuberLoss(delta=float(cfg.get('huber_delta', 1.0)), reduction='mean')
        # pos_weight for tail BCE: upweights positives (~22% base rate → weight ≈ 2.0)
        self.tail_pos_weight = float(cfg.get('tail_pos_weight', 2.0))
        self.ce = nn.CrossEntropyLoss(weight=regime_class_weights, reduction='mean')
    
    def forward(self,outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            outputs: SurfaceAlphaModel.forward() dict
                     rv_forecast (B,), tail_prob (B,),
                     regime_logits (B,6), expert_preds (B,6)
            batch:   DataLoader batch
                     target_rv (B,), target_tail (B,), target_regime (B,)
        Returns:
            total_loss, {component: float}
        """
        l_vol = self.huber(outputs['rv_forecast'], batch['target_rv'])
        t = batch['target_tail'].float()
        w = torch.where(t > 0.5,
                        torch.full_like(t, self.tail_pos_weight),
                        torch.ones_like(t))
        l_tail = F.binary_cross_entropy(outputs['tail_prob'], t, weight=w)
        l_reg = self.ce(outputs['regime_logits'], batch['target_regime'].long())
        l_smooth = outputs['expert_preds'].var(dim=1).mean() # (B,K) -> var over K

        total = (
            self.lambda_vol * l_vol + self.lambda_tail * l_tail + self.lambda_reg * l_reg + self.lambda_smooth * l_smooth
        )
        return total, {
            "vol": l_vol.item(),
            "tail":   l_tail.item(),
            "regime": l_reg.item(),
            "smooth": l_smooth.item(),
            "total":  total.item(),
        }

class SingleTaskLoss(nn.Module):
    """Huber loss for deep_ts baselines (single scalar output, no aux heads)"""

    def __init__(self, huber_delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta, reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.huber(pred, target)
