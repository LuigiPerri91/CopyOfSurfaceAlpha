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

class SurfaceAlphaLoss(nn.Module):
    def __init__(self, cfg: dict):
        """Args: cfg = cfg["training"] (the training sub-config)."""
        super().__init__()
        w = cfg.get('loss_weights',{})
        self.lambda_vol = float(w.get('lambda_vol', 1.0))
        self.lambda_tail = float(w.get('lambda_tail', 0.3))
        self.lambda_reg = float(w.get('lambda_reg',0.2))
        self.lambda_smooth = float(w.get('lambda_smooth', 0.05))

        self.huber = nn.HuberLoss(delta=float(cfg.get('huber_delta', 1.0)), reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
    
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
        l_tail = self.bce(outputs['tail_prob'], batch['target_tail'].float())
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
