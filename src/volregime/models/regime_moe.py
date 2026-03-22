"""
Regime-conditional Mixture of Experts for volatility prediction.

Each of K experts produces its own log(RV) forecast.
The regime head's softmax probabilities gate them:
    ŷ = Σ_k  p_k · ŷ_k

Each expert specialises in a regime's distinct vol dynamics:
    0 bull_quiet     — low steady vol
    1 bull_volatile  — elevated trending vol
    2 bear_quiet     — quiet downtrend
    3 bear_volatile  — crisis spike dynamics
    4 sideways_quiet — mean-reversion
    5 sideways_volatile — extreme chop, hardest to forecast
"""

import torch
import torch.nn as nn 

class RegimeMoE(nn.Module):
    def __init__(self, in_dim: int = 128, num_experts: int =6, expert_hidden_dim : int = 32, expert_num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            self._make_expert(in_dim, expert_hidden_dim, expert_num_layers, dropout) for _ in range(num_experts)
        ])

    @staticmethod
    def _make_expert(in_dim: int, hidden_dim: int, num_layers:int, dropout: float) -> nn.Module:
        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d,1))
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, regime_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z:            (batch, in_dim)
            regime_probs: (batch, K)  — softmax probs, rows sum to 1
        Returns:
            blended:      (batch,)   — weighted expert mixture
            expert_preds: (batch, K) — individual expert outputs
        """
        expert_preds = torch.stack(
            [e(z).squeeze(-1) for e in self.experts], dim=1
        ) # (batch, K)
        blended = (regime_probs * expert_preds).sum(dim=1)
        return blended, expert_preds