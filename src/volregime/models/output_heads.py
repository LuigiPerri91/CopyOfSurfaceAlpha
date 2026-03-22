"""
Three task-specific output heads, all consuming the fused z.

    VolatilityHead -> scalar log(RV) forecast (no activation)
    TailRiskHead -> P(tail event) ∈ [0, 1] (sigmoid applied here)
    RegimeHead -> K=6 regime logits (softmax applied in full_model)
"""

import torch
import torch.nn as nn 

def _mlp(in_dim: int, hidden_dim:int, out_dim: int, num_layers: int, dropout: float, final_activation: nn.Module | None = None) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(num_layers + 1):
        layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)

class VolatilityHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim : int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(in_dim, hidden_dim, 1, num_layers, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1) # (batch,)

class TailRiskHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim : int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(in_dim, hidden_dim, 1, num_layers, dropout, final_activation=nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1) # (batch,) in [0, 1]

class RegimeHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim : int = 64,num_regimes: int =6, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = _mlp(in_dim, hidden_dim, num_regimes, num_layers, dropout)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) # (batch, num_regimes) raw logits

