"""
Context encoder: vol-history + market state → z_ctx

Input:  vol_history (batch, F_vh ~11) — may contain NaN for missing dates
        market_state (batch, F_mkt ~3)
        concatenated → (batch, 14)
Output: (batch, context_dim=32)
"""

import torch
import torch.nn as nn 

class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dims: list[int] | None = None, dropout:float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64,32]
        layers, d = [nn.LayerNorm(input_dim)], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d,h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, vol_history: torch.Tensor, market_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol_history:  (batch, F_vh)  — NaN → 0 before MLP
            market_state: (batch, F_mkt)
        Returns:
            z_ctx: (batch, output_dim)
        """
        vol_history = torch.nan_to_num(vol_history, nan=0.0)
        market_state = torch.nan_to_num(market_state, nan=0.0)
        x = torch.cat([vol_history,market_state], dim=-1)
        return self.mlp(x)