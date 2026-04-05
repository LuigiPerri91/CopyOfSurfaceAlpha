"""
Context encoder: vol-history + market state → z_ctx

Dual-stream architecture: vol_history and market_state are processed by
separate LayerNorm + MLP branches before being combined. This prevents
the large-magnitude vol_history features (days_since_* up to 252) from
dominating the shared LayerNorm and washing out the low-variance macro
features (vix, spy_return, risk_free_rate).

Input:  vol_history  (batch, vol_history_dim ~11) — may contain NaN
        market_state (batch, macro_dim ~3)
Output: (batch, output_dim=32)
"""

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.GELU(),
    )


class ContextEncoder(nn.Module):
    def __init__(
        self,
        vol_history_dim: int = 11,
        macro_dim: int = 3,
        vol_hidden_dim: int = 64,
        macro_hidden_dim: int = 16,
        output_dim: int = 32,
        dropout: float = 0.2,
        # legacy compat: if called with input_dim + hidden_dims, map to dual-stream
        input_dim: int | None = None,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        # legacy single-arg path: derive dims from old-style input_dim (14) + hidden_dims
        if input_dim is not None and hidden_dims is not None:
            macro_dim = 3
            vol_history_dim = input_dim - macro_dim
            vol_hidden_dim = hidden_dims[0] if hidden_dims else 64
            output_dim = hidden_dims[-1] if hidden_dims else 32
            macro_hidden_dim = max(8, output_dim // 2)

        self.vol_mlp = _build_mlp(vol_history_dim, vol_hidden_dim, output_dim, dropout)
        self.macro_mlp = _build_mlp(macro_dim, macro_hidden_dim, output_dim // 2, dropout)
        self.proj = nn.Linear(output_dim + output_dim // 2, output_dim)
        self.output_dim = output_dim

    def forward(self, vol_history: torch.Tensor, market_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol_history:  (batch, vol_history_dim)  — NaN → 0 before MLP
            market_state: (batch, macro_dim)
        Returns:
            z_ctx: (batch, output_dim)
        """
        vol_history = torch.nan_to_num(vol_history, nan=0.0)
        market_state = torch.nan_to_num(market_state, nan=0.0)
        z_vol = self.vol_mlp(vol_history)         # (batch, output_dim)
        z_macro = self.macro_mlp(market_state)    # (batch, output_dim // 2)
        return self.proj(torch.cat([z_vol, z_macro], dim=-1))