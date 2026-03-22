"""
Temporal encoder for the returns lookback window.

Input:  (batch, L=60, F_ret=8)
Output: (batch, hidden_dim)   — z_ret

Three backbone options: gru (default), tcn, transformer.
"""
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
sys.path.append(project_root) 

import torch
import torch.nn as nn
from ..baselines.deep_ts import _TCNBlock

class GRUReturnsEncoder(nn.Module):
    """Bidirectional GRU -> last hidden state -> linear projection -> z_ret"""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int =2, dropout: float = 0.2):
        super(). __init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(returns)
        return self.proj(torch.cat([h_n[-2],h_n[-1]], dim=-1)) # (B, hidden_dim)

class TCNReturnsEncoder:
    """
    Dilated causal TCN → global average pool → z_ret.
    """
    def __init__(self, input_dim: int = 8, num_channels: list[int] | None = None, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        layers, in_ch = [], input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.hidden_dim = num_channels[-1]

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        x = returns.permute(0,2,1) # (batch, F_ret, L)
        return self.network(x).mean(dim=-1) # (B, hidden_dim)

class TransformerReturnsEncoder(nn.Module):
    """Causal transformer -> last token -> z_ret"""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64,
    num_layers: int = 2, num_heads: int = 4,
    ffn_dim: int = 128, dropout: float = 0.2, max_seq_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
        dim_feedforward=ffn_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        B, L, _ = returns.shape
        x = self.input_proj(returns) + self.pos_embed[:,:L, :]
        x = self.norm(self.transformer(x))
        return x[:,-1,:] # last token → (B, hidden_dim)

def build_returns_encoder(config: dict) -> nn.Module:
    """Factory: reads config['returns_encoder'] and returns the right encoder."""
    cfg = config.get("returns_encoder", {})
    backbone = cfg.get('backbone','gru')
    input_dim = cfg.get('input_dim',8)
    hidden_dim = cfg.get('hidden_dim',64)
    num_layers = cfg.get('num_layers',2)
    dropout = cfg.get('dropout', 0.2)

    if backbone == 'gru':
        return GRUReturnsEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif backbone == 'tcn':
        return TCNReturnsEncoder(input_dim, cfg.get('tcn_num_channels',[64,64,64]), cfg.get('tcn_kernel_size',3), dropout)
    elif backbone == 'transformer':
        return TransformerReturnsEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=cfg.get('transformer_heads',4), ffn_dim=cfg.get('transformer_ffn_dim', 128), dropout=dropout
        )
    else:
        raise ValueError(f"Unknown returns encoder backbone: {backbone}")