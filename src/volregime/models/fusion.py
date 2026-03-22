"""
Fusion: combines [z_surf, z_ret, z_ctx] → z

Two modes:
    concat_mlp — concatenate then MLP (default, simpler, more stable)
    cross_attention — z_surf (query) attends to z_ret + z_ctx (kv)
"""

import torch
import torch.nn as nn 

class ConcatFusion(nn.Module):
    """
    Concatenate embeddings and pass through 2-layer MLP.
    Default input: 128 + 64 + 32 = 224 → 128.
    """

    def __init__(self, surf_dim: int = 128, ret_dim: int =64, ctx_dim: int =32, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(surf_dim + ret_dim + ctx_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.output_dim = hidden_dim
    
    def forward(self, z_surf: torch.Tensor, z_ret: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z_surf, z_ret,z_ctx], dim=-1))

class CrossAttentionFusion(nn.Module):
    """
    z_surf (query) cross-attends to a single kv token = concat(z_ret, z_ctx).
    """
    def __init(self, surf_dim:int =128, ret_dim: int =64, ctx_dim: int = 32, hidden_dim: int= 128, num_heads: int =4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(surf_dim, hidden_dim)
        self.k_proj = nn.Linear(ret_dim + ctx_dim, hidden_dim)
        self.v_proj = nn.Linear(ret_dim + ctx_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim *2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim *2, hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, z_surf: torch.Tensor, z_ret: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(z_surf).unsqueeze(1) # (B, 1, hidden_dim)
        kv_in = torch.cat([z_ret, z_ctx], dim =-1).unsqueeze(1)
        k, v = self.k_proj(kv_in), self.v_proj(kv_in) # (B, 1, hidden_dim)
        out , _ = self.attn(q,k,v) # (B, 1, hidden_dim)
        z = self.norm(out.squeeze(1) + self.q_proj(z_surf))
        return z + self.ffn(z)

def build_fusion(config: dict, surf_dim: int =128, ret_dim: int =64, ctx_dim: int =32) -> nn.Module:
    cfg = config.get('fusion',{})
    method = cfg.get('method','concat_mlp')
    hidden_dim = cfg.get('hidden_dim', 128)
    dropout = cfg.get('dropout',0.1)

    if method == 'concat_mlp':
        return ConcatFusion(surf_dim, ret_dim, ctx_dim, hidden_dim, dropout)
    elif method == 'cross_attention':
        return CrossAttentionFusion(surf_dim, ret_dim, ctx_dim, hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion method: {method}")