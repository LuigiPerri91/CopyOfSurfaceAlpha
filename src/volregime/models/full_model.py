"""
SurfaceAlphaModel — assembles all sub-modules into one nn.Module.

Forward inputs:
    surface:      (batch, 6, 12, 20)
    returns:      (batch, 60, 8)
    vol_history:  (batch, 11)   — may contain NaN
    market_state: (batch, 3)

Forward outputs (dict):
    rv_forecast   (batch,)     — MoE blended log(RV) prediction
    tail_prob     (batch,)     — P(tail event)
    regime_logits (batch, 6)   — raw logits (for CrossEntropyLoss)
    regime_probs  (batch, 6)   — softmax probabilities
    expert_preds  (batch, 6)   — per-expert log(RV) predictions
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .surface_encoder import SurfaceEncoder
from .returns_encoder import build_returns_encoder
from .context_encoder import ContextEncoder
from .fusion import build_fusion
from .output_heads import VolatilityHead, TailRiskHead, RegimeHead
from .regime_moe import RegimeMoE

class SurfaceAlphaModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        #accept both full config dict and model-only sub-dict
        mcfg = config.get('model', config)

        surf_cfg = mcfg.get('surface_encoder', {})
        ret_cfg = mcfg.get('returns_encoder',{})
        ctx_cfg = mcfg.get('context_encoder',{})
        heads_cfg = mcfg.get('output_heads',{})
        moe_cfg = mcfg.get('regime_moe',{})

        num_regimes = heads_cfg.get('regime_head',{}).get('num_regimes',6)

        # encoders 
        self.surface_encoder = SurfaceEncoder(
            num_input_channels=surf_cfg.get('num_input_channels', 6),
            patch_size_h=surf_cfg.get("patch_size_h", 3),
            patch_size_w=surf_cfg.get("patch_size_w", 4),
            embed_dim=surf_cfg.get("embed_dim", 128),
            num_heads=surf_cfg.get("num_heads", 4),
            num_layers=surf_cfg.get("num_layers", 4),
            mlp_ratio=surf_cfg.get("mlp_ratio", 4.0),
            dropout=surf_cfg.get("dropout", 0.1),
            attn_dropout=surf_cfg.get("attn_dropout", 0.1),
            grid_h=12, grid_w=20,
        )
        surf_dim = surf_cfg.get('embed_dim', 128)

        self.returns_encoder = build_returns_encoder(mcfg)
        ret_dim = ret_cfg.get('hidden_dim', 64)

        self.context_encoder = ContextEncoder(
            input_dim=ctx_cfg.get("input_dim", 14),
            hidden_dims=ctx_cfg.get("hidden_dims", [64, 32]),
            dropout=ctx_cfg.get("dropout", 0.2),
        )
        ctx_dim = ctx_cfg.get('hidden_dims',[64,32])[-1]

        # fusion
        self.fusion = build_fusion(mcfg, surf_dim=surf_dim, ret_dim=ret_dim, ctx_dim=ctx_dim)
        fusion_dim = self.fusion.output_dim

        # output heads
        vol_h = heads_cfg.get('vol_head',{})
        tail_h = heads_cfg.get('tail_head', {})
        reg_h = heads_cfg.get('regime_head', {})

        self.vol_head = VolatilityHead(fusion_dim, vol_h.get('hidden_dim',64), vol_h.get('num_layers',2))
        self.tail_head = TailRiskHead(fusion_dim, tail_h.get('hidden_dim',32), tail_h.get('num_layers',2))
        self.regime_head = RegimeHead(fusion_dim, reg_h.get('hidden_dim',64), num_regimes, reg_h.get('num_layers',2))

        # mixture of experts
        self.regime_moe = RegimeMoE(
            in_dim=fusion_dim, 
            num_experts=moe_cfg.get('num_experts',num_regimes), 
            expert_hidden_dim=moe_cfg.get('expert_hidden_dim', 32),
            expert_num_layers=moe_cfg.get('expert_num_layers',2)
        )

    def forward(self, surface: torch.Tensor, returns: torch.Tensor, vol_history: torch.Tensor, market_state: torch.Tensor) -> dict[str, torch.Tensor]:
        z_surf = self.surface_encoder(surface)
        z_ret = self.returns_encoder(returns)
        z_ctx = self.context_encoder(vol_history, market_state)

        z = self.fusion(z_surf, z_ret, z_ctx)

        regime_logits = self.regime_head(z)
        regime_probs = F.softmax(regime_logits, dim=-1)
        tail_prob = self.tail_head(z)
        rv_forecast, expert_preds = self.regime_moe(z, regime_probs)

        return {
            "rv_forecast": rv_forecast,
            'tail_prob' : tail_prob,
            "regime_logits": regime_logits,
            'regime_probs': regime_probs,
            'expert_preds': expert_preds,
        }

    def count_parameters(self) -> dict[str, int]:
        modules = {
            "surface_encoder": self.surface_encoder,
            "returns_encoder": self.returns_encoder,
            "context_encoder": self.context_encoder,
            "fusion":          self.fusion,
            "vol_head":        self.vol_head,
            "tail_head":       self.tail_head,
            "regime_head":     self.regime_head,
            "regime_moe":      self.regime_moe,
        }
        counts = {n: sum(p.numel() for p in m.parameters())
                  for n, m in modules.items()}
        counts["total"] = sum(counts.values())
        return counts