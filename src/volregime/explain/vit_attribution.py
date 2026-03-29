"""
Gradient-based attribution for the ViT surface encoder.

The surface encoder processes (B, 6, 12, 20) IV surfaces through a Vision
Transformer. Patches are 3×4 (maturity × moneyness), giving a 4×5 grid of
20 patches.

Surface channels (data.yaml): iv, spread_norm, obs_mask, staleness, delta, vega

Two attribution methods:

1. gradient_saliency (recommended):
   Computes |∂output/∂surface| × |surface|, takes the L2 norm over channels
   to produce a (12, 20) spatial heatmap of which strike/maturity regions
   matter most.

2. attention_rollout:
   Manually unrolls the TransformerEncoder layers, calling each layer's
   self_attn module directly with need_weights=True — no forward hooks, no
   recursion risk. Rolls the resulting attention matrices back from CLS to
   patches (Abnar & Zuidema 2020), then upsamples the 4×5 patch grid to
   (12, 20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np 
import torch
import torch.nn as nn

SURFACE_CHANNELS = ['iv','spread_norm','obs_mask', 'staleness', 'delta', 'vega']

GRID_H, GRID_W = 12, 20
PATCH_H, PATCH_W = 3, 4
N_PATCH_H = GRID_H // PATCH_H
N_PATCH_W = GRID_W // PATCH_W
N_PATCHES = N_PATCH_H * N_PATCH_W

@dataclass
class SurfaceAttributionResult:
    heatmap: np.ndarray
    patch_importance: np.ndarray
    method: str # 'gradient' or 'attention_rollout'
    channel_saliency: np.ndarray | None # (6,12,20) per-channel; gradient only

def gradient_saliency(
    model: nn.Module,
    surface: torch.Tensor,
    returns: torch.Tensor,
    vol_history: torch.Tensor,
    market_state: torch.Tensor,
    output: str = 'rv_forecast',
    aggregate: str = 'mean'
) -> SurfaceAttributionResult:
    """
    Gradient × input saliency for the IV surface.

    Computes ∂output/∂surface × surface, takes L2 norm over channels
    for a (12, 20) spatial importance map.

    Args:
        output:    'rv_forecast' | 'tail_prob' | 'regime_0'..'regime_5'
        aggregate: how to collapse the batch dimension
    """
    model.eval()
    surface = surface.detach().float().requires_grad_(True)
    ret = returns.detach().float()
    vh = vol_history.detach().float()
    ms = market_state.detach().float()

    out_dict = model(surface, ret, vh, ms)
    scalar = _extract_scalar(out_dict, output)
    scalar.sum().backward()

    with torch.no_grad():
        # gradient x input: (B, 6, 12, 20)
        gxi = (surface.grad * surface).detach().cpu().numpy()
        # L2 over channels -> (B, 12, 20)
        spatial = np.sqrt((gxi ** 2).sum(axis=1))

    if aggregate == 'mean':
        heatmap = spatial.mean(axis=0) # (12,20)
        channel_sal = gxi.mean(axis=0) # (6,12 ,20)
    else:
        idx = spatial.reshape(spatial.shape[0], -1).sum(axis=1).argmax()
        heatmap = spatial[idx]
        channel_sal = gxi[idx]

    heatmap = _normalize(heatmap)
    patch_importance = _to_patch_grid(heatmap)

    return SurfaceAttributionResult(
        heatmap, patch_importance, 'gradient', channel_sal
    )

def attention_rollout(
    model: nn.Module,
    surface: torch.Tensor,
    returns: torch.Tensor,
    vol_history: torch.Tensor,
    market_state: torch.Tensor,
    discard_ratio: float = 0.1,
    aggregate: str = "mean"
) -> SurfaceAttributionResult:
    """
    Attention rollout (Abnar & Zuidema 2020) for the surface ViT.

    Instead of forward hooks, we manually run the TransformerEncoder
    layer-by-layer. At each layer we call self_attn(..., need_weights=True)
    directly — the same operation the layer performs internally, but with
    weights returned. This avoids hooks and any recursion risk entirely.

    The surface encoder must follow the standard nn.TransformerEncoder /
    nn.TransformerEncoderLayer structure. norm_first=True is assumed (matches
    the model config).

    Args:
        discard_ratio: zero out the lowest fraction of attention weights
                       per layer before rollout (reduces noise)
    """
    model.eval()

    with torch.no_grad():
        surf = surface.detach().float()
        z_tokens = _embed_surface_tokens(model.surface_encoder, surf) # (B, 1+N, D)
        attn_weights = _forward_transformer_with_attn(
            model.surface_encoder.transformer, z_tokens
        ) # list of (B, 1+N, 1+N) tensors, one per layer

    rollout = _compute_rollout(attn_weights, discard_ratio)
    cls_to_patches = rollout[:, 0, 1:] #(B, N_PATCHES = 20)

    patch_flat = (
        cls_to_patches.numpy().mean(axis=0) if aggregate == 'mean' else cls_to_patches.numpy().max(axis= 0)
    )

    patch_importance = _normalize(patch_flat.reshape(N_PATCH_H, N_PATCH_W))
    heatmap = _upsample_patch_to_surface(patch_importance)

    return SurfaceAttributionResult(
        heatmap, patch_importance, 'attention_rollout', None
    )

def _embed_surface_tokens(
    surface_encoder: nn.Module,
    surface: torch.Tensor # (B, 6, 12, 20)
) -> torch.Tensor:
    """
    Run just the patch embedding + CLS token + positional embedding,
    stopping before the TransformerEncoder.

    Assumes the ViT surface encoder has attributes:
        patch_embed — PatchEmbedding or Conv2d
        cls_token — nn.Parameter (1, 1, D)
        pos_embed — nn.Parameter (1, 1+N, D)
    """
    se = surface_encoder

    # (B, num_patches, embed_dim) if PatchEmbedding else (B, D, nH, nW) if Conv2d
    x = se.patch_embed(surface)
    if x.ndim == 4:
        B, D, nh, nw = x.shape
        x = x.flatten(2).transpose(1, 2)
    else:
        B = x.shape[0]

    cls = se.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)

    x = x + se.pos_embed[:, : x.shape[1], :]

    return x

def _forward_transformer_with_attn(
    transformer: nn.TransformerEncoder,
    src : torch.Tensor # (B, seq, D)
) -> list[torch.Tensor]:
    """
    Manually unroll nn.TransformerEncoder, returning a list of attention
    weight tensors (one per layer) alongside the final output.

    Each layer in nn.TransformerEncoderLayer (norm_first=True) computes:
        x = x + dropout1(self_attn(norm1(x))[0])
        x = x + dropout2(ffn(norm2(x)))

    We replicate this exactly but call self_attn with need_weights=True so
    PyTorch returns the (B, seq, seq) averaged attention matrix.

    In eval mode dropout is a no-op, so this is numerically identical to
    the normal forward pass.
    """
    x = src
    attn_weights = []

    for layer in transformer.layers:
        normed = layer.norm1(x)
        attn_out, w = layer.self_attn(
            normed, normed, normed, need_weights=True, average_attn_weights = True
        )
        attn_weights.append(w.detach().cpu())

        x = x + layer.dropout1(attn_out)

        x = x+ layer.dropout2(
            layer.linear2(
                layer.dropout(
                    layer.activation(layer.linear1(layer.norm2(x)))
                )
            )
        )
    if transformer.norm is not None:
        x = transformer.norm(x)
    
    return attn_weights

# Rollout helpers

def _compute_rollout(
    attn_list: list[torch.Tensor],
    discard_ratio: float
) -> torch.Tensor:
    """
    Roll attention matrices across layers (Abnar & Zuidema 2020).

    For each layer:
      1. Optionally zero the lowest discard_ratio weights (noise reduction)
      2. Add the identity matrix (residual connection)
      3. Re-normalise rows
      4. Left-multiply into the running rollout matrix

    Returns (B, seq, seq) rollout matrix.
    """
    B, seq_len, _ = attn_list[0].shape
    rollout = torch.eye(seq_len).unsqueeze(0).expand(B, -1, -1).clone()

    for attn in attn_list:
        if discard_ratio > 0:
            flat = attn.reshape(B, -1)
            thresh = flat.quantile(discard_ratio, dim=1).reshape(B, 1, 1)
            attn = attn * (attn >= thresh).float()
        
        attn = attn + torch.eye(seq_len).unsqueeze(0)
        row_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        attn = attn / row_sum
        rollout = torch.bmm(attn, rollout)

    return rollout

# general helpers

def _extract_scalar(out_dict: dict, output: str) -> torch.Tensor:
    if output =='rv_forecast':
        return out_dict["rv_forecast"]
    if output == "tail_prob":
        return out_dict["tail_prob"]
    if output.startswith("regime_"):
        k = int(output.split("_")[1])
        return out_dict["regime_probs"][:, k]
    raise ValueError(f"Unknown output: {output!r}")

def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)

def _to_patch_grid(heatmap: np.ndarray) -> np.ndarray:
    """Average (12, 20) heatmap within each 3x4 patch -> (4, 5)."""
    grid = np.zeros((N_PATCH_H, N_PATCH_W))
    for i in range(N_PATCH_H):
        for j in range(N_PATCH_W):
            grid[i, j] = heatmap[
                i * PATCH_H : (i + 1) * PATCH_H,
                j * PATCH_W : (j + 1) * PATCH_W,
            ].mean()
    return _normalize(grid)

def _upsample_patch_to_surface(patch_grid: np.ndarray) -> np.ndarray:
    """Repeat each patch value over its 3x4 region -> (12, 20)."""
    return np.kron(patch_grid, np.ones((PATCH_H, PATCH_W)))




