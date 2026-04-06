"""
MoE gating weight analysis.

Answers: "Which context features drive each expert's activation?"

The MoE gating weights are regime_probs (softmax of regime head logits).
For each expert k, we compute the Spearman correlation between p_k and every
context feature across the test set.  This is proper regime-conditional attribution
because it uses the soft gating weights (not argmax), capturing partial activations.

Outputs:
    gating_correlations.json   — Spearman r per expert per feature, sorted by |r|
    gating_feature_means.csv   — mean feature value when expert is "active" (p_k > 0.3)
                                 vs "inactive" (p_k < 0.1)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .regime_importance import DEFAULT_FEATURE_NAMES, VOL_HISTORY_DIM

logger = logging.getLogger(__name__)

REGIME_NAMES = [
    "bull_quiet", "bull_volatile", "bear_quiet",
    "bear_volatile", "sideways_quiet", "sideways_volatile",
]


@dataclass
class GatingResult:
    correlations: dict[str, dict[str, float]]   # regime → {feature: spearman_r}
    feature_means_active: dict[str, dict[str, float]]   # regime → {feature: mean when p_k > thresh}
    feature_means_inactive: dict[str, dict[str, float]]
    feature_names: list[str]


class GatingAnalysis:
    """
    Collect MoE gating weights and context features over a DataLoader,
    then compute Spearman correlation between each expert's gating weight
    and each context feature.

    Args:
        model:        trained SurfaceAlphaModel (eval mode)
        device:       torch device
        macro_dim:    if set, truncates market_state to first N features (for V4 compat)
        active_thresh:  p_k threshold to call an expert "active" (default 0.3)
        inactive_thresh: p_k threshold to call an expert "inactive" (default 0.1)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        macro_dim: int | None = None,
        active_thresh: float = 0.3,
        inactive_thresh: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.macro_dim = macro_dim
        self.active_thresh = active_thresh
        self.inactive_thresh = inactive_thresh
        self.model.eval()

    def compute(self, loader: DataLoader) -> GatingResult:
        all_probs: list[np.ndarray] = []
        all_ctx: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                surf = batch["surface"].to(self.device).float()
                ret = batch["returns"].to(self.device).float()
                vh = batch["vol_history"].to(self.device).float()
                ms = batch["market_state"].to(self.device).float()
                if self.macro_dim is not None:
                    ms = ms[:, :self.macro_dim]

                out = self.model(surf, ret, vh, ms)
                probs = out["regime_probs"].cpu().numpy()  # (B, 6)
                all_probs.append(probs)

                ms_ctx = batch["market_state"].float()
                if self.macro_dim is not None:
                    ms_ctx = ms_ctx[:, :self.macro_dim]
                ctx = torch.nan_to_num(
                    torch.cat([batch["vol_history"].float(), ms_ctx], dim=-1),
                    nan=0.0,
                ).numpy()
                all_ctx.append(ctx)

        probs_all = np.concatenate(all_probs, axis=0)   # (N, 6)
        ctx_all = np.concatenate(all_ctx, axis=0)        # (N, ctx_dim)

        n_ctx = ctx_all.shape[1]
        feature_names = DEFAULT_FEATURE_NAMES[:n_ctx]
        n_experts = probs_all.shape[1]

        correlations: dict[str, dict[str, float]] = {}
        means_active: dict[str, dict[str, float]] = {}
        means_inactive: dict[str, dict[str, float]] = {}

        for k in range(n_experts):
            rname = REGIME_NAMES[k]
            p_k = probs_all[:, k]

            # Spearman correlation between p_k and each context feature
            corrs = {}
            for j, feat in enumerate(feature_names):
                r, _ = spearmanr(p_k, ctx_all[:, j])
                corrs[feat] = round(float(r), 4) if not np.isnan(r) else 0.0
            correlations[rname] = dict(
                sorted(corrs.items(), key=lambda x: -abs(x[1]))
            )

            # Mean feature values when expert is active vs inactive
            active_mask = p_k > self.active_thresh
            inactive_mask = p_k < self.inactive_thresh
            means_active[rname] = {
                feat: round(float(ctx_all[active_mask, j].mean()), 4) if active_mask.sum() > 0 else float("nan")
                for j, feat in enumerate(feature_names)
            }
            means_inactive[rname] = {
                feat: round(float(ctx_all[inactive_mask, j].mean()), 4) if inactive_mask.sum() > 0 else float("nan")
                for j, feat in enumerate(feature_names)
            }

            logger.info(
                "Expert %d (%s) | active=%d (%.1f%%) | top corr: %s=%.3f",
                k, rname,
                int(active_mask.sum()), 100 * active_mask.mean(),
                list(correlations[rname].keys())[0],
                list(correlations[rname].values())[0],
            )

        return GatingResult(correlations, means_active, means_inactive, feature_names)
