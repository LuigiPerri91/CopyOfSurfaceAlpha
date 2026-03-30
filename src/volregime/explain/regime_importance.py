"""
Regime-conditional feature importance.

Answers: "Which context features drive the model to predict each regime?"

vol_history (11 features from data.yaml):
    iv_rank, hv_rank, vol_risk_premium,
    iv_momentum_short, iv_momentum_medium,
    hv_momentum_short, hv_momentum_medium,
    days_since_iv_year_high, days_since_iv_year_low,
    days_since_hv_year_high, days_since_hv_year_low

market_state (3 features from data.yaml):
    vix, spy_return, risk_free_rate

Two approaches:

1. gradient_x_input_regime:
   For each sample, ∂regime_logits[k]/∂context x context for all k in {0..5}.
   Returns (B, 6, 14) — per-sample, per-regime, per-feature attribution.

2. RegimeImportance.compute:
   Aggregates over a full DataLoader:
   - mean |attribution| per regime -> which features fire in each regime
   - mean feature value by predicted regime -> regime fingerprints
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEFAULT_FEATURE_NAMES: list[str] = [
    # vol_history (11)
    "iv_rank",
    "hv_rank",
    "vol_risk_premium",
    "iv_momentum_short",
    "iv_momentum_medium",
    "hv_momentum_short",
    "hv_momentum_medium",
    "days_since_iv_year_high",
    "days_since_iv_year_low",
    "days_since_hv_year_high",
    "days_since_hv_year_low",
    # market_state (3)
    "vix",
    "spy_return",
    "risk_free_rate",
]

REGIME_NAMES: list[str] = [
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
    "sideways_quiet",
    "sideways_volatile",
]

VOL_HISTORY_DIM = 11

@dataclass
class RegimeImportanceResult:
    mean_attribution: np.ndarray
    mean_feature_by_regime: np.ndarray
    regime_counts: np.ndarray
    feature_names: list[str]
    regime_names: list[str] = field(default_factory=lambda: list(REGIME_NAMES))

    def to_attribution_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.mean_attribution,
            index = self.regime_names,
            columns=self.feature_names
        )
    
    def to_feature_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.mean_feature_by_regime,
            index = self.regime_names,
            columns = self.feature_names
        )

    def top_features_per_regime(self, n: int =5) -> dict[str, list[str]]:
        """Return the n most important context features per regime."""
        return {
            name: [self.feature_names[j] for j in np.argsort(self.mean_attribution[i])[::-1][:n]] for i, name in enumerate(self.regime_names)
        }
    
def gradient_x_input_regime(
    model: nn.Module,
    vol_history: torch.Tensor,
    market_state: torch.Tensor,
    surface: torch.Tensor,
    returns: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Gradient x input attribution for all 6 regime logits w.r.t. the
    concatenated context [vol_history | market_state].

    Loops over k=0..5, backpropagating each regime logit with
    retain_graph=True to preserve the computation graph between passes.

    Returns:
        (B, 6, 14) numpy array
        attribution[b, k, f] = contribution of feature f to regime-k
                                logit for sample b
    """
    model.eval()

    surf = surface.to(device).float().detach()
    ret = returns.to(device).float().detach()

    # concatenate into a single leaf tensor so we take one gradient
    context = torch.cat([
        vol_history.to(device).float(),
        market_state.to(device).float(),
    ], dim=-1).detach().requires_grad_(True) # (B, 14)

    vh = context[:, :VOL_HISTORY_DIM]
    ms = context[:, VOL_HISTORY_DIM:]

    out = model(surf, ret, vh, ms)
    logits = out['regime_logits'] # (B,6)

    B, K = logits.shape
    grads = torch.zeros(B, K, context.shape[1], device=device)

    for k in range(K):
        if context.grad is not None:
            context.grad.zero_()
        logits[:, k].sum().backward(retain_graph = (k < K - 1))
        grads[: ,k,:] = context.grad.detach()

    attribution = (grads * context.detach().unsqueeze(1)).cpu().numpy() 
    return attribution # (B, 6, 14)

class RegimeImportance:
    """
    Compute and aggregate regime-conditional feature importance over a DataLoader.

    The DataLoader must yield batches with keys:
        surface, returns, vol_history, market_state

    Args:
        model:           trained SurfaceAlphaModel (eval mode)
        device:          torch device
        feature_names:   14 context feature names (defaults to DEFAULT_FEATURE_NAMES)
        vol_history_dim: number of vol_history features (default 11)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        feature_names: list[str] | None = None,
        vol_history_dim: int = VOL_HISTORY_DIM
    ):
        self.model = model
        self.device = device
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self.vh_dim = vol_history_dim
        self.model.eval()

    def compute(self, loader) -> RegimeImportanceResult:
        """
        Run the full DataLoader and accumulate attributions per regime.

        Returns:
            RegimeImportanceResult
        """
        all_attrs: list[np.ndarray] = []
        all_context: list[np.ndarray] = []
        all_regimes: list[np.ndarray] = []

        for batch in loader:
            surf = batch['surface'].to(self.device).float()
            ret = batch['returns'].to(self.device).float()
            vh = batch['vol_history'].to(self.device).float()
            ms = batch['market_state'].to(self.device).float()

            attrs = gradient_x_input_regime(
                self.model, vh, ms, surf, ret, self.device 
            ) # (B, 6, 14)
            all_attrs.append(attrs)

            ctx = torch.cat([
                batch['vol_history'].float(),
                batch['market_state'].float()
            ], dim = -1).numpy()
            all_context.append(ctx)

            with torch.no_grad():
                out = self.model(surf, ret, vh, ms)
                pred = out['regime_probs'].argmax(dim=1).cpu().numpy()
            all_regimes.append(pred)

        attrs_all = np.concatenate(all_attrs, axis=0) # (N, 6, 14)
        context_all = np.concatenate(all_context, axis=0) # (N, 14)
        regimes_all = np.concatenate(all_regimes, axis=0) # (N,)

        # Mean |attribution| per regime over all samples
        mean_attr = np.abs(attrs_all).mean(axis=0) # (6,14)

        # mean feature values for samples predicted in each regime
        mean_feat = np.zeros((6, len(self.feature_names)))
        counts = np.zeros(6, dtype=int)
        for k in range(6):
            mask = regimes_all == k
            counts[k] = int(mask.sum())
            if counts[k] > 0:
                mean_feat[k] = context_all[mask].mean(axis=0)

        return RegimeImportanceResult(
            mean_attr, mean_feat, counts, self.feature_names
        )

    def attribution_for_batch(self, batch: dict) -> np.ndarray:
        """Compute (B, 6, 14) attribution for a single batch dict."""
        return gradient_x_input_regime(
            self.model,
            batch['vol_history'].to(self.device).float(),
            batch['market_state'].to(self.device).float(),
            batch['surface'].to(self.device).float(),
            batch['returns'].to(self.device).float(),
            self.device
        )

