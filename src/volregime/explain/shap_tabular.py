"""
SHAP-based attribution for the context encoder inputs.

The context encoder receives vol_history (11 features) + market_state (3 features)
concatenated to (batch, 14). This module wraps the full model to hold the surface
and returns encodings fixed at a baseline, then applies GradientExplainer to compute
per-feature SHAP values for the 14-dimensional context input.

Why GradientExplainer over KernelExplainer:
    - The model is a PyTorch nn.Module — gradients are cheap and exact.
    - KernelExplainer is model-agnostic but requires thousands of forward passes.
    - GradientExplainer gives the same Shapley attribution via integrated gradients
      in a fraction of the time.

Usage:
    explainer = SHAPExplainer(model, background_loader, device=device)
    result = explainer.explain(batch)
    # result.shap_values: (N, 14)  — one row per sample
    # result.feature_names: list[str] — names for all 14 context features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np 
import shap
import torch
import torch.nn as nn 

# vol_history: 11 features (from data.yaml vol_history.features)
# market_state: 3 features (from data.yaml market_state)
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

VOL_HISTORY_DIM = 11
MARKET_STATE_DIM = 3

@dataclass
class ShapResult:
    shap_values: np.ndarray # (N, 14)
    base_values: np.ndarray # (N,) - expected output over background
    data: np.ndarray # (N, 14) — input values
    feature_names: list[str]
    output_name: str

class _ContextWrapper(nn.Module):
    """
    Wraps SurfaceAlphaModel to accept only the concatenated context vector
    [vol_history | market_state] (14-dim) as the varying input, holding
    surface and returns at fixed baseline tensors.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        surface_baseline: torch.Tensor, # (1, 6, 12, 20)
        returns_baseline: torch.Tensor, # (1, 60, 8)
        output: str = 'rv_forecast',
        vol_history_dim: int = VOL_HISTORY_DIM,
    ):
        super().__init__()
        self.model = model
        self.register_buffer('surface_baseline', surface_baseline)
        self.register_buffer('returns_baseline', returns_baseline)
        self.output = output
        self.vh_dim = vol_history_dim

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, 14) — [vol_history | market_state]
        Returns:
            (B,) scalar output
        """
        B = context.shape[0]
        vol_history = context[:, : self.vh_dim]
        market_state = context[:, self.vh_dim: ]
        surface = self.surface_baseline.expand(B, -1, -1, -1)
        returns = self.returns_baseline.expand(B, -1, -1)

        out = self.model(surface, returns, vol_history, market_state)

        if self.output == 'rv_forecast':
            return out['rv_forecast']
        if self.output == 'tail_prob':
            return out['tail_prob']
        if self.output.startswith('regime_'):
            k = int(self.output.split('_')[1])
            return out['regime_probs'][:, k]
        raise ValueError(f"Unknown output target: {self.output!r}")

class SHAPExplainer: 
    """
    Gradient-based SHAP explainer for the 14-dim context encoder input.

    Args:
        model:            trained SurfaceAlphaModel (eval mode)
        background:       (K, 14) numpy array of context samples from training set
                          (typically ~100 samples)
        surface_baseline: (1, 6, 12, 20) mean surface tensor from training set
        returns_baseline: (1, 60, 8) mean returns tensor from training set
        device:           torch device
        output:           which model output to explain:
                          'rv_forecast' | 'tail_prob' | 'regime_0'..'regime_5'
        feature_names:    14 feature names (uses DEFAULT_FEATURE_NAMES if None)
        vol_history_dim:  number of vol_history features (default 11)
    """

    def __init__(
        self, 
        model: nn.Module,
        background: np.ndarray,
        surface_baseline: torch.Tensor,
        returns_baseline: torch.Tensor,
        device: torch.device,
        output: str = "rv_forecast",
        feature_names: list[str] | None = None,
        vol_history_dim: int = VOL_HISTORY_DIM
    ):
        model.eval()
        self.device = device
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self.output = output
        self._background = background

        self.wrapper = _ContextWrapper(
            model,
            surface_baseline.to(device),
            returns_baseline.to(device),
            output=output,
            vol_history_dim=vol_history_dim,
        ).to(device)
        self.wrapper.eval()

        bg_tensor = torch.tensor(background, dtype=torch.float32, device=device)
        self.explainer = shap.GradientExplainer(self.wrapper, bg_tensor)

    def explain(self, context: np.ndarray, n_samples: int= 200) -> ShapResult:
        """
        Compute SHAP values for a batch of context inputs.

        Args:
            context:   (N, 14) numpy array of [vol_history | market_state]
            n_samples: number of background samples for gradient estimation

        Returns:
            ShapResult with .shap_values shape (N, 14)
        """
        x = torch.tensor(context, dtype=torch.float32, device=self.device)
        shap_vals = self.explainer.shap_values(x, nsamples=n_samples)
        shap_arr = np.array(shap_vals)

        with torch.no_grad():
            bg = torch.tensor(self._background, dtype=torch.float32, device=self.device)
            base_val = float(self.wrapper(bg).mean().item())

        return ShapResult(
            shap_values= shap_arr,
            base_values= np.full(len(context), base_val),
            data = context,
            feature_names= self.feature_names,
            output_name = self.output
        )

    def mean_absolute_importance(self, result: ShapResult) -> dict[str, float]:
        """Global feature importance: mean |SHAP| per feature, sorted descending."""
        importance = np.abs(result.shap_values).mean(axis=0)
        return dict(
            sorted(
                zip(result.feature_names, importance.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
        )

    def to_shap_explanation(self, result: ShapResult) -> shap.Explanation:
        """Convert to shap.Explanation for use with shap.plots.*"""
        return shap.Explanation(
            values= result.shap_values,
            base_values= result.base_values,
            data = result.data,
            feature_names= result.feature_names
        )
