"""
Explanation script — SHAP + gradient saliency + regime importance
for a trained SurfaceAlpha model checkpoint.

Usage:
    python scripts/explain.py --fold 0
    python scripts/explain.py --fold 2 --output regime_3 --method gradient
    python scripts/explain.py --checkpoint outputs/checkpoints/fold_0/best.pt

Outputs (written to outputs/explain/):
    shap_values.npy          (N, 14) SHAP values for context features
    shap_importance.json     global feature importance dict
    surface_heatmap.npy      (12, 20) averaged surface attribution
    patch_importance.npy     (4, 5) per-patch importance
    regime_attribution.npy   (6, 14) mean |grad x input| per regime
    regime_feature_means.csv mean feature value when each regime is predicted
    top_features.json        top-5 context features per regime

Dataset paths follow the convention set by scripts/build_surfaces.py:
    data/processed/sample_index.parquet
    data/processed/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np 
import pandas as pd 
import torch
import yaml
from torch.utils.data import DataLoader

from volregime.data.dataset import SurfaceAlphaDataset
from volregime.models.full_model import SurfaceAlphaModel
from volregime.utils.config import get_project_root, load_config
from volregime.utils.io import load_checkpoint
from volregime.explain import (
    DEFAULT_FEATURE_NAMES,
    GatingAnalysis,
    RegimeImportance,
    SHAPExplainer,
    attention_rollout,
    gradient_saliency,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="SurfaceAlpha explanation runner")
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). "
             "Defaults to outputs/checkpoints/fold_{fold}/best.pt",
    )
    p.add_argument("--fold", type=int, default=0,
                   help="Walk-forward fold index to load predictions from")
    p.add_argument(
        "--output", type=str, default="rv_forecast",
        choices=[
            "rv_forecast", "tail_prob",
            "regime_0", "regime_1", "regime_2",
            "regime_3", "regime_4", "regime_5",
        ],
        help="Model output to explain with SHAP",
    )
    p.add_argument(
        "--method", type=str, default="gradient",
        choices=["gradient", "attention_rollout"],
        help="Surface attribution method",
    )
    p.add_argument("--n-background", type=int, default=100,
                   help="Background samples for SHAP GradientExplainer")
    p.add_argument("--n-explain", type=int, default=200,
                   help="Test samples to explain")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--explain-output", type=str, default=None,
                   help="Output directory for explain results "
                        "(default: <project_root>/outputs/explain)")
    p.add_argument("--macro-dim", type=int, default=None,
                   help="Override model.context_encoder.macro_dim (use 3 for V4 checkpoints)")
    return p.parse_args()

def _collect_tensors(loader, key: str, n: int) -> torch.Tensor:
    """Collect up to n samples of tensor `key` from a DataLoader."""
    parts = []
    total = 0
    for batch in loader:
        t = batch[key].float()
        parts.append(t)
        total += t.shape[0]
        if total >= n:
            break
    return torch.cat(parts, dim=0)[:n]


def _collect_context(loader, n: int, market_state_dim: int | None = None) -> np.ndarray:
    """Collect [vol_history | market_state] as (N, vh_dim+ms_dim) numpy array.

    market_state_dim: if set, truncates market_state to first N features.
    Use when running explain on an older checkpoint trained with fewer macro features.
    """
    parts = []
    total = 0
    for batch in loader:
        vh  = batch["vol_history"].float()
        ms  = batch["market_state"].float()
        if market_state_dim is not None:
            ms = ms[:, :market_state_dim]
        ctx = torch.cat([vh, ms], dim=-1).numpy()
        parts.append(ctx)
        total += ctx.shape[0]
        if total >= n:
            break
    return np.nan_to_num(np.concatenate(parts, axis=0)[:n], nan=0.0)

def main():
    args = parse_args()
    root = get_project_root()
    cfg = load_config()
    device = torch.device(args.device)

    # dataset
    processed_dir = Path(cfg['paths']['processed_dir'])
    sample_index_path = processed_dir / 'sample_index.parquet'

    if not sample_index_path.exists():
        raise FileNotFoundError(
            f"sample_index.parquet not found at {sample_index_path}. Run scripts/build_surface.py first"
        )
    
    dataset = SurfaceAlphaDataset(
        sample_index_path=str(sample_index_path),
        processed_dir= str(processed_dir)
    )

    log.info("Dataset: %d samples", len(dataset))

    # use the first n_background samples as background, last n_explain as test
    dates = dataset.get_dates()
    bg_ds = dataset.get_subset(dates[0], dates[args.n_background - 1])
    test_ds = dataset.get_subset(dates[-args.n_explain], dates[-1])

    bg_loader = DataLoader(bg_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # model
    if args.macro_dim is not None:
        cfg['model']['context_encoder']['macro_dim'] = args.macro_dim
    model = SurfaceAlphaModel(cfg).to(device)
    model.eval()

    ckpt_path = args.checkpoint or str(
        root / 'outputs' / 'checkpoints' / f'fold_{args.fold}' / 'best.pt'
    )
    load_checkpoint(ckpt_path, model, strict=False)
    log.info('Loaded checkpoint: %s', ckpt_path)

    # output dir
    out_dir = Path(args.explain_output) if args.explain_output else root / 'outputs' / 'explain'
    out_dir.mkdir(parents=True, exist_ok=True)

    # baselines (mean of background set)
    log.info('Collecting background tensors...')
    background_ctx = _collect_context(bg_loader, args.n_background, market_state_dim=args.macro_dim)
    surf_baseline = _collect_tensors(bg_loader, 'surface', args.n_background).mean(0, keepdim=True).to(device)
    ret_baseline = _collect_tensors(bg_loader, "returns", args.n_background).mean(0, keepdim=True).to(device)

    log.info("Collecting test context...")
    test_ctx = _collect_context(test_loader, args.n_explain, market_state_dim=args.macro_dim)

    # SHAP
    log.info("Running SHAP GradientExplainer (output=%s)...", args.output)
    shap_explainer = SHAPExplainer(
        model = model,
        background= background_ctx,
        surface_baseline= surf_baseline,
        returns_baseline= ret_baseline,
        device = device,
        output = args.output,
        feature_names = DEFAULT_FEATURE_NAMES
    )
    shap_result = shap_explainer.explain(test_ctx)
    np.save(out_dir / 'shap_values.npy', shap_result.shap_values)

    importance = shap_explainer.mean_absolute_importance(shap_result)
    with open(out_dir / "shap_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    log.info("Context feature importance (SHAP, top 10):")
    for feat, val in list(importance.items())[:10]:
        log.info("  %-28s %.5f", feat, val)

    # Surface attribution
    log.info("Running surface attribution (method=%s)...", args.method)
    batch = next(iter(test_loader))
    surf = batch["surface"].to(device).float()
    ret = batch["returns"].to(device).float()
    vh = batch["vol_history"].to(device).float()
    ms = batch["market_state"].to(device).float()
    if args.macro_dim is not None:
        ms = ms[:, :args.macro_dim]

    if args.method == "gradient":
        attr = gradient_saliency(model, surf, ret, vh, ms, output=args.output)
    else:
        attr = attention_rollout(model, surf, ret, vh, ms)

    np.save(out_dir / "surface_heatmap.npy",  attr.heatmap)
    np.save(out_dir / "patch_importance.npy", attr.patch_importance)
    log.info("Surface heatmap saved -> %s/surface_heatmap.npy", out_dir)
    log.info("Patch grid (4 maturity × 5 moneyness):\n%s", np.round(attr.patch_importance, 3))

    # Regime importance
    log.info("Running regime importance over test set...")
    ri = RegimeImportance(model=model, device=device, feature_names=DEFAULT_FEATURE_NAMES, macro_dim=args.macro_dim)
    ri_result = ri.compute(test_loader)

    np.save(out_dir / "regime_attribution.npy", ri_result.mean_attribution)
    ri_result.to_feature_df().to_csv(out_dir / "regime_feature_means.csv")

    top = ri_result.top_features_per_regime(n=5)
    with open(out_dir / "top_features.json", "w") as f:
        json.dump(top, f, indent=2)

    log.info("Top-3 context features per regime:")
    for regime, feats in ri_result.top_features_per_regime(n=3).items():
        k = list(ri_result.regime_names).index(regime)
        count = int(ri_result.regime_counts[k])
        log.info("  %-22s (%4d samples): %s", regime, count, ", ".join(feats))

    log.info("All outputs saved -> %s/", out_dir)

    # MoE gating analysis — Spearman correlation between expert weights and context features
    log.info("Running MoE gating analysis...")
    ga = GatingAnalysis(model=model, device=device, macro_dim=args.macro_dim)
    ga_result = ga.compute(test_loader)

    with open(out_dir / "gating_correlations.json", "w") as f:
        json.dump(ga_result.correlations, f, indent=2)

    rows = []
    for regime in ga_result.feature_names and ga_result.correlations:
        active = ga_result.feature_means_active[regime]
        inactive = ga_result.feature_means_inactive[regime]
        for feat in ga_result.feature_names:
            rows.append({
                "regime": regime,
                "feature": feat,
                "mean_active": active[feat],
                "mean_inactive": inactive[feat],
                "diff": round(active[feat] - inactive[feat], 4) if not (
                    isinstance(active[feat], float) and isinstance(inactive[feat], float)
                    and (active[feat] != active[feat] or inactive[feat] != inactive[feat])
                ) else float("nan"),
            })
    pd.DataFrame(rows).to_csv(out_dir / "gating_feature_means.csv", index=False)

    log.info("MoE gating — top Spearman correlation per expert:")
    for regime, corrs in ga_result.correlations.items():
        top_feat, top_r = next(iter(corrs.items()))
        log.info("  %-22s top: %s = %.3f", regime, top_feat, top_r)

if __name__ == "__main__":
    main()