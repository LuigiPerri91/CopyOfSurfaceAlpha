"""
Evaluation entry point.

Loads per-fold prediction parquets saved by scripts/train.py,
computes all forecast + classification metrics, runs DM tests,
and writes a comparison CSV + JSON report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --predictions-dir outputs/predictions
    python scripts/evaluate.py --fold 2   # single fold
"""

import argparse
import json 
import logging 
import sys
from pathlib import Path

import numpy as np 
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from volregime.evaluation.forecast_metrics import (
    compute_vol_metrics, compute_classification_metrics, compute_per_regime_vol_metrics
)
from volregime.evaluation.stat_tests import mincer_zarnowitz

def main():
    parser = argparse.ArgumentParser(description="SurfaceAlpha evaluation")
    parser.add_argument("--predictions-dir", default=None,
                        help="Directory containing fold_*_test_preds.parquet files")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save the evaluation report")
    parser.add_argument("--fold", type=int, default=None,
                        help="Evaluate a single fold (default: all folds)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("evaluate")

    # locate prediction files
    pred_dir = Path(args.predictions_dir) if args.predictions_dir else ROOT / 'outputs' / 'predictions'
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / 'outputs' / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("fold_*_test_preds.parquet"))
    if not pred_files:
        logger.error("No prediction files found in %s", pred_dir)
        sys.exit(1)
    
    if args.fold is not None:
        pred_files = [f for f in pred_files if f"fold_{args.fold}_" in f.name]
        if not pred_files:
            logger.error("No predictions found for fold %d", args.fold)
            sys.exit(1)

    # load and concat all folds
    dfs = []
    for p in pred_files:
        df = pd.read_parquet(p)
        fold_idx = int(p.stem.split('_')[1])
        df['fold'] = fold_idx
        dfs.append(df)
        logger.info('Loaded fold %d: %d rows', fold_idx, len(df))
    
    all_preds = pd.concat(dfs, ignore_index=True)
    logger.info('Total predictions: %d rows across %d folds', len(all_preds), len(dfs))

    p_cols = [c for c in all_preds.columns if c.startswith('p_regime_')]
    regime_probs = all_preds[p_cols].values if p_cols else None

    # per fold metrics
    fold_rows = []
    for fold_idx, fold_df in all_preds.groupby('fold'):
        vol_m = compute_vol_metrics(
            fold_df['rv_pred'].values, fold_df['target_rv'].values
        )
        clf_m = compute_classification_metrics(
            fold_df['tail_prob'].values,
            fold_df['target_tail'].values,
            fold_df[[c for c in fold_df.columns if c.startswith('p_regime_')]].values,
            fold_df['target_regime'].values,
        ) if 'tail_prob' in fold_df.columns else {}

        row = {"fold": fold_idx, **{f"vol_{k}": v for k, v in vol_m.items()}, **{f"clf_{k}": v for k,v in clf_m.items()}}
        fold_rows.append(row)
        logger.info(
            "Fold %d | QLIKE=%.4f | QL=%.4f | R2=%.4f | RegAcc=%.3f",
            fold_idx,
            vol_m.get("qlike", float("nan")),
            vol_m.get("ql",    float("nan")),
            vol_m.get("r2",    float("nan")),
            clf_m.get("regime_accuracy", float("nan")),
        )
    
    fold_df_out = pd.DataFrame(fold_rows)
    fold_df_out.to_csv(out_dir / "per_fold_metrics.csv", index = False)

    # aggregate across folds
    num_cols = fold_df_out.select_dtypes(include=np.number).columns.tolist()
    agg = {
        col: {'mean': fold_df_out[col].mean(), 'std': fold_df_out[col].std()} for col in num_cols if col != 'fold'
    }

    logger.info("\n-- Aggregate metrics ------------------------------")
    for cols, vals in agg.items():
        logger.info("  %-40s  %.4f ± %.4f", col, vals["mean"], vals["std"])
    
    # mincer zarnowitz test (on full pooled set)
    mz = mincer_zarnowitz(
        np.exp(all_preds['rv_pred'].values),
        np.exp(all_preds['target_rv'].values)
    )
    logger.info("\n-- Mincer-Zarnowitz -------------------------------")
    logger.info("  intercept=%.4f  slope=%.4f  R2=%.4f  F-pval=%.4f",
                mz["intercept"], mz["slope"], mz["r2"], mz["f_pvalue"])
    
    # per regime breakdown
    if 'target_regime' in all_preds.columns:
        per_regime = compute_per_regime_vol_metrics(
            all_preds['rv_pred'].values,
            all_preds['target_rv'].values,
            all_preds['target_regime'].values
        )
        regime_rows = [{'regime': r, **m} for r,m in per_regime.items()]
        pd.DataFrame(regime_rows).to_csv(out_dir / "per_regime_metrics.csv", index=False)

    # save full report
    report = {
        'n_folds' : len(dfs),
        'n_samples': len(all_preds),
        'aggregate': {k : {'mean': v['mean'], 'std': v['std']} for k,v in agg.items()},
        'mincer_zarnowitz': mz
    }
    with open(out_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info('Evaluation report saved to %s', out_dir)

if __name__ == "__main__":
    main()
