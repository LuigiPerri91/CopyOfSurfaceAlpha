"""
Diagnostic script — three investigations from post-training analysis.

Usage:
    ACTIVE_SYMBOLS=liquid_core DATA_DIR=./data/liquid_core \\
        python scripts/diagnose_v3.py \\
            --predictions-dir runs/liquid_core_v3/outputs/predictions \\
            --equity-curve   runs/liquid_core_v3/outputs/backtest/equity_curve.csv \\
            --run-dir        runs/liquid_core_v3/outputs \\
            --output         runs/liquid_core_v3/outputs/diagnostics

Outputs (written to --output dir):
    moe_expert_diversity.json   — per-regime expert prediction std; MoE collapse verdict
    bull_quiet_sigma.json       — sigma_hat vs actual RV in bull_quiet; overforecast ratio
    fold_stability.json         — per-fold date ranges + vol_qlike trend
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

REGIME_NAMES = {
    0: 'bull_quiet', 1: 'bull_volatile', 2: 'bear_quiet',
    3: 'bear_volatile', 4: 'sideways_quiet', 5: 'sideways_volatile',
}


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    files = sorted(pred_dir.glob('fold_*_test_preds.parquet'))
    if not files:
        raise FileNotFoundError(f'No fold parquets found in {pred_dir}')
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        fold_idx = int(f.stem.split('_')[1])
        df['fold'] = fold_idx
        dfs.append(df)
    preds = pd.concat(dfs, ignore_index=True)
    preds['date'] = pd.to_datetime(preds['date'])
    log.info('Loaded %d prediction rows across %d folds', len(preds), len(files))
    return preds


# MoE Expert Diversity

def diagnose_moe(preds: pd.DataFrame, out_dir: Path) -> dict:
    expert_cols = [c for c in preds.columns if c.startswith('expert_')]
    if not expert_cols:
        log.warning('No expert_* columns found — skipping MoE diagnostic')
        return {}

    log.info('MoE diagnostic: found expert columns %s', expert_cols)
    expert_mat = preds[expert_cols].values.astype(np.float32)  # (N, K)

    preds = preds.copy()
    preds['expert_std']   = expert_mat.std(axis=1)
    preds['expert_range'] = expert_mat.max(axis=1) - expert_mat.min(axis=1)

    global_mean_std = float(preds['expert_std'].mean())
    global_med_std  = float(preds['expert_std'].median())
    collapsed = global_mean_std < 0.01

    log.info('─── MoE Expert Diversity ───────────────────────────────')
    log.info('  Global mean expert_std : %.5f', global_mean_std)
    log.info('  Global median expert_std: %.5f', global_med_std)
    log.info('  Verdict: %s', 'MoE COLLAPSED (std < 0.01)' if collapsed else 'MoE SPECIALISED')

    per_regime = {}
    if 'regime_pred' in preds.columns:
        for ridx, rname in REGIME_NAMES.items():
            mask = preds['regime_pred'] == ridx
            if mask.sum() < 5:
                continue
            subset = preds[mask]
            per_regime[rname] = {
                'n': int(mask.sum()),
                'mean_expert_std': round(float(subset['expert_std'].mean()), 6),
                'median_expert_std': round(float(subset['expert_std'].median()), 6),
                'pct_std_gt_005': round(float((subset['expert_std'] > 0.05).mean()), 4),
                'mean_expert_range': round(float(subset['expert_range'].mean()), 6),
            }
            log.info('  %-20s n=%4d  mean_std=%.5f  pct>0.05=%.2f',
                     rname, mask.sum(),
                     per_regime[rname]['mean_expert_std'],
                     per_regime[rname]['pct_std_gt_005'])

    result = {
        'global': {
            'mean_expert_std': round(global_mean_std, 6),
            'median_expert_std': round(global_med_std, 6),
            'pct_std_gt_005': round(float((preds['expert_std'] > 0.05).mean()), 4),
            'collapsed': collapsed,
            'verdict': 'COLLAPSED' if collapsed else 'SPECIALISED',
        },
        'per_regime': per_regime,
        'n_experts': len(expert_cols),
    }

    out_path = out_dir / 'moe_expert_diversity.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info('Saved → %s', out_path)
    return result


# Bull_quiet Sigma Analysis 

def diagnose_bull_quiet(preds: pd.DataFrame, equity_curve_path: Path,
                        out_dir: Path, sigma_target: float = 0.10) -> dict:
    if not equity_curve_path.exists():
        log.warning('Equity curve not found at %s — skipping bull_quiet diagnostic', equity_curve_path)
        return {}

    ec = pd.read_csv(equity_curve_path, index_col='date', parse_dates=True)
    ec.index = pd.to_datetime(ec.index)

    # aggregate preds to daily (mean rv_pred and target_rv per date)
    daily = (
        preds.groupby('date')[['rv_pred', 'target_rv']]
        .mean()
        .reset_index()
    )
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.set_index('date')

    joined = ec.join(daily[['rv_pred', 'target_rv']], how='inner')
    bq = joined[joined['regime_name'] == 'bull_quiet'].dropna(subset=['sigma_hat', 'target_rv'])

    if len(bq) == 0:
        log.warning('No bull_quiet days found in equity curve — skipping')
        return {}

    # actual annualised vol from target_rv (log-RV, 21-day horizon)
    bq = bq.copy()
    bq['actual_sigma_ann'] = np.exp(bq['target_rv']) * math.sqrt(252 / 21)
    bq['ratio'] = bq['sigma_hat'] / bq['actual_sigma_ann']
    bq['implied_weight'] = (sigma_target / bq['sigma_hat']).clip(0, 1.5)
    bq['ideal_weight']   = (sigma_target / bq['actual_sigma_ann']).clip(0, 1.5)

    mean_ratio        = float(bq['ratio'].mean())
    pct_overforecast  = float((bq['ratio'] > 1.0).mean())
    avg_implied_weight = float(bq['implied_weight'].mean())
    avg_ideal_weight   = float(bq['ideal_weight'].mean())

    log.info('─── Bull_quiet Sigma Analysis ──────────────────────────')
    log.info('  Days in bull_quiet      : %d', len(bq))
    log.info('  Mean sigma_hat/actual   : %.3f  (1.0 = perfectly calibrated)', mean_ratio)
    log.info('  %% days over-forecasting: %.1f%%', pct_overforecast * 100)
    log.info('  Avg implied weight      : %.3f', avg_implied_weight)
    log.info('  Avg ideal weight        : %.3f  (weight if perfectly calibrated)', avg_ideal_weight)
    log.info('  Return drag (ideal-impl): %.3f', avg_ideal_weight - avg_implied_weight)

    result = {
        'n_bull_quiet_days': len(bq),
        'mean_ratio': round(mean_ratio, 4),
        'pct_overforecast': round(pct_overforecast, 4),
        'avg_sigma_hat': round(float(bq['sigma_hat'].mean()), 4),
        'avg_actual_sigma': round(float(bq['actual_sigma_ann'].mean()), 4),
        'avg_implied_weight': round(avg_implied_weight, 4),
        'avg_ideal_weight': round(avg_ideal_weight, 4),
        'weight_drag': round(avg_ideal_weight - avg_implied_weight, 4),
    }

    out_path = out_dir / 'bull_quiet_sigma.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info('Saved → %s', out_path)
    return result


# Fold Stability

def diagnose_fold_stability(run_dir: Path, out_dir: Path) -> dict:
    summary_files = sorted(run_dir.glob('fold_*_summary.json'))
    metrics_path  = run_dir / 'evaluation' / 'per_fold_metrics.csv'

    if not summary_files:
        log.warning('No fold_*_summary.json found in %s — skipping fold stability', run_dir)
        return {}

    fold_metrics = {}
    if metrics_path.exists():
        fm = pd.read_csv(metrics_path)
        for _, row in fm.iterrows():
            fold_metrics[int(row['fold'])] = {
                'vol_qlike': round(float(row['vol_qlike']), 4),
                'vol_r2':    round(float(row['vol_r2']), 4),
                'tail_auc':  round(float(row.get('clf_tail_auc', float('nan'))), 4),
            }

    folds = []
    log.info('─── Fold Stability Analysis ────────────────────────────')
    for sf in summary_files:
        with open(sf) as f:
            s = json.load(f)
        fold_idx = s.get('fold_idx', int(sf.stem.split('_')[1]))
        test_start = str(s.get('test_start', ''))
        test_end   = str(s.get('test_end', ''))
        train_start = str(s.get('train_start', ''))
        train_end   = str(s.get('train_end', ''))

        test_year = test_start[:4] if test_start else 'unknown'
        m = fold_metrics.get(fold_idx, {})

        entry = {
            'fold': fold_idx,
            'train_start': train_start,
            'train_end':   train_end,
            'test_start':  test_start,
            'test_end':    test_end,
            'test_year':   test_year,
            **m,
        }
        folds.append(entry)
        log.info('  Fold %d | test=%s→%s | qlike=%s | r2=%s',
                 fold_idx, test_start, test_end,
                 m.get('vol_qlike', 'n/a'), m.get('vol_r2', 'n/a'))

    result = {'folds': folds}
    out_path = out_dir / 'fold_stability.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info('Saved → %s', out_path)
    return result


# Main

def main():
    p = argparse.ArgumentParser(description='V3 diagnostics')
    p.add_argument('--predictions-dir', required=True)
    p.add_argument('--equity-curve',    required=True)
    p.add_argument('--run-dir',         required=True)
    p.add_argument('--output',          required=True)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(Path(args.predictions_dir))

    diagnose_moe(preds, out_dir)
    diagnose_bull_quiet(preds, Path(args.equity_curve), out_dir)
    diagnose_fold_stability(Path(args.run_dir), out_dir)

    log.info('All diagnostics complete → %s/', out_dir)


if __name__ == '__main__':
    main()
