"""
Backtest entry point.

Loads per-fold prediction parquets from outputs/predictions/,
loads OHLCV + VIX data, and runs BacktestEngine.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --predictions-dir outputs/predictions
    python scripts/backtest.py --output ./runs/backtest_v1
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volregime.utils.config import get_project_root, load_config
from volregime.portfolio.backtest_engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(description="SurfaceAlpha backtest")
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--ohlcv",  default=None,
                        help="Path to underlying OHLCV parquet (e.g. data/raw/underlying/SPY.parquet)")
    parser.add_argument("--vix",    default=None,
                        help="Path to VIX parquet (e.g. data/raw/underlying/^VIX.parquet)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("backtest")

    # config
    project_root = get_project_root()
    cfg = load_config()

    # load predictions
    pred_dir = Path(args.predictions_dir) if args.predictions_dir else project_root / 'outputs' / 'predictions'
    pred_files = sorted(pred_dir.glob('fold_*_test_preds.parquet'))
    if not pred_files:
        logger.error('No prediction files found in %s', pred_dir)
        sys.exit(1)

    preds_df = pd.concat([pd.read_parquet(f) for f in pred_files], ignore_index=True)
    logger.info('Loaded %d prediction rows for %d folds', len(preds_df), len(pred_files))

    # load OHLCV
    raw_dir = Path(cfg['paths']['raw_dir'])
    active_sym = cfg.get('active_symbols_list', ['SPY'])[0] # use first symbol

    ohlcv_path = args.ohlcv or raw_dir / 'underlying' / f'{active_sym}.parquet'
    if not Path(ohlcv_path).exists():
        logger.error("OHLCV file not found: %s", ohlcv_path)
        sys.exit(1)
    
    ohlcv_df = pd.read_parquet(ohlcv_path)
    # ensure index is a DatetimeIndex — parquet may store dates as a column
    if 'date' in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.set_index('date')
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
    ohlcv_df = ohlcv_df.sort_index()
    logger.info("OHLCV: %d rows, %s -> %s", len(ohlcv_df), ohlcv_df.index[0].date(), ohlcv_df.index[-1].date())

    # load vix (optional)
    vix_series = None
    vix_path = args.vix or raw_dir / 'underlying' / '^VIX.parquet'
    if Path(vix_path).exists():
        vix_df = pd.read_parquet(vix_path)
        if 'date' in vix_df.columns:
            vix_df = vix_df.set_index('date')
        vix_df.index = pd.to_datetime(vix_df.index)
        vix_series = vix_df['close'].sort_index()
        logger.info("VIX loaded: %d rows", len(vix_series))

    # run backtest
    output_dir = args.output or str(project_root)
    engine = BacktestEngine(cfg, output_dir=output_dir)
    result = engine.run(preds_df, ohlcv_df, vix_series)

    # print summary
    s = result.summary
    logger.info("\n── Backtest Summary ───────────────────────────────────────────")
    logger.info("  Annual Return:  %.2f%%", s.get("ann_return_pct", float("nan")))
    logger.info("  Annual Vol:     %.2f%%", s.get("ann_vol_pct", float("nan")))
    logger.info("  Sharpe:         %.3f", s.get("sharpe", float("nan")))
    logger.info("  Sortino:        %.3f", s.get("sortino", float("nan")))
    logger.info("  Max Drawdown:   %.2f%%", s.get("max_drawdown_pct", float("nan")))
    logger.info("  Calmar:         %.3f", s.get("calmar", float("nan")))
    logger.info("  Turnover (ann): %.1f", s.get("turnover_ann", float("nan")))
    logger.info("  Total Return:   %.2f%%", s.get("total_return_pct", float("nan")))
    logger.info("Outputs saved to %s", output_dir)

if __name__ == "__main__":
    main()

