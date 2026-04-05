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

    # load OHLCV for all active symbols
    raw_dir = Path(cfg['paths']['raw_dir'])
    active_symbols = cfg.get('active_symbols_list', ['SPY'])

    def _load_ohlcv(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df = df.set_index('date')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    ohlcv_dict: dict[str, pd.DataFrame] = {}
    for sym in active_symbols:
        sym_path = raw_dir / 'underlying' / f'{sym}.parquet'
        if sym_path.exists():
            ohlcv_dict[sym] = _load_ohlcv(sym_path)
            logger.info("OHLCV loaded: %-6s  %d rows, %s -> %s",
                        sym, len(ohlcv_dict[sym]),
                        ohlcv_dict[sym].index[0].date(), ohlcv_dict[sym].index[-1].date())
        else:
            logger.warning("OHLCV not found for %s at %s — skipping", sym, sym_path)

    # ensure SPY is present for macro regime signals
    if 'SPY' not in ohlcv_dict:
        spy_path = raw_dir / 'underlying' / 'SPY.parquet'
        if spy_path.exists():
            ohlcv_dict['SPY'] = _load_ohlcv(spy_path)
            logger.info("SPY loaded separately for macro regime signals")
        else:
            logger.warning("SPY OHLCV not found; macro regime signals will be unavailable")

    if not ohlcv_dict:
        logger.error("No OHLCV files loaded — check raw_dir: %s", raw_dir)
        sys.exit(1)

    logger.info("Loaded OHLCV for %d symbols: %s", len(ohlcv_dict), sorted(ohlcv_dict))

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
    result = engine.run(preds_df, ohlcv_dict, vix_series)

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

    bs = result.benchmark_summary
    if bs:
        logger.info("\n── Benchmark Comparison ───────────────────────────────────────")
        header = f"  {'strategy':<25}  {'AnnRet%':>8}  {'Sharpe':>8}  {'MaxDD%':>8}  {'Calmar':>8}"
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        logger.info("  %-25s  %8.2f  %8.3f  %8.2f  %8.3f", "surfacealpha",
                    s.get("ann_return_pct", float("nan")),
                    s.get("sharpe", float("nan")),
                    s.get("max_drawdown_pct", float("nan")),
                    s.get("calmar", float("nan")))
        for bname, bm in bs.items():
            logger.info("  %-25s  %8.2f  %8.3f  %8.2f  %8.3f", bname,
                        bm.get("ann_return_pct", float("nan")),
                        bm.get("sharpe", float("nan")),
                        bm.get("max_drawdown_pct", float("nan")),
                        bm.get("calmar", float("nan")))

    logger.info("Outputs saved to %s", output_dir)

if __name__ == "__main__":
    main()

