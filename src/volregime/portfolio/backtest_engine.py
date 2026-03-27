"""
Full portfolio simulation engine.

Simulates the SurfaceAlpha strategy day-by-day using:
    - Model predictions from fold parquets
    - RegimeIdentifier on OHLCV window
    - PortfolioOverlay for position sizing
    - Transaction costs per trade

Also simulates benchmark strategies:
    buy_and_hold, inverse_vol, constant_vol_target

Outputs:
    outputs/backtest/equity_curve.csv
    outputs/backtest/position_history.csv
    outputs/backtest/benchmarks.csv
    outputs/backtest/summary.json
"""

from email.policy import default
from ipaddress import summarize_address_range
from torch import log_
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np 
import pandas as pd

from .regime_rules import identify_regime, REGIME_INT_TO_NAME
from .overlay import PortfolioOverlay
from ..evaluation.economic_metrics import compute_economic_metrics, compute_benchmark_metrics

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    position_history: pd.DataFrame
    benchmarks: dict[str, pd.DataFrame]
    summary: dict[str, float]
    benchmark_summary: dict[str, dict]

class BacktestEngine:

    def __init__(self, cfg: dict, output_dir: str | None = None):
        self.cfg =cfg
        self.overlay = PortfolioOverlay(cfg)
        self.out_root = Path(output_dir or '.') / 'outputs' / 'backtest'
        self.out_root.mkdir(parents=True, exist_ok=True)

        ev_cfg = cfg.get('backtest', {}).get('evaluation',{})
        self.ann_factor = int(ev_cfg.get('trading_days_per_year', 252))
        self.rfr = float(ev_cfg.get('risk_free_rate', 0.0))

    def run(
        self,
        predictions_df: pd.DataFrame,
        ohclv_df : pd.DataFrame,
        vix_series: pd.Series | None = None,
        initial_equity: float = 1.0,
    ) -> BacktestResult:
        """
        Run the full backtest simulation.

        Args:
            predictions_df: loaded from outputs/predictions/fold_*_test_preds.parquet
                            Required columns: date, rv_pred, p_regime_0..5
                            Optional: tail_prob
            ohlcv_df:       OHLCV DataFrame indexed by date string or datetime
                            Columns: open, high, low, close, volume
                            Must cover predictions period + 250 days of warmup
            vix_series:     VIX close values indexed by date (optional)
            initial_equity: starting portfolio value

        Returns:
            BacktestResult with equity curve, position history, benchmarks, summary
        """
        ohclv_df = ohclv_df.copy()
        ohclv_df.index = pd.to_datetime(ohclv_df.index)
        pred_dates = sorted(pd.to_datetime(ohclv_df.index))
        p_regex = [c for c in predictions_df.columns if c.startswith('p_regime_')]

        logger.info("Backtesting %d prediction dates...", len(pred_dates))

        # simulation loop
        equity, w_prev = initial_equity, 0.0
        eq_records, pos_records = [], []

        for dt in pred_dates:
            row = predictions_df[pd.to_datetime(predictions_df['date']) == dt]
            if len(row) == 0:
                continue
            row = row.iloc[0]

            # OHLCV window for regime identification
            loc = ohclv_df.index.get_indexer([dt], method='nearest')[0]
            if loc < 250:
                # not enough history for 200-day MA
                eq_records.append({
                    'date': dt, 'equity': equity, 'weight': 0.0,
                    'strategy_ret': 0.0, 'cost': 0.0, 'drawdown': 0.0
                })
                continue

            win = ohclv_df.iloc[max(0, loc- 250): loc+1]
            h = win['high'].values
            l = win['low'].values
            c = win['close'].values

            vix_win = None
            if vix_series is not None:
                try:
                    vix_win = vix_series.reindex(win.index, method='nearest').values
                except Exception:
                    pass

            _, signals = identify_regime(h, l, c, self.cfg, vix=vix_win)

            # model regime probs
            regime_probs = np.array(
                [float(row.get(f'p_regime_{k}', 1/6)) for k in range(6)],
                dtype=np.float32
            )
            regime_probs /= regime_probs.sum()

            overlay_out = self.overlay.compute(
                log_rv_pred= float(row['rv_pred']),
                regime_probs= regime_probs,
                signals= signals,
            )
            w_target = overlay_out['weight']

            # skip rebalance if change is below threshold
            if not self.overlay.should_rebalance(w_prev, w_target):
                w_target = w_prev

            # transaction cost
            cost = self.overlay.transaction_cost(w_prev, w_target)

            # daily return: position set at today's close, return next day
            if loc + 1 < len(ohclv_df):
                next_ret = float(np.log(
                    ohclv_df['close'].iloc[loc+1] / ohclv_df['close'].iloc[loc]
                ))
            else:
                next_ret = 0.0

            strategy_ret = w_prev * next_ret - cost
            equity *= (1.0 + strategy_ret)
            w_prev = w_target

            eq_records.append({
                "date": dt,
                "equity": equity,
                "weight": w_target,
                "underlying_ret": next_ret,
                "strategy_ret": strategy_ret,
                "cost": cost,
                "drawdown": 0.0,
                "regime_name": signals["regime_name"],
                "sigma_hat": overlay_out["sigma_hat_ann"],
                "p_crisis": overlay_out["p_crisis"],
            })
            pos_records.append({**overlay_out, 'date':dt, **signals})

        ec_df = pd.DataFrame(eq_records).set_index('date')
        pos_df = pd.DataFrame(pos_records)

        if len(ec_df) == 0:
            logger.warning("No backtest records - check prediction dates vs OHLCV range.")
            return BacktestResult(ec_df, pos_df, {}, {}, {})

        # drawdown
        rolling_max = ec_df['equity'].cummax()
        ec_df['drawdown'] = (ec_df['equity'] - rolling_max) / rolling_max

        # strategy summary
        summary = compute_economic_metrics(
            returns = ec_df['strategy_ret'].values,
            equity = ec_df['equity'].values,
            weights = ec_df['weights'].values,
            sigma_hat = ec_df['sigma_hat'].values,
            sigma_target= self.overlay.sigma_target,
            risk_free_rate= self.rfr,
            ann_factor= self.ann_factor,
        )

        # per regime performance
        regime_summary = {}
        for rname in REGIME_INT_TO_NAME.values():
            mask = ec_df['regime_name'] == rname
            subset = ec_df[mask]['strategy_ret'].dropna()
            if len(subset) < 5:
                regime_summary[rname] = {'n': int(mask.sum()), 'sharpe': float('nan')}
                continue
            from ..evaluation.economic_metrics import sharpe_ratio
            regime_summary[rname] = {
                'n': int(mask.sum()),
                'sharpe': round(sharpe_ratio(subset.values, self.rfr, self.ann_factor),3),
                'avg_weight': round(float(ec_df[mask]['weight'].mean()), 4)
            }
        summary['per_regime'] = regime_summary

        # benchmarks
        # get underlying returns aligned with backtest dates
        undl_idx = ohclv_df.index.get_indexer(ec_df.index, method='nearest')
        undl_rets = np.array([
            float(np.log(ohclv_df['close'].iloc[i+1] / ohclv_df['close'].iloc[i])) if i + 1 < len(ohclv_df) else 0.0 for i in undl_idx
        ])
        bmark = compute_benchmark_metrics(undl_rets, self.overlay.sigma_target, self.rfr, self.ann_factor)

        # save
        ec_df.to_csv(self.out_root / 'equity_curve.csv')
        pos_df.to_csv(self.out_root / 'position_history.csv', index=False)

        with open(self.out_root / 'summary_json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        with open(self.out_root / 'benchmark_summary.json', 'w') as f:
            json.dump(bmark, f, indent=2, default=str)

        logger.info(
            "Backtest done | Sharpe=%.2f | MaxDD=%.1f%% | AnnReturn=%.1f%% | Turnover=%.1f",
            summary.get("sharpe", float("nan")),
            summary.get("max_drawdown_pct", float("nan")),
            summary.get("ann_return_pct", float("nan")),
            summary.get("turnover_ann", float("nan")),
        )
        return BacktestResult(
            equity_curve = ec_df,
            position_history = pos_df,
            benchmarks = bmark,
            summary = summary,
            benchmark_summary = bmark,
        )