"""
Full portfolio simulation engine — multi-asset.

Simulates the SurfaceAlpha strategy day-by-day using:
    - Model predictions from fold parquets (per date, per symbol)
    - RegimeIdentifier on SPY OHLCV window (macro regime / ADX signals)
    - Per-symbol PortfolioOverlay using each symbol's own rv_pred + regime_probs
    - Equal-weight portfolio return across active symbols each day
    - Transaction costs per trade per symbol

Benchmarks:
    buy_and_hold:   equal-weight 100% in all symbols, no rebalancing
    inverse_vol:    per-symbol vol-targeting without model, equal-weight aggregate

Outputs:
    outputs/backtest/equity_curve.csv
    outputs/backtest/position_history.csv
    outputs/backtest/benchmark_summary.json
    outputs/backtest/summary.json
"""
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
        self.cfg = cfg
        self.overlay = PortfolioOverlay(cfg)
        self.out_root = Path(output_dir or '.') / 'outputs' / 'backtest'
        self.out_root.mkdir(parents=True, exist_ok=True)

        ev_cfg = cfg.get('backtest', {}).get('evaluation', {})
        self.ann_factor = int(ev_cfg.get('trading_days_per_year', 252))
        self.rfr = float(ev_cfg.get('risk_free_rate', 0.0))

    def _compute_rolling_betas(
        self, ohlcv_dict: dict[str, "pd.DataFrame"], window: int = 60
    ) -> dict[str, "pd.Series"]:
        """Compute rolling beta vs SPY for each symbol using log returns.

        Returns {symbol: pd.Series(beta, index=ohlcv.index)}.
        Symbols without sufficient history default to beta=1.0.
        SPY itself returns beta=1.0 by definition.
        """
        if 'SPY' not in ohlcv_dict:
            logger.warning("SPY not in ohlcv_dict — beta-weighted shorts disabled")
            return {}

        spy_ret = np.log(
            ohlcv_dict['SPY']['close'] / ohlcv_dict['SPY']['close'].shift(1)
        )
        spy_var = spy_ret.rolling(window).var()

        betas: dict[str, pd.Series] = {}
        for sym, ohlcv in ohlcv_dict.items():
            sym_ret = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
            cov = sym_ret.rolling(window).cov(spy_ret)
            beta = (cov / spy_var.reindex(cov.index)).fillna(1.0).clip(-3.0, 5.0)
            betas[sym] = beta
        return betas

    def _apply_beta_short(self, w_target: float, sym: str, dt, rolling_betas: dict) -> float:
        """Scale a short weight by rolling beta; symbols below min_beta go flat.

        Only called when w_target < 0 (short regime) and beta_short is enabled.
        """
        bs_cfg = self.cfg.get('backtest', {}).get('beta_short', {})
        min_beta = float(bs_cfg.get('min_beta_to_short', 1.0))
        max_mult = float(bs_cfg.get('max_beta_multiplier', 2.0))

        if sym not in rolling_betas:
            return w_target

        beta_series = rolling_betas[sym]
        loc = beta_series.index.get_indexer([dt], method='nearest')[0]
        if loc < 0:
            return w_target
        beta_val = float(beta_series.iloc[loc])

        if beta_val < min_beta:
            return 0.0  # flat — counter-cyclical or low-beta name

        # scale base short by beta, cap at max_mult × base
        scaled = w_target * min(beta_val, max_mult)
        return float(np.clip(scaled, self.overlay.w_min, 0.0))

    def _calibrate_rv(self, log_rv_pred: float) -> float:
        """Apply post-hoc MZ calibration to a log-RV prediction.

        Maps raw model output to the best linear predictor of actual log-RV
        using Mincer-Zarnowitz coefficients from backtest config. For typical
        values (rv_pred ~ -3.0) this produces a lower vol estimate, increasing
        overlay weights in calm regimes and reducing return drag.
        """
        cal = self.cfg.get('backtest', {}).get('calibration', {})
        if not cal.get('enabled', False):
            return log_rv_pred
        return float(cal['intercept']) + float(cal['slope']) * log_rv_pred

    def run(
        self,
        predictions_df: pd.DataFrame,
        ohlcv_dict: dict[str, pd.DataFrame],
        vix_series: pd.Series | None = None,
        initial_equity: float = 1.0,
    ) -> BacktestResult:
        """
        Run the multi-asset backtest simulation.

        Args:
            predictions_df: loaded from fold_*_test_preds.parquet
                            Required columns: date, symbol, rv_pred, p_regime_0..5
            ohlcv_dict:     {symbol: OHLCV DataFrame indexed by datetime}
                            SPY must be present for macro regime signals.
            vix_series:     VIX close values indexed by datetime (optional)
            initial_equity: starting portfolio value
        """
        predictions_df = predictions_df.copy()
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])

        # macro regime signals come from SPY (or first available symbol)
        macro_sym = 'SPY' if 'SPY' in ohlcv_dict else next(iter(ohlcv_dict))
        macro_ohlcv = ohlcv_dict[macro_sym]

        symbols = sorted(ohlcv_dict.keys())
        pred_dates = sorted(predictions_df['date'].unique())
        logger.info("Multi-asset backtest: %d symbols, %d prediction dates",
                    len(symbols), len(pred_dates))

        # precompute rolling betas for beta-weighted short sizing
        bs_cfg = self.cfg.get('backtest', {}).get('beta_short', {})
        beta_window = int(bs_cfg.get('window', 60))
        rolling_betas = (
            self._compute_rolling_betas(ohlcv_dict, window=beta_window)
            if bs_cfg.get('enabled', False) else {}
        )

        equity = initial_equity
        w_prev = {sym: 0.0 for sym in symbols}   # per-symbol previous weight
        eq_records, pos_records = [], []

        for dt in pred_dates:
            date_preds = predictions_df[predictions_df['date'] == dt]
            if len(date_preds) == 0:
                continue

            # macro signals for ADX override (SPY window)
            loc_macro = macro_ohlcv.index.get_indexer([dt], method='nearest')[0]
            signals = None
            if loc_macro >= 250:
                win = macro_ohlcv.iloc[max(0, loc_macro - 250): loc_macro + 1]
                vix_win = None
                if vix_series is not None:
                    try:
                        vix_win = vix_series.reindex(win.index, method='nearest').values
                    except Exception:
                        pass
                _, signals = identify_regime(
                    win['high'].values, win['low'].values, win['close'].values,
                    self.cfg, vix=vix_win,
                )

            # per-symbol overlay
            sym_strategy_rets, sym_weights, sym_underlying_rets = {}, {}, {}

            for _, pred_row in date_preds.iterrows():
                sym = pred_row['symbol']
                if sym not in ohlcv_dict:
                    continue
                sym_ohlcv = ohlcv_dict[sym]
                loc = sym_ohlcv.index.get_indexer([dt], method='nearest')[0]
                if loc + 1 >= len(sym_ohlcv):
                    continue

                regime_probs = np.array(
                    [float(pred_row.get(f'p_regime_{k}', 1 / 6)) for k in range(6)],
                    dtype=np.float32,
                )
                regime_probs /= regime_probs.sum()

                macro_regime = signals['regime_name'] if signals else None
                overlay_out = self.overlay.compute(
                    log_rv_pred=self._calibrate_rv(float(pred_row['rv_pred'])),
                    regime_probs=regime_probs,
                    signals=signals,
                    macro_regime_name=macro_regime,
                )
                w_target = overlay_out['weight']

                # beta-weighted short: scale per-symbol when in a short regime
                if w_target < 0 and rolling_betas:
                    w_target = self._apply_beta_short(w_target, sym, dt, rolling_betas)

                if not self.overlay.should_rebalance(w_prev[sym], w_target):
                    w_target = w_prev[sym]

                cost = self.overlay.transaction_cost(w_prev[sym], w_target)
                next_ret = float(np.log(
                    sym_ohlcv['close'].iloc[loc + 1] / sym_ohlcv['close'].iloc[loc]
                ))
                strategy_ret = w_prev[sym] * next_ret - cost

                sym_strategy_rets[sym] = strategy_ret
                sym_underlying_rets[sym] = next_ret
                sym_weights[sym] = w_target
                w_prev[sym] = w_target

            if not sym_strategy_rets:
                continue

            # equal-weight portfolio aggregation
            portfolio_ret = float(np.mean(list(sym_strategy_rets.values())))
            avg_weight = float(np.mean(list(sym_weights.values())))
            avg_undl_ret = float(np.mean(list(sym_underlying_rets.values())))
            equity *= (1.0 + portfolio_ret)

            regime_name = signals['regime_name'] if signals else 'unknown'
            p_crisis = float(np.mean([
                self.overlay.compute(
                    self._calibrate_rv(float(r['rv_pred'])),
                    np.array([float(r.get(f'p_regime_{k}', 1/6)) for k in range(6)], dtype=np.float32),
                )['p_crisis']
                for _, r in date_preds[date_preds['symbol'].isin(sym_strategy_rets)].iterrows()
            ]))
            sigma_hat = float(np.mean([
                self.overlay.compute(
                    self._calibrate_rv(float(r['rv_pred'])),
                    np.array([float(r.get(f'p_regime_{k}', 1/6)) for k in range(6)], dtype=np.float32),
                )['sigma_hat_ann']
                for _, r in date_preds[date_preds['symbol'].isin(sym_strategy_rets)].iterrows()
            ]))

            eq_records.append({
                'date': dt,
                'equity': equity,
                'weight': avg_weight,
                'underlying_ret': avg_undl_ret,
                'strategy_ret': portfolio_ret,
                'cost': float(np.mean([
                    self.overlay.transaction_cost(0.0, w) for w in sym_weights.values()
                ])),
                'drawdown': 0.0,
                'regime_name': regime_name,
                'sigma_hat': sigma_hat,
                'p_crisis': p_crisis,
                'n_symbols': len(sym_strategy_rets),
            })
            pos_records.append({
                'date': dt,
                'regime_name': regime_name,
                'n_symbols': len(sym_strategy_rets),
                **{f'w_{sym}': sym_weights.get(sym, float('nan')) for sym in symbols},
                **{f'ret_{sym}': sym_underlying_rets.get(sym, float('nan')) for sym in symbols},
            })

        ec_df = pd.DataFrame(eq_records).set_index('date')
        pos_df = pd.DataFrame(pos_records)

        if len(ec_df) == 0:
            logger.warning("No backtest records — check prediction dates vs OHLCV range.")
            return BacktestResult(ec_df, pos_df, {}, {}, {})

        # drawdown
        rolling_max = ec_df['equity'].cummax()
        ec_df['drawdown'] = (ec_df['equity'] - rolling_max) / rolling_max

        # strategy summary
        summary = compute_economic_metrics(
            returns=ec_df['strategy_ret'].values,
            equity=ec_df['equity'].values,
            weights=ec_df['weight'].values,
            sigma_hat=ec_df['sigma_hat'].values,
            sigma_target=self.overlay.sigma_target,
            risk_free_rate=self.rfr,
            ann_factor=self.ann_factor,
        )

        # per-regime performance
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
                'sharpe': round(sharpe_ratio(subset.values, self.rfr, self.ann_factor), 3),
                'avg_weight': round(float(ec_df[mask]['weight'].mean()), 4),
            }
        summary['per_regime'] = regime_summary

        # equal-weight benchmark returns (average underlying return across all symbols each day)
        undl_rets = ec_df['underlying_ret'].values
        bmark = compute_benchmark_metrics(undl_rets, self.overlay.sigma_target, self.rfr, self.ann_factor)

        # save
        ec_df.to_csv(self.out_root / 'equity_curve.csv')
        pos_df.to_csv(self.out_root / 'position_history.csv', index=False)

        with open(self.out_root / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        with open(self.out_root / 'benchmark_summary.json', 'w') as f:
            json.dump(bmark, f, indent=2, default=str)

        logger.info(
            "Backtest done | Sharpe=%.3f | MaxDD=%.1f%% | AnnReturn=%.1f%% | Symbols=%d",
            summary.get('sharpe', float('nan')),
            summary.get('max_drawdown_pct', float('nan')),
            summary.get('ann_return_pct', float('nan')),
            len(symbols),
        )
        return BacktestResult(
            equity_curve=ec_df,
            position_history=pos_df,
            benchmarks=bmark,
            summary=summary,
            benchmark_summary=bmark,
        )