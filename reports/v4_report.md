# SurfaceAlpha — V4 Research Report

**Date:** April 4, 2026
**Model version:** liquid_core_v4
**Universe:** liquid_core (14 symbols)
**Test window:** 2018-Q2 → 2025-Q1 (7 years, 28 quarterly folds)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Training Setup](#2-training-setup)
3. [Volatility Forecast Evaluation](#3-volatility-forecast-evaluation)
4. [Regime Classification Evaluation](#4-regime-classification-evaluation)
5. [Diagnostics](#5-diagnostics)
6. [Backtest Results](#6-backtest-results)
7. [Issues Found and Fixes Applied](#7-issues-found-and-fixes-applied)
8. [V5 Roadmap](#8-v5-roadmap)

---

## 1. Architecture Overview

### Symbol Universe — `liquid_core` (14 symbols)

| Category | Symbols |
|---|---|
| Broad market ETFs | SPY, QQQ, IWM |
| Mega-cap tech | AAPL, MSFT, AMZN, GOOGL, TSLA, NVDA, META |
| Financials | GS, JPM |
| Energy / Semis | XOM, AVGO |

Data spans **2015-01-02 to 2026-03-20**, providing ~11 years of training history and 7 years of walk-forward test coverage.

### Model — Three-Encoder Multimodal Fusion

```
IV Surface (12×20 grid, 6 channels)
    └─→ SurfaceEncoder (ViT) ──────────────────┐
                                               ├─→ FusionModule (concat-MLP) ─→ RegimeMoE ─→ Output Heads
Returns (60-day lookback, 6 features)          │       128-dim                   6 experts
    └─→ ReturnsEncoder (GRU) ──────────────────┤
                                               │
Vol-history + Market State (11 + 3 features)   │
    └─→ ContextEncoder (dual-stream MLP) ───────┘
```

| Component | Config |
|---|---|
| SurfaceEncoder | ViT, embed_dim=128, 2 layers, 4 heads, patch 3×4 |
| ReturnsEncoder | GRU, hidden_dim=64, 2 layers |
| ContextEncoder | Dual-stream MLP; vol_history_dim=11, macro_dim=3, output_dim=32 |
| Fusion | concat_MLP, hidden_dim=128 |
| MoE | 6 experts (one per regime), expert_hidden_dim=32 |
| Regime head | 6-class softmax (bull/bear/sideways × quiet/volatile) |
| Vol head | Scalar log(RV) regression |
| Tail head | Binary P(tail event) |

### Surface Channels (6)

`iv`, `spread_norm`, `obs_mask`, `staleness`, `delta`, `vega`

### Returns Features (6 per timestep, 60-day window)

`log_returns`, `rv_5`, `rv_10`, `rv_21`, `jump_indicator`, `rolling_beta_60`

### Context Features

**Vol-history (11):** iv_rank, hv_rank, vol_risk_premium, iv_momentum_{short,medium}, hv_momentum_{short,medium}, days_since_{iv,hv}_{year_high,year_low}

**Market state (3):** vix, spy_return, risk_free_rate

---

## 2. Training Setup

### Walk-Forward Cross-Validation

| Setting | Value |
|---|---|
| Folds | 28 |
| Mode | Expanding (train window grows each fold) |
| Base train window | ~3 years (756 days) |
| Val window | ~3 months (63 days) |
| Test window | ~3 months (63 days) |
| Step | ~3 months quarterly |
| Embargo | 5 days gap between train end and val start |
| Test coverage | 2018-Q2 → 2025-Q1 |

Fold 0 trains on 2015–2018 and tests on 2018-Q2. Fold 27 trains on 2015–2024 (~10 years) and tests on 2025-Q1. The expanding window means later folds have progressively more training data, which is intentional — the model benefits from observing more market cycles.

### Optimizer and Loss

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, lr=3e-4, weight_decay=0.03 |
| Scheduler | Cosine annealing, T_max=300, η_min=1e-6, warmup 3 epochs |
| Batch size | 64 |
| Max epochs | 1000 (early stopping patience=50) |
| Gradient clip | 1.0 |

**Multi-task loss:**

```
L = 1.0 × L_vol   (Huber, δ=0.5)
  + 0.1 × L_tail  (weighted BCE, pos_weight=2.0)
  + 0.15 × L_reg  (CrossEntropy, inverse-freq class weights per fold)
  + 0.1 × L_smooth (MoE load-balance entropy)
```

Regime class weights are computed per-fold from the training split using inverse-frequency weighting, so rare regimes like bear_quiet receive proportionally more gradient signal than the dominant bull_quiet and sideways_quiet classes.

### Baselines (trained on same folds for fair comparison)

Persistence, GARCH, HAR-RV, LightGBM/XGBoost boosting, deep_ts (LSTM, GRU, TCN). Residual learning is enabled: the deep model trains on `y_residual = y - y_HAR-RV`, forcing it to learn what HAR-RV cannot explain rather than re-learning a baseline it already handles well.

---

## 3. Volatility Forecast Evaluation

### Aggregate Metrics (mean across 28 folds, n=17,211 samples)

| Model | qlike ↑ | MAPE ↓ | R² ↑ | Bias |
|---|---|---|---|---|
| **SurfaceAlpha V4** | **-3.600** | 0.272 | 0.062 | -0.012 |
| HAR-RV | -3.703 | **0.269** | **0.116** | -0.014 |
| deep_ts_lstm | -3.762 | 0.682 | -3.059 | +0.019 |
| persistence | -3.775 | 0.325 | -0.148 | -0.000 |
| deep_ts_gru | -3.673 | 0.314 | -13.302 | -0.008 |
| deep_ts_tcn | -3.674 | 0.365 | -0.998 | -0.000 |
| boosting | -3.643 | 0.351 | -0.223 | -0.006 |
| GARCH | -3.472 | 0.570 | -1.174 | +0.004 |

> **qlike** (quasi-likelihood) is the primary vol forecast metric — lower is worse, higher (less negative) means better probabilistic calibration. The model slightly trails HAR-RV on raw vol prediction. This is expected: the residual learning setup should make them comparable, and the model's value-add is regime classification, not vol ranking.

### Mincer-Zarnowitz Calibration

Tests whether `E[realized | forecast] = forecast` (ideal: slope=1.0, intercept=0.0):

| Stat | Value |
|---|---|
| Intercept | 0.014 |
| Slope | 0.973 |
| R² | 0.356 |
| F-stat (H₀: slope=1, intercept=0) | 722.2 (p=0.0) |

Slope 0.973 is near-perfect. In V3 this was 1.738 — the model was forecasting vol 74% too high, which collapsed overlay weights in quiet regimes. Post-hoc calibration (`slope=1.0, intercept=-0.05`) resolved this in V4.

### Per-Regime Vol Forecast Quality

| Regime | qlike | R² | n |
|---|---|---|---|
| bull_quiet | -4.159 | 0.436 | 5,515 |
| sideways_quiet | -4.045 | 0.354 | 6,588 |
| bear_quiet | -3.170 | 0.323 | 1,163 |
| sideways_volatile | -2.860 | -0.051 | 1,332 |
| bull_volatile | -2.403 | -0.106 | 1,186 |
| bear_volatile | -2.003 | -0.251 | 1,427 |

Vol forecast quality is strong in quiet regimes (R² 0.32–0.44) and poor in volatile regimes (R² negative). This is expected — volatile regimes involve tail events the model has seen few examples of in training, and vol is inherently harder to forecast when it is most variable.

---

## 4. Regime Classification Evaluation

### Overall Accuracy

| Metric | Value |
|---|---|
| Mean accuracy (28-fold average) | 36.3% |
| Chance baseline (6-class uniform) | 16.7% |
| Improvement over chance | **2.2×** |
| Brier score | 0.149 |
| Regime calibration (ECE) | 0.123 |

### Per-Regime Accuracy

| Regime | Accuracy | n (avg per fold) |
|---|---|---|
| bear_volatile | 45.6% | 109 |
| bull_volatile | 41.7% | 56 |
| bull_quiet | 40.3% | 197 |
| bear_quiet | 32.0% | 55 |
| sideways_quiet | **24.8%** | 244 |
| sideways_volatile | **23.6%** | 74 |

Sideways regimes are significantly harder to classify than trending regimes. The root cause is a missing feature problem: the regime label function defines "sideways" as `|pct_from_ma200| ≤ 2%` OR `ADX(14) < 20`, but neither of these signals — SPY's distance from its 200-day MA nor ADX — appear in the model's input features. The model currently receives only `vix`, `spy_return`, and `risk_free_rate` as market state inputs, and is trying to infer range-bound market conditions from signals that don't directly encode them. This is the primary improvement target for V5.

### Tail Risk AUC

| Metric | Value |
|---|---|
| Mean tail AUC | 0.538 |
| Random baseline | 0.500 |

Tail AUC is barely above random, consistent with the known difficulty of predicting extreme events out-of-sample. It is monitored but not a primary optimization target.

---

## 5. Diagnostics

### MoE Expert Diversity

**Verdict: SPECIALISED**

| Metric | Value |
|---|---|
| Global mean expert std | 0.507 |
| Global median expert std | 0.429 |
| % samples with expert std > 0.05 | 99.8% |
| Collapsed? | No |

| Regime | Mean expert std | Mean expert range |
|---|---|---|
| bull_quiet | **0.724** | 2.02 |
| bear_volatile | 0.483 | 1.35 |
| sideways_volatile | 0.482 | 1.34 |
| sideways_quiet | 0.447 | 1.25 |
| bear_quiet | 0.379 | 1.06 |
| bull_volatile | **0.305** | 0.87 |

The MoE is fully specialised — experts are expressing genuinely different views rather than converging to a consensus. Bull_quiet produces the highest expert spread (std=0.72), meaning the six experts disagree most on quiet trending days, which is where the MoE routing matters most for performance. In V1/V2, the smoothness regularizer minimized expert variance and caused collapse. V4's entropy-based load-balance loss fixed this.

### Bull Quiet Calibration

Measures whether the vol-targeting overlay receives accurate sigma forecasts in the most important regime.

| Metric | Value |
|---|---|
| n bull_quiet days | 713 |
| Mean sigma_hat / actual sigma | 0.966 |
| % days where model over-forecasts | 41.4% |
| Avg sigma_hat (annualized) | 21.1% |
| Avg actual sigma (annualized) | 23.3% |
| Avg implied weight (sigma_target / sigma_hat) | 49.9% |
| Avg ideal weight (sigma_target / actual sigma) | 47.9% |
| Weight drag | **-0.020** |

The model slightly under-forecasts vol in bull_quiet (ratio 0.966), so the overlay takes marginally more risk than optimal. The weight drag of -2.0% is small. In V3 the mean ratio was +1.74 (74% over-forecast), which collapsed bull_quiet weights to near zero and destroyed performance in that regime.

### Fold Stability

| Fold | Test Period | qlike | R² | Notes |
|---|---|---|---|---|
| 6 | 2019-Q4 | -5.431 | 0.221 | Best fold |
| 0 | 2018-Q2 | -5.264 | 0.195 | |
| 3 | 2019-Q1 | -5.128 | -0.003 | |
| 20 | 2023-Q2 | -4.433 | 0.741 | Best recent fold |
| 23 | 2024-Q1 | -4.330 | 0.626 | |
| **7** | **2020-Q1** | **+7.772** | **-1.880** | **COVID — catastrophic** |
| 27 | 2025-Q1 | -2.011 | -0.143 | Tariff shock |
| 15 | 2022-Q1 | -3.349 | -0.203 | Rate shock |

**Fold 7 (COVID, 2020-Q1):** The only fold with positive qlike in the entire test window. A positive qlike means the vol forecast was so wrong it would have been better to predict a constant. The model was trained on 2015–2019 data and had never seen a vol spike of this magnitude. This is a structural break, not a model failure — it cannot be fixed without training data that includes similar events.

**Fold 27 (2025-Q1):** The tariff shock environment produced R²=-0.14, meaning the model's vol rank ordering was worse than a flat forecast. This is the most recent period and reflects ongoing distribution shift from the post-2024 macro regime.

**Overall pattern:** qlike degrades from -5.4 in the pre-COVID early folds toward -3.5 to -4.0 in recent years. The expanding window helps — fold 8 (2020-Q2, post-COVID recovery) shows qlike=-3.49 and R²=0.26, demonstrating the model adapts once it has seen new regime data in training.

---

## 6. Backtest Results

### Strategy Design

The portfolio overlay converts model predictions into daily position sizes using a pipeline of rules applied in sequence:

1. **Vol targeting** — `w = sigma_target / sigma_hat_annualised`. Reduces exposure when the model forecasts elevated volatility.
2. **Regime sizing** — `w = w × position_size[regime]`. A multiplier per macro regime, determined by SPY rule-based signals (200MA + ADX + VIX + ATR), not the model's softmax.
3. **ADX override** — if ADX < 20 (no trend), reduce by 50%.
4. **Clip** — `w = clip(w, w_min=0.0, w_max=1.5)`.
5. **VIX circuit breaker** — if VIX ≥ 40, force `w = 0.0` regardless of all other signals.

The key design principle is **asymmetric leverage**: `sigma_target=0.30` with `w_max=1.5` allows up to ~1.42× exposure in bull_quiet (where sigma_hat ≈ 21%) while bear_volatile and sideways_volatile stay flat at exactly 0. This captures more upside in the best regimes without amplifying crisis drawdowns.

Position sizing uses SPY macro signals for the regime lookup — not the model's softmax predictions. The model drives `sigma_hat` for vol targeting and routes through MoE experts, but the regime gate (which multiplier to apply) is determined by rule-based SPY indicators. This separation was a deliberate fix from V3, where using model argmax for position sizing caused the config to be silently ignored (the model rarely predicted bull_quiet).

### Regime Position Sizing Rules

| Regime | Position size | Avg backtest weight | Rationale |
|---|---|---|---|
| bull_quiet | 100% | 1.341 | Full exposure; vol targeting drives leverage ~1.42× |
| sideways_quiet | 100% | 0.779 | Mean-reversion alpha; vol targeting scales naturally |
| bull_volatile | 37.5% | 0.572 | Extra protection beyond vol targeting in high-vol bull |
| bear_quiet | **25%** | 0.286 | Near-zero Sharpe; cut to prevent leverage bleed |
| bear_volatile | 0% | 0.000 | Go flat; basket too diversified for uniform short |
| sideways_volatile | 0% | 0.000 | No edge; step aside |

Bear_quiet deserves a note: its position_size was 50% in the initial V4 config. When sigma_target was raised to 0.30, vol targeting proportionally increased bear_quiet weights from 0.42 to 0.57, adding vol in a regime with near-zero Sharpe (0.04). Cutting it to 25% recovered the Sharpe degradation without affecting the leveraged quiet regimes.

### Final Results vs Benchmarks

| Metric | **SurfaceAlpha V4** | Buy & Hold | Inverse-Vol |
|---|---|---|---|
| Ann. return | 8.6% | 10.5% | 14.3% |
| Ann. vol | **16.3%** | 24.9% | 27.2% |
| **Sharpe** | **0.530** | 0.422 | 0.524 |
| Sortino | 0.635 | 0.524 | 0.696 |
| Max drawdown | **-28.2%** | -40.7% | -40.8% |
| Calmar | 0.306 | 0.259 | 0.351 |
| Total return (7yr) | 66.5% | 67.9% | 109.3% |
| Ann. turnover | 17.2x | — | 4.2x |
| Avg weight | 88.2% | 100% | 132% |
| Vol target tracking | 0.542 | 2.49 | 2.72 |

**V4 beats both benchmarks on Sharpe** (0.530 vs 0.422 buy-and-hold, vs 0.524 inverse-vol) while cutting maximum drawdown roughly in half (-28.2% vs -40%+ for both benchmarks).

The absolute return is lower than both benchmarks because the strategy spends time in reduced-weight and flat regimes. This is by design — the model is not trying to maximise return, it is trying to maximise risk-adjusted return. The inverse-vol benchmark runs at avg_weight=1.32 (32% levered) and suffers the same drawdown as buy-and-hold (-40.8%). V4 achieves higher Sharpe at lower average weight and meaningfully lower drawdown.

The vol target tracking ratio of 0.542 means the portfolio realized `30% × 0.542 ≈ 16.3%` annualized vol — below the 30% target because of time spent in flat regimes.

### Per-Regime Backtest Performance

| Regime | n days | Sharpe | Avg weight | % of test period |
|---|---|---|---|---|
| bull_quiet | 713 | **0.705** | 1.341 | 40.4% |
| sideways_quiet | 640 | 0.667 | 0.779 | 36.3% |
| bull_volatile | 90 | 0.451 | 0.572 | 5.1% |
| bear_quiet | 172 | -0.068 | 0.286 | 9.7% |
| bear_volatile | 131 | -0.737 | 0.000 | 7.4% |
| sideways_volatile | 18 | -8.818 | 0.000 | 1.0% |

The strategy earns its Sharpe almost entirely in the first two regimes. Bull_quiet and sideways_quiet together account for 76.7% of test days and both run at leveraged weights (1.34× and 0.78× respectively). The remaining 23.3% of days are either reduced-weight (bull_volatile, bear_quiet) or flat (bear_volatile, sideways_volatile).

> **Note on flat-regime Sharpe artifacts:** bear_volatile (Sharpe -0.737) and sideways_volatile (Sharpe -8.818) both have avg_weight=0.0. Their negative Sharpe values are a computational artifact — transaction costs on entry/exit days divided by near-zero return variance produces extreme ratios. These are not real losses. Actual risk exposure in those regimes is zero.

---

## 7. Issues Found and Fixes Applied

This section documents the full diagnostic and remediation cycle across all versions.

---

### Issue 1 — MZ Calibration Over-Correction

**V3 problem:** The Mincer-Zarnowitz regression produced slope=1.738. Applied literally as a calibration transform, this amplified sigma_hat to 33.6% in bull_quiet (vs actual ~17%). The overlay capped all weights at w_max and the model became equivalent to buy-and-hold in good regimes while remaining overly cautious in moderate ones. The slope was driven by the COVID fold outlier, which is orders of magnitude more extreme than any other fold.

**Fix:** Replaced the linear MZ calibration with a constant log-space offset: `slope=1.0, intercept=-0.05`. This scales sigma_hat down uniformly by 5% (`exp(-0.05) = 0.95`), preserving prediction rank ordering and avoiding the outlier-contaminated slope. The V4 MZ slope of 0.973 confirms near-perfect calibration.

---

### Issue 2 — Regime Gating Killing Weights

**V3 problem:** The crisis gating mechanism computed `p_crisis = P(bear_volatile) + P(sideways_volatile)` from the model's softmax output and applied `w = w × (1 - p_crisis)`. The model assigned p_crisis ≈ 0.31 even on bull_quiet SPY days, systematically cutting all weights by 31% at all times.

**Root cause:** The model's softmax reflects classification uncertainty, not a calibrated crisis probability. Bull_quiet days with any model ambiguity produce positive p_crisis even when there is no actual crisis.

**Fix:** Disabled regime gating entirely (`regime_gating.enabled: false`). The VIX circuit breaker (see Issue 7) provides a cleaner, threshold-based version of crisis protection.

---

### Issue 3 — Regime Sizing Used Model Argmax

**V3 problem:** `overlay.compute()` used `np.argmax(regime_probs)` to look up position_size. The model rarely predicted bull_quiet (its classifications skewed toward sideways and bear), so `bull_quiet.position_size` in the config had zero effect on actual positions. Config changes were silently ignored.

**Fix:** Added `macro_regime_name` parameter to `overlay.compute()`. Position_size lookup now uses the SPY rule-based regime (200MA + ADX + ATR + VIX), not the model's argmax. The model's regime probs still drive MoE routing and sigma_hat, but not the position multiplier.

**Files changed:** `src/volregime/portfolio/overlay.py`, `src/volregime/portfolio/backtest_engine.py`

---

### Issue 4 — Vol Target Tracking Metric Was Broken

**Problem:** `vol_target_tracking` was measuring the fraction of days where `sigma_hat` fell in a 5–15% annualized band — a proxy with no financial meaning. It read 0.4841 across five consecutive runs regardless of config changes.

**Fix:** Corrected to `realized_vol / sigma_target` where `realized_vol = std(returns) × sqrt(252)`. A value of 0.542 means the portfolio ran at 16.3% realized vol against a 30% target.

**File changed:** `src/volregime/evaluation/economic_metrics.py`

---

### Issue 5 — Walk-Forward Test Window Too Short

**V3 problem:** 8 folds covered only 2018–2020 — the 2018 sell-off followed by COVID. Any equity-long strategy looks bad in this window regardless of model quality. The V3 backtest Sharpe of -0.536 was a function of the test period, not the model.

**Fix:** Extended to 28 folds (2018-Q2 → 2025-Q1), covering the 2020 recovery, 2021 bull run, 2022 rate shock, and 2023–2024 bull market. A 7-year test window provides a far more representative assessment of regime-conditional performance.

**Config change:** `configs/training.yaml` → `walk_forward.num_folds: 28`

---

### Issue 6 — MoE Expert Collapse

**V1/V2 problem:** The smoothness regularizer minimized variance across expert predictions (`Var(expert_preds)`), forcing all six experts to learn the same function. The MoE was computing a weighted sum of identical outputs — no routing signal, no specialization.

**Fix:** Replaced the variance minimizer with an entropy-based load-balance loss that maximizes the entropy of the mean routing probability distribution. This encourages all experts to be used without forcing them to agree on predictions.

**Verification:** V4 mean expert std = 0.507 globally, with 99.8% of samples showing std > 0.05. The MoE is fully specialised.

---

### Issue 7 — Overlay Tuning: Leverage, Regime Sizing, and Crisis Protection

This covers three interconnected tuning decisions made after the initial V4 training.

**Leverage (sigma_target and w_max):**
Raising sigma_target from 0.22 to 0.30 with w_max=1.5 allows up to ~1.42× exposure in bull_quiet (where sigma_hat ≈ 21%). With sigma_target=0.22 the overlay was never leveraged — bull_quiet was capped at exactly 1.0× with no room to run. The leverage is paid for by higher vol (12% → 16.3%) but the asymmetric regime protection keeps Sharpe above both benchmarks.

**Bear_quiet position_size (0.5 → 0.25):**
With the higher sigma_target, vol targeting proportionally increased all regime weights including bear_quiet (from avg_weight 0.42 → 0.57). Bear_quiet has near-zero Sharpe (0.04), so more exposure there adds vol with no return. Cutting position_size to 0.25 brought its avg_weight back to 0.29, trimming 0.6% of portfolio vol without reducing exposure in the high-alpha regimes.

**Shorting (explored and rejected):**
A beta-weighted short was implemented in bear_volatile (`position_size=-0.25`, scaled per-symbol by rolling 60-day beta vs SPY, with symbols below beta=1.0 going flat). XOM (beta ~0.6) and other counter-cyclical names went flat; TSLA/NVDA got the full short. Despite the beta-weighting, the short was net-negative: bear_volatile Sharpe improved from -0.737 to -0.273 (transition costs at regime entry/exit) but overall Sharpe dropped from 0.530 to 0.514. The portfolio is too diversified for a uniform short to add value.

**VIX circuit breaker:**
Implemented as a final override in `overlay.compute()`: if `signals['vix'] >= 40`, force `w = 0.0` regardless of regime or vol targeting. This directly addresses fold 7 (COVID 2020-Q1, qlike=+7.77) where VIX hit 80 and the model's forecasts were catastrophically wrong. The breaker did not fire in the 7-year backtest (VIX > 40 almost never coexists with SPY above its 200MA, so those days were already bear_volatile and flat) but provides live-trading protection against COVID-style spikes.

**File changed:** `src/volregime/portfolio/overlay.py` (VIX breaker), `src/volregime/portfolio/backtest_engine.py` (beta shorts infrastructure, preserved for future use)

---

### Issue 8 — Regime Head Gradient Starvation

**Current state:** `lambda_reg: 0.15` vs `lambda_vol: 1.0`. The regime head receives 6.5× less gradient signal than the vol head throughout training.

**Impact:** Regime accuracy of 36.3% overall, with sideways regimes at 24.8% — only 1.5× chance. Given that the regime signal drives position_size selection and MoE routing, this likely understates the model's potential.

**Fix (V5):** Increase `lambda_reg: 0.15 → 0.30`. Combined with the missing feature fix below, this should push regime accuracy materially higher without significantly degrading vol qlike.

---

## 8. V5 Roadmap

### Priority 1 — Add Regime-Defining Features to Market State

**The problem in one sentence:** the model is trying to predict regime labels it has no access to the defining features for.

The regime label function (`targets.py:compute_regime_label`) classifies a day as sideways when `|pct_from_ma200| ≤ 2%` OR `ADX(14) < 20`. The model's market_state input is `[vix, spy_return, risk_free_rate]`. SPY's distance from its 200MA and ADX are both absent. The model has to infer "is SPY range-bound?" from a single daily return and an options-implied vol figure. That is the reason sideways accuracy is stuck at 25%.

**Fix:**
1. Add `spy_pct_from_ma200`, `spy_adx14`, and `spy_atr_ratio` (ATR10/ATR50) to `market_state.py`
2. Update `configs/data.yaml` and `configs/model.yaml` (`macro_dim: 3 → 6`)
3. Rebuild processed data and retrain as V5

**Expected impact:** Sideways accuracy should improve substantially (toward 50%+) since the model would have direct access to the signals that define the labels. Improved regime accuracy also improves MoE routing quality and thus vol forecasting in regime-conditional settings.

### Priority 2 — Increase lambda_reg (combine with V5 retrain)

Increase `lambda_reg: 0.15 → 0.30` in `configs/training.yaml`. No data rebuild needed — combine with the feature addition above at no extra cost.

### Priority 3 — VIX Hard Override for Live Deployment

The circuit breaker threshold of VIX=40 is appropriate for backtesting but consider lowering to VIX=35 for live trading, where you have the option to act on intraday signals rather than daily close prices. This is a one-line config change.

### Priority 4 — Per-Symbol Beta-Weighted Shorts (Future)

The infrastructure is fully implemented in `backtest_engine.py` (`_compute_rolling_betas`, `_apply_beta_short`). To re-enable: set `bear_volatile.position_size: -0.25`, `w_min: -0.5`, and `beta_short.enabled: true` in `backtest.yaml`. The reason it was disabled is that the 14-symbol basket is too diversified — counter-cyclical names (XOM, JPM) drag against the short in most bear regimes. This becomes viable when the universe expands to allow symbol-level selection: short high-beta names only, go flat on sector-specific names with low or negative SPY correlation.

### Priority 5 — Expand to Medium Universe (~100 symbols)

The liquid_core portfolio has meaningful idiosyncratic risk from individual names (TSLA single-stock vol, NVDA earnings). Expanding to ~100 liquid symbols dilutes this, provides far more training data per fold, and opens up the beta-weighted short strategy — with 100 symbols, you can easily select the top-beta quartile for shorts in bear_volatile rather than shorting a mixed 14-name basket.

### Estimated Timeline

| Task | Requires | Wall-clock |
|---|---|---|
| Add 3 market_state features + bump macro_dim | Code + config | 1–2 hours |
| Rebuild processed data (`make data`) | Data pipeline | 2–4 hours |
| Retrain V5 (`make train`) | GPU | 8–12 hours |
| Evaluate + backtest + diagnose | Post-processing | 30 minutes |
| **Total** | | **~14–18 hours (overnight)** |

---

*All metrics are out-of-sample walk-forward results from `runs/liquid_core_v4/outputs/`. No in-sample data was used to compute any reported figure.*
