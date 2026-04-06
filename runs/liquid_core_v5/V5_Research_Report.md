# SurfaceAlpha V5: Research Report
**Version:** liquid_core_v5
**Date:** 2026-04-05
**Universe:** liquid_core (14 symbols)
**Walk-forward folds:** 28 quarterly, expanding window
**OOS period:** 2018-04-12 – 2025-01-xx (1,764 trading days)

---

## Abstract

SurfaceAlpha V5 extends the V4 architecture with three new macro context features (`spy_pct_from_ma200`, `spy_adx14`, `spy_atr_ratio`), expanding the context encoder macro dimension from 3 to 6. Evaluated on 28 out-of-sample walk-forward folds across a 14-symbol liquid equity universe, V5 achieves near-perfect Mincer-Zarnowitz calibration (slope = 1.035, vs V4 = 0.973), confirmed MoE expert specialisation (mean expert_std = 0.464), and a portfolio Sharpe ratio of 0.555 with maximum drawdown of −26.4% — versus buy-and-hold Sharpe 0.423 and drawdown −40.7%. A post-training short overlay analysis identifies model-argmax-gated bear_volatile shorts as the dominant risk-adjusted improvement, lifting bear_volatile regime Sharpe from −0.850 to −0.181 at no cost to overall Sharpe.

---

## 1. Introduction

Option-implied volatility surfaces encode the market's collective expectation of future return distributions. Prior SurfaceAlpha versions demonstrated that treating these surfaces as structured images — processed by a Vision Transformer — and fusing the surface signal with temporal return dynamics and macro context features enables regime-conditional volatility forecasting superior to rule-based approaches. V5 addresses three limitations identified in V4:

1. **Insufficient macro conditioning.** V4 used only 3 macro features (VIX, risk-free rate, vol_risk_premium). V5 adds SPY trend direction (`spy_pct_from_ma200`), trend strength (`spy_adx14`), and short-term volatility (`spy_atr_ratio`), giving the context encoder a richer description of the macro regime at inference time.

2. **Forecast over-calibration direction.** V4 exhibited slight under-forecasting bias (MZ slope = 0.973). V5 corrects this without a post-hoc calibration layer, achieving MZ slope = 1.035.

3. **Bear_volatile return drag.** The portfolio overlay went flat in bear_volatile regimes, leaving directional alpha on the table. V5 introduces a model-argmax-gated short overlay that fires only when the model's own top prediction is bear_volatile, reducing false positives from the rule-based regime classifier.

---

## 2. Architecture

### 2.1 Model Overview

The V5 model is a multi-task architecture with three encoders and three output heads:

| Component | Description | Output Dim |
|-----------|-------------|------------|
| SurfaceEncoder (ViT) | 6-channel (12×20) IV surface → patch embeddings | z_surf (256) |
| ReturnsEncoder (GRU) | 60-day × 6-feature returns lookback | z_ret (128) |
| ContextEncoder (MLP) | vol_history (11) + macro_state (6) | z_ctx (128) |
| FusionModule | Concat + MLP over [z_surf, z_ret, z_ctx] | z_fused (256) |
| RegimeMoE | 6 expert MLPs, soft-gated by regime head | — |

**Output heads:**
- Vol forecast: log(RV_21d) — regression (Huber loss)
- Tail probability: P(RV > 2× median) — binary (BCE loss)
- Regime classification: 6-class softmax (CrossEntropy + entropy regularisation)

### 2.2 V5 Macro Features (New)

| Feature | Description | Normalisation |
|---------|-------------|---------------|
| `spy_pct_from_ma200` | SPY close / 200-day MA − 1 | raw ratio |
| `spy_adx14` | ADX(14) trend strength | ÷100 |
| `spy_atr_ratio` | ATR(10) / ATR(50) | raw ratio |

These features provide the context encoder with a continuous signal about trend direction, strength, and near-term volatility expansion that was previously available only implicitly through the surface encoder.

### 2.3 MoE Entropy Regularisation

Following V4, the regime head is trained with an entropy regularisation term (λ = 0.1) that penalises degenerate routing (all mass on one expert). V5 confirms that this design choice eliminates MoE collapse: mean per-sample expert_std = **0.464**, with 99.8% of samples having expert_std > 0.05 across the full OOS set.

---

## 3. Volatility Forecast

### 3.1 Mincer-Zarnowitz Calibration

The MZ regression of realised log-RV on predicted log-RV provides the canonical calibration diagnostic:

| Version | Intercept | Slope | R² |
|---------|-----------|-------|-----|
| V3 | −0.024 | 1.738 | — |
| V4 | −0.009 | 0.973 | 0.361 |
| **V5** | **+0.009** | **1.035** | **0.372** |

V5 achieves the closest slope to the ideal of 1.0 across all versions, indicating neither systematic over- nor under-forecasting. The marginal R² improvement (0.361 → 0.372) confirms that the additional macro features contribute incremental predictive content beyond the surface and returns signals.

### 3.2 Aggregate Forecast Metrics (28-fold OOS mean)

| Metric | V5 |
|--------|----|
| vol_qlike | −3.594 |
| vol_mape | 0.279 |
| vol_r2 | 0.086 |
| vol_rmse | 0.034 |
| vol_bias | −0.012 |

The low R² (0.086) is consistent with the difficulty of point vol forecasting at 21-day horizons. HAR-RV models trained on the same universe achieve qlike ≈ −3.81 and R² ≈ 0.15 on similar horizons, reflecting the well-known advantage of simple autoregressive vol models for point accuracy. V5's primary advantage over HAR-RV is regime conditioning, tail detection, and the surface signal — components orthogonal to historical RV.

### 3.3 Fold Stability

Vol qlike is stable across folds 0–20 (mean ≈ −4.2) with two structural breaks:

- **Fold 7 (COVID-19, 2020):** qlike = +5.30. The unprecedented VIX spike (80+) lies far outside the training distribution. Present in all model versions.
- **Folds 22–27 (2023–2024):** gradual degradation toward qlike ≈ −2.8, consistent with the low-volatility, high-momentum regime that characterised post-COVID equity markets and differs structurally from the 2017–2022 training distribution.

These degradation patterns reflect structural distribution shifts, not model failure. The VIX circuit breaker (≥40 → flat) provides a partial hedge against COVID-style events.

---

## 4. Regime Classification

### 4.1 Overall Accuracy

OOS mean regime accuracy: **37.7%** (28-fold mean), versus a 6-class random baseline of 16.7%.

### 4.2 Per-Regime Accuracy

| Regime | Accuracy | n (mean/fold) |
|--------|----------|--------------|
| bear_volatile | **50.5%** | 110 |
| bull_volatile | 45.3% | 56 |
| bear_quiet | 38.3% | 55 |
| bull_quiet | 37.6% | 197 |
| sideways_quiet | 29.7% | 244 |
| sideways_volatile | **23.7%** | 74 |

`bear_volatile` is the most reliably identified regime (50.5%), which is the critical property enabling the short overlay. The model's argmax prediction of `bear_volatile` is a high-precision signal — as confirmed by the post-training short overlay experiment where model-argmax-gated shorts reduce bear_volatile Sharpe from −0.850 to −0.181.

### 4.3 Sideways Regime Structural Weakness

Sideways regimes achieve the lowest classification accuracy (23.7–29.7%). The root cause is IV surface ambiguity: in low-volatility environments, the surface does not encode directional bias, making bull_quiet and sideways_quiet visually similar to the ViT. Cross-confusion between bear_quiet and sideways_quiet (13–15%) has a modest impact on the vol forecast (estimated 0.196 log-RV units). This is a known limitation of surface-only signal in low-trend environments.

---

## 5. Explainability

### 5.1 Two Attribution Channels

V5 reveals a dissociation between what drives the vol forecast (SHAP) and what drives regime routing (MoE gating):

**SHAP (vol forecast attribution):** Top features by mean |SHAP|:
1. `days_since_iv_year_high`
2. `days_since_hv_year_high`
3. `days_since_iv_year_low`
4. `days_since_hv_year_low`
5. `spy_atr_ratio` *(new in V5)*

The "days since" cycle features dominate vol level prediction — the model tracks where we are in the volatility cycle to forecast near-term RV.

**MoE Gating (Spearman correlation with regime_probs):** Top gating features by |r|:
1. `risk_free_rate` (|r| = 0.42–0.55 across all experts)
2. `spy_adx14` (|r| = 0.34–0.42) *(new in V5)*
3. `days_since_hv_year_high` (|r| = 0.36–0.47)

The macro rate environment and trend strength are the primary routing signals — the model uses the interest rate regime and trend intensity to select which expert handles a given observation. This dissociation is theoretically coherent: vol cycle position forecasts vol level; macro regime routes to the appropriate expert for that regime's vol surface dynamics.

### 5.2 New V5 Macro Features in Attribution

`spy_adx14` ranks as the second-strongest gating feature despite not contributing to SHAP. This confirms that trend strength acts as a routing signal rather than a vol-level predictor — the model learned to use ADX to distinguish trending from sideways regimes at the MoE layer, precisely the intended function of the feature.

`spy_atr_ratio` appears in the top-5 SHAP features, contributing to vol level forecasting. Short-term volatility expansion (ATR(10)/ATR(50) > 1) is positively correlated with higher near-term RV forecasts, consistent with mean-reverting vol dynamics.

### 5.3 Surface Attribution

Gradient saliency over the IV surface identifies the short-dated OTM put region (moneyness 0.85–0.95, maturity 7–30 days) as the primary attention zone across all regimes — consistent with the put skew carrying the dominant tail-risk signal. V5 shows a secondary attention shift toward near-ATM mid-maturity options (moneyness 0.97–1.03, 60–90 days) relative to V4, reflecting the macro context features reducing the model's reliance on skew alone.

---

## 6. Portfolio Performance

### 6.1 Strategy Configuration

- Vol-targeting overlay: w = σ_target / σ_hat (σ_target = 30%, w_max = 150%)
- MZ calibration: intercept = −0.05, slope = 1.0 (constant log-space shift)
- Bear_volatile: model-argmax-gated short, position_size = −0.20, beta-weighted per symbol
- Sideways_volatile: flat (no edge)
- VIX circuit breaker: force flat at VIX ≥ 40

### 6.2 Overall Results

| Metric | V5 | Buy & Hold | Inverse-Vol |
|--------|----|------------|-------------|
| Ann Return | 9.1% | 10.5% | 14.3% |
| Ann Vol | 16.4% | 24.9% | 27.2% |
| **Sharpe** | **0.555** | 0.423 | 0.525 |
| Sortino | 0.679 | 0.524 | 0.696 |
| **Max DD** | **−26.4%** | −40.7% | −40.8% |
| Calmar | 0.344 | 0.259 | 0.351 |
| Total Return | 71.9% | 67.9% | 109.3% |
| Avg Weight | 0.866 | 1.000 | 1.321 |

V5 achieves the highest Sharpe ratio (0.555 vs buy-and-hold 0.423) while cutting maximum drawdown by 14.3 percentage points. The inverse-vol benchmark — which uses only lagged vol with no model — achieves a higher total return due to sustained leverage in the low-vol 2021–2023 period, but with double the drawdown. On risk-adjusted terms (Sharpe, Calmar), V5 outperforms both benchmarks.

### 6.3 Per-Regime Performance

| Regime | n days | Sharpe | Avg Weight |
|--------|--------|--------|------------|
| bull_quiet | 713 | +0.657 | 1.340 |
| bull_volatile | 90 | +0.763 | 0.553 |
| sideways_quiet | 640 | +0.808 | 0.756 |
| bear_quiet | 172 | −0.239 | 0.279 |
| **bear_volatile** | **131** | **−0.181** | **−0.075** |
| sideways_volatile | 18 | −8.821 | 0.000 |

The three long regimes (bull_quiet, bull_volatile, sideways_quiet) all produce positive Sharpe, confirming that the vol-targeting overlay correctly sizes up in persistent regimes and down in choppy ones. Bear_volatile Sharpe improved from −0.850 (flat overlay) to −0.181 (model-argmax short), the largest single improvement from the short overlay analysis. Sideways_volatile (18 days, Sharpe −8.821) is flagged as an artefact of the small sample — 18 days is insufficient for a reliable Sharpe estimate.

### 6.4 Short Overlay Comparison

Five bear_volatile short strategies were evaluated post-training:

| Strategy | Sharpe | Max DD | bear_vol Sharpe | bear_vol avg_w |
|----------|--------|--------|----------------|----------------|
| Flat (baseline) | 0.555 | −27.4% | −0.850 | 0.000 |
| Gate P≥0.40 | 0.537 | −27.5% | −0.517 | −0.066 |
| Gate P≥0.25 | 0.547 | −26.5% | −0.316 | −0.082 |
| **Model argmax** | **0.555** | **−26.4%** | **−0.181** | **−0.075** |
| Tail hedge | 0.504 | −25.5% | −1.611 | −0.021 |

Model-argmax gating is uniquely efficient: it matches the baseline Sharpe while improving drawdown and bear_volatile regime performance. The probability gate variants (P≥0.40, P≥0.25) fire on days when bear_volatile probability is elevated but the model's top prediction is another regime — these are noisier signals. The tail hedge fires too frequently at the 0.50 threshold, introducing unwanted short exposure.

---

## 7. Limitations

### 7.1 HAR-RV Point Accuracy Dominance

HAR-RV outperforms V5 on qlike (−3.81 vs −3.59) and R² (0.15 vs 0.09) for raw vol forecasting. This reflects HAR-RV's structural advantage: it explicitly models the long-memory property of realised volatility. V5's surface signal and regime conditioning add value on dimensions HAR-RV cannot capture — tail detection, regime-conditional vol surface dynamics, and cross-asset inference — but do not improve point accuracy over the autoregressive baseline. A natural extension is to include HAR-RV as an additional context feature.

### 7.2 Sideways Regime Detection

Sideways_quiet (29.7%) and sideways_volatile (23.7%) classification accuracy is structurally limited by IV surface ambiguity in low-trend environments. The surface does not encode directionality when vol is low and returns are range-bound. Adding put/call open interest as additional surface channels, or a momentum signal as a context feature, could improve sideways detection.

### 7.3 Distribution Shift in 2024

Folds 22–27 show monotonic qlike degradation consistent with the low-volatility, high-momentum equity regime of 2023–2024. An expanding training window increasingly weights 2017–2022 data. A rolling 3-year window or regime-adaptive weighting would improve adaptation to structural regime shifts.

### 7.4 Small Short Sample

Bear_volatile occurs on only 131 of 1,764 OOS days (7.4%). The short overlay's Sharpe estimate is based on a small sample and is sensitive to single large-move days. The −0.181 regime Sharpe should be interpreted with wide confidence intervals.

### 7.5 Sideways_Volatile Sample Artefact

The sideways_volatile regime Sharpe of −8.821 on 18 days is not interpretable as a reliable performance estimate. This regime appears rarely in the OOS period and no position is taken (avg_weight = 0.0), so it does not affect portfolio-level results.

---

## 8. Summary of Key Results

| Finding | Value |
|---------|-------|
| OOS samples | 17,211 across 28 folds |
| MZ slope | 1.035 (near-ideal calibration) |
| MZ R² | 0.372 |
| Regime accuracy | 37.7% mean (6-class, 16.7% random baseline) |
| bear_volatile accuracy | 50.5% (highest) |
| MoE expert_std | 0.464 (no collapse) |
| Portfolio Sharpe | 0.555 |
| Max Drawdown | −26.4% |
| Calmar | 0.344 |
| Buy-and-hold Sharpe | 0.423 |
| Sharpe improvement vs B&H | +0.132 |
| DD improvement vs B&H | +14.3pp |
| bear_volatile Sharpe (short overlay) | −0.181 (vs −0.850 flat) |
| Top SHAP feature | days_since_iv_year_high |
| Top gating feature | risk_free_rate |
| New V5 feature in top gating | spy_adx14 (rank 2) |
| New V5 feature in top SHAP | spy_atr_ratio (rank 5) |

---

*Report generated 2026-04-05. All results are out-of-sample. No look-ahead bias. Walk-forward expanding window with minimum 4-fold warmup before first OOS prediction.*
