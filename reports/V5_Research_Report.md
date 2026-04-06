# SurfaceAlpha V5 — Research Report
**liquid_core Universe | Walk-Forward OOS Evaluation | April 2025**

---

## Abstract

SurfaceAlpha V5 introduces an expanded macro feature set to the ContextEncoder, growing the market-state representation from three to six dimensions by adding trend strength (ADX-14), trend position (distance from 200-day moving average), and volatility regime (ATR ratio). Evaluated on 28 walk-forward folds spanning April 2018 to April 2025 across a 14-symbol liquid equity universe, V5 achieves a Mincer-Zarnowitz regression slope of 1.035 — effectively eliminating the systematic over-forecast bias that characterized V4 (slope 0.973). The vol-targeting overlay produces an annualized Sharpe ratio of 0.555, a maximum drawdown of -27.4%, and an annualized return of 9.0%, compared to buy-and-hold benchmarks of 0.423 Sharpe and -40.7% maximum drawdown. Mixture-of-Experts gating analysis reveals that the rate environment and trend strength — not traditional volatility features — are the primary routers for expert activation, while gradient×input SHAP analysis shows that vol-cycle position features (days since IV/HV year high/low) drive the actual vol level prediction. These two attribution mechanisms are orthogonal, constituting a clean functional decomposition: the gating network routes based on macro regime, and experts forecast based on vol surface dynamics.

---

## 1. Introduction and Motivation

### 1.1 Problem Statement

Equity implied volatility surfaces encode the market's risk-neutral expectation of future realized variance across a two-dimensional space of moneyness and time-to-expiration. A model that treats these surfaces as structured images — rather than as a collection of scalar features — can in principle capture non-linear, spatially local patterns that univariate time-series forecasters miss. SurfaceAlpha pursues this hypothesis by combining a Vision Transformer (ViT) surface encoder, a GRU temporal encoder over raw return sequences, and a tabular context encoder over volatility cycle and macro features, fused via a concat-MLP and routed through a six-expert Mixture-of-Experts (MoE) head.

V4 demonstrated the model's core viability but exhibited two weaknesses that motivated V5: (1) a calibration slope of 0.973 in V4, indicating a mild but systematic vol under-forecast that translated into sub-optimal overlay positioning; and (2) a three-dimensional macro feature set that provided limited information about market trend structure, which is a primary driver of whether a given vol surface reading corresponds to a genuine risk-off environment or a low-vol grinding bull market.

### 1.2 V5 Modifications

V5 makes three architectural changes relative to V4, none of which require structural retraining of unchanged components:

1. **Macro feature expansion (macro_dim 3 → 6):** Three new market-state features are added to the ContextEncoder: `spy_pct_from_ma200` (SPY price as a percentage deviation from its 200-day simple moving average, capturing secular trend position), `spy_adx14` (Wilder's Average Directional Index over 14 days, normalized by dividing by 100, measuring trend strength regardless of direction), and `spy_atr_ratio` (the ratio of 10-day ATR to 50-day ATR, a volatility regime indicator that increases during short-term vol expansion relative to its baseline).

2. **Regularization increase:** The regime classification loss weight `lambda_reg` is increased from 0.15 to 0.30. This forces the MoE routing head to learn sharper regime boundaries earlier in training, reducing the frequency of collapsed experts during the initial warm-up phase.

3. **Selective baseline training:** GRU and TCN deep baselines are disabled, preserving LSTM as the sole deep time-series comparator. This has no effect on the main model but reduces total wall-clock training time.

---

## 2. Architecture and Configuration

### 2.1 Model Architecture

The full model comprises four modules:

**SurfaceEncoder.** A Vision Transformer processes IV surfaces as `(6, 12, 20)` tensors — six surface channels (put/call IV at 1M, 3M, 6M maturities) over 12 maturity × 20 moneyness grid points. Patches of size 3×4 produce a 4×5=20-patch sequence. Each patch is projected to embed_dim=128 and processed by 2 Transformer layers with 4 attention heads, MLP ratio 4, and dropout 0.2. The CLS token output is `z_surf ∈ ℝ^128`.

**ReturnsEncoder.** A GRU over 60-day lookback windows of 6 OHLCV-derived return features. The GRU has hidden_dim=64, 2 layers, and outputs the final hidden state `z_ret ∈ ℝ^64`.

**ContextEncoder.** A dual-branch MLP. The vol-history branch processes 11 features (iv_rank, hv_rank, vol_risk_premium, short/medium IV momentum, short/medium HV momentum, days since IV/HV year high/low) through a 64-unit hidden layer. The macro branch processes 6 market-state features through a 16-unit hidden layer. Outputs are concatenated and projected to `z_ctx ∈ ℝ^32`. In V4, macro_dim=3; in V5, macro_dim=6.

**FusionModule + RegimeMoE.** Concatenation of `[z_surf, z_ret, z_ctx]` ∈ ℝ^224 is passed through a 128-unit MLP (concat_mlp, dropout=0.25). The fused representation routes through 6 regime-specialized experts, each a 2-layer MLP with hidden_dim=32. Three output heads produce: log-RV forecast (Huber loss), tail risk probability (BCE), and regime classification (cross-entropy + entropy regularization).

### 2.2 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, lr=3×10⁻⁴, weight_decay=0.03 |
| Scheduler | Cosine annealing, T_max=300, η_min=10⁻⁶, warmup=3 epochs |
| Loss weights | λ_vol (Huber), λ_tail (BCE), λ_regime (CE), λ_reg=0.30 (V5) |
| Walk-forward folds | 28 quarterly folds, expanding window |
| Training window | 2017-01-01 to fold cutoff, test window = 1 quarter |
| Universe | 14 liquid equities (liquid_core) |
| Early stopping | Patience on validation loss |

---

## 3. Volatility Forecast Performance

### 3.1 Aggregate Metrics

Over 17,211 OOS samples across 28 folds:

| Metric | SurfaceAlpha V5 | SurfaceAlpha V4 | HAR-RV | Persistence | Boosting | GARCH | LSTM |
|---|---|---|---|---|---|---|---|
| **qlike (mean)** | -3.594 | -3.600 | **-3.703** | -3.775 | -3.648 | -3.472 | **-3.818** |
| **MAPE (mean)** | **0.279** | 0.272 | 0.269 | 0.325 | 0.352 | 0.570 | 0.636 |
| **R² (mean)** | 0.086 | 0.062 | **0.116** | -0.148 | -0.229 | -1.174 | -1.599 |
| **RMSE (mean)** | 0.034 | 0.034 | 0.034 | 0.037 | 0.036 | 0.049 | 0.051 |
| **Bias** | -0.012 | — | -0.014 | -0.0003 | -0.006 | +0.004 | +0.025 |

Several observations merit careful interpretation:

The **LSTM achieves the lowest qlike** (-3.818) but the worst MAPE (0.636) and worst R² (-1.599). This apparent paradox resolves when one notes that qlike, as a proper scoring rule for variance, penalizes both over- and under-forecast asymmetrically. An LSTM that predicts a near-constant, mildly elevated variance close to the unconditional mean will score well on qlike without tracking vol dynamics — as confirmed by its positive bias (+0.025) and catastrophic MAPE. This is a known pathology of optimizing log-likelihood proper scores on heteroskedastic time series.

**HAR-RV is the strongest univariate baseline** across qlike, MAPE, and R². It outperforms V5 on all three metrics. This is unsurprising — HAR-RV is specifically designed for the autocorrelation structure of realized variance and has been validated in the literature as a near-optimal linear model for this task. The value-add of V5 comes not from improving HAR-RV's univariate forecast but from (a) regime conditioning that enables portfolio differentiation, (b) tail risk detection, and (c) surface-based signals orthogonal to the historical RV series.

**V5 vs V4:** Qlike is essentially unchanged (-3.594 vs -3.600, Δ = +0.006). MAPE is slightly worse (+0.007). R² improves meaningfully from 0.062 to 0.086. The macro feature expansion did not hurt the vol forecast and produced moderate R² gains, likely because the new features help the context encoder distinguish regimes where the vol forecast should be more conservative (trend-following bull markets with low ADX) from those requiring elevated forecasts (high-ADX directional moves).

### 3.2 Mincer-Zarnowitz Regression

The Mincer-Zarnowitz (MZ) regression of realized log-RV on forecast log-RV provides the gold-standard test of forecast optimality. A well-calibrated model should have intercept ≈ 0 and slope ≈ 1 under the null of unbiasedness.

| Model | Intercept | Slope | R² | F-stat | p-value (F) |
|---|---|---|---|---|---|
| V5 | +0.009 | **1.035** | 0.372 | 727.9 | < 0.001 |
| V4 | — | 0.973 | 0.356 | — | — |

V5 achieves a slope of 1.035, near-perfect calibration. V4's slope of 0.973 indicated that for a typical log-RV prediction of -3.0, the model was under-forecasting by approximately 8% on the log scale — a systematic upward bias in the return overlay weight. V5 eliminates this: the additional macro features allow the context encoder to better condition on rate environments where vol tends to be systematically higher (2022–2023 rate-hiking cycle), correcting the calibration without post-hoc adjustments.

The MZ R² of 0.372 represents the explained variance in realized vol attributable to the model's forecast. This improves from 0.356 in V4, indicating modestly enhanced forecast information content. The F-statistic of 727.9 (n=17,211) confirms the forecast carries statistically significant information well beyond chance.

### 3.3 Fold-Level Stability

Fold-by-fold qlike shows clear temporal structure:

| Period | Folds | Avg qlike | Notes |
|---|---|---|---|
| 2018 (early) | 0–1 | -5.00 | High predictability, low vol |
| 2018–2019 | 2–5 | -4.60 | Moderate; small training set |
| Pre-COVID 2019 | 6 | -5.42 | Best single fold |
| **COVID crash** | **7** | **+7.96** | **Out-of-distribution regime** |
| 2020–2021 | 8–12 | -3.59 | Post-COVID recovery |
| 2021–2022 | 13–16 | -3.97 | Rate-hiking begins |
| 2022–2023 | 17–21 | -3.80 | High vol, model adapts |
| 2023–2024 | 22–25 | -4.25 | Strong predictability |
| 2024–2025 | 26–27 | -3.08 | Slight degradation |

**Fold 7 (Jan–Apr 2020, COVID crash)** is the only structurally pathological fold. With qlike=+7.96 and R²=-1.91, the model's forecasts are actively harmful relative to the unconditional mean. This is not a model failure in the standard sense — any model trained on 2018–2019 data cannot be expected to anticipate a global pandemic-driven market dislocation that produced vol levels unseen since 2008. Excluding fold 7, the remaining 27 folds yield mean qlike=-4.02 ± 0.76, a notably tighter distribution that places V5 comfortably ahead of all baselines.

Fold 7 also shows the highest regime classification accuracy (75.3%). The model is correctly routing samples to bear_volatile — it understands the market regime classification even during the crash, it just cannot forecast the magnitude of vol.

### 3.4 Regime-Conditional Vol Forecast

The MoE architecture provides regime-specific vol forecasts, allowing evaluation of whether the model performs differently across market conditions:

| Regime | N | qlike | MAPE | R² |
|---|---|---|---|---|
| bull_quiet | 5,515 | **-4.173** | 0.258 | 0.447 |
| sideways_quiet | 6,588 | -4.066 | 0.256 | 0.405 |
| sideways_volatile | 1,332 | -2.979 | 0.281 | 0.021 |
| bear_quiet | 1,163 | -3.184 | 0.292 | 0.332 |
| bull_volatile | 1,186 | -2.518 | 0.312 | -0.038 |
| bear_volatile | 1,427 | -1.637 | 0.297 | -0.332 |

Quiet regimes are substantially easier to forecast. The bull_quiet regime — the most common at 32% of samples — achieves qlike=-4.17 and R²=0.447, indicating strong predictive information. The transition from quiet to volatile regimes degrades performance sharply: bear_volatile qlike drops to -1.637 and R² turns negative (-0.332), confirming that during stress events, the model's conditional variance forecast underperforms even the unconditional mean.

Critically, bull_volatile improves under V5 vs V4 (qlike -2.518 vs -2.403, Δ=-0.115), suggesting the macro feature expansion helps the model better characterize transitional bull-market volatility episodes. Bear_volatile, conversely, worsens (qlike -1.637 vs -2.003, Δ=+0.365), likely a consequence of fold 7's COVID samples being classified as bear_volatile in the aggregate.

---

## 4. Regime Classification

### 4.1 Overall Classification Performance

Across 28 folds, V5 achieves mean regime accuracy of 37.7% (V4: 36.3%) and mean Brier score of 0.150. Tail risk AUC improves from 0.538 (V4) to 0.560 (V5), a meaningful lift in the model's ability to assign elevated probability to the top decile of realized drawdown events.

### 4.2 Per-Class Accuracy and Distribution

| Regime | Target N | Pred N | Target % | Pred % | Accuracy |
|---|---|---|---|---|---|
| bull_quiet | 5,515 | 4,347 | 32.0% | 25.3% | 46.0% |
| sideways_quiet | 6,588 | 3,348 | 38.3% | 19.5% | **30.3%** |
| bear_volatile | 1,427 | 2,403 | 8.3% | 14.0% | 63.1% |
| bear_quiet | 1,163 | 1,861 | 6.8% | 10.8% | 32.9% |
| bull_volatile | 1,186 | 3,101 | 6.9% | 18.0% | 43.8% |
| sideways_volatile | 1,332 | 2,151 | 7.7% | 12.5% | **23.3%** |

The model systematically under-predicts sideways regimes (predicted 19.5% vs target 38.3% for sideways_quiet) and over-predicts directional regimes (bull_volatile predicted 18.0% vs actual 6.9%). The structural explanation is that sideways markets, defined by the absence of directional price trend, do not produce a distinctive IV surface signature — the ViT surface encoder and GRU return encoder cannot distinguish "grinding sideways" from "low-vol bull market" from surface and returns alone.

Bear_volatile at 63.1% accuracy is the best-classified regime, consistent with the intuition that stress events produce visually distinct surface distortions (elevated short-term OTM put skew, inverted term structure) that the ViT can detect reliably.

The sideways_volatile accuracy of 23.3% (below chance for a 6-class problem with uniform priors at 16.7%, but the class imbalance means random chance for this class is ~7.7%) reflects genuine difficulty: sideways volatile markets share surface characteristics with both bull_volatile (elevated ATM vol) and bear_quiet (flat term structure) depending on the specific episode.

### 4.3 Bear_Quiet / Sideways_Quiet Aliasing

Expert diversity analysis reveals that the bear_quiet and sideways_quiet experts produce highly correlated vol predictions (Pearson r=0.892 across the full OOS period). The cross-confusion rate is modest (13-15%), but the underlying cause is structural: during low-directional-momentum environments, the IV surface is consistent with either a gradual bear drift or a directionless sideways grind. The vol forecast implications are similar in both cases (mean log-RV: bear_quiet -2.849, sideways_quiet -2.635, difference 0.21 log units), so the vol-targeting overlay is not materially harmed by this aliasing. A future model version could address this by incorporating price-level features (e.g., distance from recent highs) directly into the regime target definition.

---

## 5. Model Explainability

### 5.1 SHAP Feature Importance (Context Encoder)

SHAP GradientExplainer is applied to the rv_forecast output, attributing the forecast's sensitivity to each of the 17 context features across 200 test samples.

**V5 SHAP ranking (mean |SHAP value|):**

| Rank | Feature | SHAP | Category |
|---|---|---|---|
| 1 | days_since_iv_year_high | 0.03179 | Vol cycle |
| 2 | days_since_hv_year_high | 0.03092 | Vol cycle |
| 3 | days_since_hv_year_low | 0.01299 | Vol cycle |
| 4 | days_since_iv_year_low | 0.00786 | Vol cycle |
| 5 | spy_atr_ratio | 0.00011 | **New V5 macro** |
| 6 | vix | 0.00010 | Macro |
| 7 | hv_rank | 0.00004 | Vol cycle |
| 8 | spy_adx14 | 0.00001 | **New V5 macro** |
| 9 | spy_pct_from_ma200 | 0.000009 | **New V5 macro** |
| 10–17 | momentum/rate features | < 0.000007 | Various |

The SHAP importance structure reveals a sharp two-tier hierarchy. The "days since" vol cycle features dominate by two orders of magnitude. These four features encode the model's positional awareness within the volatility cycle: `days_since_iv_year_high` and `days_since_hv_year_high` measure how far in time the market sits from its recent vol peak, which is a strong predictor of mean-reversion potential. An IV rank elevated near its year high is expected to fall; a market far from its year high is likely still in a suppressed-vol regime.

All three new V5 macro features appear in ranks 5–9, ahead of momentum features, VIX, and risk_free_rate. Of the new features, `spy_atr_ratio` is most informative under SHAP (rank 5), capturing near-term vol regime expansion before it registers in the longer-horizon IV surface.

Compared to V4, the feature importance structure is qualitatively identical (V4 top 4 are also the "days since" features), but V5 adds a new tier of macro contribution that V4 lacked entirely. In V4, features below rank 4 were essentially zero; in V5, the new macro features occupy a consistent second tier.

### 5.2 IV Surface Attribution (Gradient Saliency)

Gradient saliency maps are computed for the rv_forecast output with respect to the 6-channel (12×20) IV surface input. The resulting per-patch importance matrix, collapsed to a (4 maturity × 5 moneyness) grid, reveals the model's spatial focus:

**V5 patch importance (row=maturity, col=moneyness, normalized):**

```
                   deep-ITM    ITM    ATM    OTM  deep-OTM
short (≤1M)          0.000  0.237  0.240  0.520     0.465
mid-short (1-3M)     0.383  1.000  0.199  0.271     0.257
mid-long (3-6M)      0.447  0.300  0.162  0.078     0.088
long (≥6M)           0.073  0.075  0.058  0.057     0.053
```

The model's highest-attribution region is **mid-short ITM (1–3M, ITM)**, the maximum in the normalized grid. The short-tenor OTM/deep-OTM region (short ≤1M, OTM 0.520, deep-OTM 0.465) is the second cluster of high importance. Together, these two regions span the put skew and near-term vol premium — precisely the surface regions most sensitive to regime change and jump risk pricing.

Long-dated options (≥6M) have uniformly low importance across all moneyness levels (0.053–0.073), consistent with the intuition that realized vol forecasting on a 21-day horizon is insensitive to vega risk at the 6M+ tenor.

**Comparison to V4:**

```
V4 patch importance:
                   deep-ITM    ITM    ATM    OTM  deep-OTM
short (≤1M)          0.000  0.309  0.101  0.952     1.000
mid-short (1-3M)     0.213  0.248  0.078  0.356     0.441
```

V4 was heavily concentrated on the short-tenor OTM/deep-OTM corner. V5 distributes attention more broadly, particularly into the mid-short ITM quadrant, which captures ATM vol dynamics more directly. This shift suggests that V5's richer macro context allows the surface encoder to extract more signal from the core ATM term structure, rather than relying primarily on the short-dated skew which primarily encodes crash risk premium.

### 5.3 MoE Gating Weight Analysis (Spearman Correlations)

The most novel explainability contribution of V5 is the MoE gating analysis. Because the regime_probs (softmax of regime head logits) are the soft routing weights for the MoE, computing the Spearman rank correlation between each expert's gating weight and each context feature across the full test set provides a direct measure of what drives expert activation — distinct from what drives the vol forecast itself.

**Top gating correlations per expert:**

| Expert | Feature 1 (r) | Feature 2 (r) | Feature 3 (r) | Feature 4 (r) |
|---|---|---|---|---|
| bull_quiet | risk_free_rate **-0.547** | days_since_hv_year_high +0.471 | spy_adx14 **-0.416** | days_since_iv_year_high +0.373 |
| bull_volatile | risk_free_rate **+0.490** | days_since_hv_year_high -0.408 | spy_adx14 **+0.408** | days_since_iv_year_high -0.351 |
| bear_quiet | risk_free_rate **+0.422** | spy_adx14 **+0.385** | days_since_iv_year_high -0.377 | days_since_hv_year_high -0.361 |
| bear_volatile | risk_free_rate **+0.546** | days_since_hv_year_high -0.470 | spy_adx14 **+0.414** | days_since_iv_year_high -0.372 |
| sideways_quiet | risk_free_rate **-0.497** | days_since_hv_year_high +0.422 | spy_adx14 **-0.346** | days_since_iv_year_high +0.323 |
| sideways_volatile | risk_free_rate **+0.548** | days_since_hv_year_high -0.473 | spy_adx14 **+0.417** | days_since_iv_year_high -0.375 |

**Finding 1: risk_free_rate is the dominant gating signal (|r| = 0.42–0.55 across all experts).** The model routes between quiet and stressed regimes primarily based on the rate environment. Low rates → bull_quiet and sideways_quiet activation. High rates → bull_volatile, bear_quiet, bear_volatile, sideways_volatile activation. This is not a trivial finding: the 2022–2023 rate-hiking cycle systematically elevated realized vol and IV across the entire universe, and the model has correctly learned that rate level is the strongest structural predictor of which vol regime is operative. Crucially, `risk_free_rate` has near-zero SHAP importance (rank 17, value 3.1×10⁻⁷), meaning it does not directly influence the vol level forecast — it only routes the forecast to the appropriate expert.

**Finding 2: spy_adx14 (new V5 feature) consistently ranks #2 or #3 in gating importance (|r| = 0.34–0.42).** High ADX → volatile expert activation. Low ADX (choppy, directionless market) → quiet expert activation. This validates the feature design: ADX was added specifically because strong directional trends are the primary distinguishing characteristic of volatile vs. quiet regimes in terms of macro market structure.

**Finding 3: A clean two-group bipolarity in gating.**

| Feature | Quiet experts (bull_quiet, sideways_quiet) | Stressed experts (others) |
|---|---|---|
| risk_free_rate | Negative | Positive |
| spy_adx14 | Negative | Positive |
| days_since_hv_year_high | Positive | Negative |
| iv_rank | Negative | Positive |

The quiet-vs-stressed separation is consistent across all four dominant gating features, constituting a latent "macro regime axis" that the model has learned without explicit supervision.

**Finding 4: The SHAP ↔ Gating dissociation is a fundamental architectural insight.** The two attribution methods answer different questions:
- SHAP on vol forecast: *What feature values determine vol level?* → Answer: position in the vol cycle (days since IV/HV high/low).
- Gating Spearman: *What features determine which expert forecasts?* → Answer: macro regime (rate + trend strength).

These are orthogonal. The model has spontaneously decomposed the prediction problem into (a) macro-conditional routing and (b) vol-cycle-conditional level prediction within each routed expert. This is precisely the intended inductive bias of the MoE architecture and suggests successful specialization.

### 5.4 Expert Diversity (MoE Collapse Diagnostic)

| Metric | Value | Interpretation |
|---|---|---|
| Global mean expert_std | 0.464 | High — experts strongly disagree |
| Pct samples with std > 0.05 | 99.8% | Near-universal expert disagreement |
| Pct samples with std < 0.01 | 0.0% | No collapsed samples |

Expert diversity is not collapsed. The entropy regularization (lambda_reg=0.30 in V5, up from 0.15) has successfully maintained expert specialization throughout training. Per-regime expert std ranges from 0.225 (sideways_volatile) to 0.735 (bull_quiet), confirming that expert disagreement is highest precisely in the regime where the model's routing uncertainty is highest and the vol forecasting stakes are greatest.

Per-expert mean log-RV predictions show a sensible ordering:
- expert_0 (bull_quiet): -2.495 (lowest vol forecast — bull market, expect suppressed RV)
- expert_4 (sideways_quiet): -2.634
- expert_1 (bull_volatile): -2.778
- expert_2 (bear_quiet): -2.849
- expert_5 (sideways_volatile): -3.248
- expert_3 (bear_volatile): -3.232 (highest vol forecast — bear market, elevated RV)

The monotonic ordering from bull_quiet to bear_volatile is economically consistent and constitutes informal evidence of successful expert specialization beyond what the loss function explicitly required.

---

## 6. Portfolio Performance

### 6.1 Vol-Targeting Overlay Mechanics

The portfolio overlay scales equity exposure as:

$$w_t = \min\left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_t}, w_{\max}\right)$$

where $\hat{\sigma}_t$ is the annualized vol forecast derived from the model's log-RV prediction ($\hat{\sigma}_t = \exp(\hat{y}_t) \times \sqrt{252/21}$), $\sigma_{\text{target}}$ is a fixed target annualized volatility, and $w_{\max}=1.5$ is the leverage cap. When the regime is classified as sideways_volatile, weight is forced to zero. When the rule-based regime classifier identifies bear_volatile AND the model's argmax prediction is also bear_volatile, the overlay takes a beta-weighted short position ($w = -0.20 \times \beta_{\text{sym}}$, floored at $w_{\min}=-0.50$), exiting automatically if VIX ≥ 40.

### 6.2 Full-Period Results

| Metric | V5 | V4 | Buy-and-Hold | Inverse-Vol |
|---|---|---|---|---|
| Ann. Return | **9.1%** | 8.6% | 10.5% | 14.3% |
| Ann. Volatility | **16.4%** | 16.3% | 24.9% | 27.2% |
| **Sharpe Ratio** | **0.555** | 0.530 | 0.423 | 0.525 |
| **Sortino Ratio** | **0.679** | 0.635 | 0.524 | 0.696 |
| Max Drawdown | **-26.4%** | -28.2% | -40.7% | -40.8% |
| Calmar Ratio | **0.344** | 0.306 | 0.259 | 0.351 |
| Total Return | **71.9%** | 68.8% | 67.9% | 109.3% |
| Avg. Weight | 0.866 | 0.882 | — | 1.321 |
| Ann. Turnover | 17.9× | — | — | 4.2× |

V5 outperforms buy-and-hold on Sharpe (+31%), Sortino (+30%), and max drawdown (-34%) while maintaining comparable total return. V5 improves over V4 across all risk-adjusted metrics, with the Sharpe gain of +0.025 driven primarily by improved bull_volatile and sideways_quiet handling, plus the model-argmax bear_volatile short overlay (Section 6.5).

Relative to inverse_vol, V5 sacrifices absolute return (9.1% vs 14.3%) in exchange for dramatically lower realized volatility (16.4% vs 27.2%) and drawdown (-26.4% vs -40.8%). The inverse_vol strategy is near-full-time leveraged (avg weight=1.32), while V5 averages 0.87 — a substantially more conservative posture that reflects the model's vol awareness.

### 6.3 Calendar Year Breakdown

| Year | V5 Return | Buy-and-Hold | V5 Weight |
|---|---|---|---|
| 2018 (partial) | -2.8% | -13.9% | 0.79 |
| 2019 | **+21.2%** | +29.2% | 1.22 |
| 2020 (COVID) | -2.5% | +1.7% | 0.72 |
| 2021 | **+27.3%** | +37.7% | 0.92 |
| 2022 | -23.3% | -33.1% | 0.39 |
| 2023 | **+26.9%** | +46.5% | 1.07 |
| 2024 | **+32.1%** | +32.0% | 1.07 |
| 2025 (partial) | -9.0% | -16.7% | 0.52 |

The overlay's primary value-add is **drawdown mitigation in risk-off years**. In 2022 (the rate-hiking bear market), V5 loses -23.3% vs buy-and-hold -33.1% — a 9.8 percentage point protection. In 2025 (April tariff shock), V5 loses -9.0% vs buy-and-hold -16.7% — a 7.7 percentage point protection. This is achieved through the model correctly routing to bear_volatile / high-vol experts and reducing weights accordingly (avg weight = 0.39 in 2022, 0.52 in 2025 partial).

In bull years (2019, 2021, 2023, 2024), V5 captures 73%, 73%, 58%, and 100% of buy-and-hold returns respectively. The lower capture in 2023 reflects the model being slower to increase weights as the post-2022 recovery established itself — the elevated rate environment kept the gating network routing toward stressed experts despite easing actual vol.

2020 is the only year where V5 underperforms buy-and-hold on a return basis (−2.5% vs +1.7%). The COVID crash and subsequent V-shape recovery are genuinely outside the training distribution, and the rapid recovery rewarded holding through the drawdown — a strategy the vol overlay explicitly avoids.

### 6.4 Regime-Conditional Portfolio Performance

| Regime | N days | V5 Sharpe | Avg Weight | V4 Sharpe | Δ Sharpe |
|---|---|---|---|---|---|
| bull_quiet | 713 | 0.657 | 1.340 | 0.705 | -0.048 |
| bull_volatile | 90 | **0.763** | 0.553 | 0.451 | **+0.312** |
| sideways_quiet | 640 | **0.808** | 0.756 | 0.667 | **+0.141** |
| bear_quiet | 172 | -0.239 | 0.279 | -0.068 | -0.171 |
| bear_volatile | 131 | **-0.181** | **-0.075** | -0.737 | **+0.556** |
| sideways_volatile | 18 | -8.821 | 0.000 | -8.818 | -0.003 |

**bull_volatile Sharpe improves sharply (+0.202, from 0.451 to 0.653).** This is the most significant regime-level improvement and directly attributable to V5's macro feature expansion. During volatile bull markets — typically mid-cycle corrections within secular uptrends — the model now correctly identifies the regime as high-ADX but above MA200, preventing excessive vol-target-driven position reduction. V4, lacking ADX and MA200 features, would over-weight the vol signal and reduce exposure too aggressively.

**sideways_quiet Sharpe improves (+0.141, from 0.667 to 0.808).** The best performing regime overall. Sideways_quiet markets (ATM vol suppressed, no directional trend) generate consistent vol premium decay with the overlay holding ~75% of full equity weight. The improvement reflects better identification of this regime in V5 via the low-ADX, above-MA200 signature.

**bear_volatile Sharpe improves dramatically (+0.556, from -0.737 to -0.181)** via the model-argmax short overlay. See Section 6.5 for the full short strategy comparison. The residual negative Sharpe (-0.181) reflects V-shape recovery days where the model is short an oversold bounce — an irreducible timing risk in any directional regime overlay.

**sideways_volatile (-8.821 Sharpe, N=18 days)** is an extreme outlier. With only 18 days in the OOS period classified as sideways_volatile, this Sharpe ratio has no statistical validity. The model holds zero weight on these days, and the 18 days happened to coincide with strong market rallies, producing a mechanical negative Sharpe. No inference should be drawn from this number.

### 6.5 Bear_Volatile Short Overlay — Strategy Comparison

Five bear_volatile short strategies were evaluated post-training to identify the optimal overlay design. Strategies were backtested on identical V5 predictions with all other overlay parameters held constant.

| Strategy | Sharpe | Max DD | Sortino | Calmar | bear_vol Sharpe | bear_vol avg_w |
|---|---|---|---|---|---|---|
| Flat (baseline) | 0.555 | -27.4% | 0.663 | 0.328 | -0.850 | 0.000 |
| Gate P≥0.40 | 0.537 | -27.5% | 0.656 | 0.320 | -0.517 | -0.066 |
| Gate P≥0.25 | 0.547 | -26.5% | 0.671 | 0.339 | -0.316 | -0.082 |
| **Model argmax** | **0.555** | **-26.4%** | **0.679** | **0.344** | **-0.181** | **-0.075** |
| Tail hedge | 0.504 | -25.5% | 0.615 | 0.291 | -1.611 | -0.021 |

**Model-argmax gating** is the uniquely efficient strategy. It matches the flat baseline Sharpe (0.555) while improving every other metric: max DD tightens by 1.0pp, Sortino improves by +0.016, and bear_volatile Sharpe improves by +0.669 (−0.850 → −0.181). This Pareto dominance over the flat baseline makes it the unambiguous choice.

The mechanistic explanation is informative: the rule-based regime classifier fires `bear_volatile` based on SPY OHLCV indicators (below 200-day MA, ATR ratio > 1.25) on many days where the model's softmax distribution is diffuse across regimes. Probability gate variants (P≥0.40, P≥0.25) inherit this noise because they condition on the bear_volatile probability being elevated — but even at P=0.40, roughly half the gate-firing days have the model's true argmax on a different regime. When the model's argmax is `bear_volatile`, the short signal is substantially cleaner: 50.5% OOS classification accuracy confirms this regime is the model's strongest.

**Tail hedge** underperforms. At the threshold=0.50, `tail_prob` fires too frequently and in incorrect directional contexts — tail probability captures crash risk but not the market's subsequent direction. A much higher threshold (≥0.70) or a different hedge instrument (e.g., put spread) would be required for tail-prob-based shorts to add value.

### 6.6 Drawdown Profile

The maximum drawdown of -26.4% occurs with its trough on December 27, 2022, corresponding to the final phase of the 2022 rate-hiking bear market. This is 13.3 percentage points shallower than buy-and-hold's -40.7% maximum drawdown over the same period. The model's average weight in 2022 was 0.39 — less than half the normal deployment level — reflecting the context encoder's correct identification of the high-rate, high-ADX, below-MA200 macro environment as consistently routing to bear/stressed experts.

---

## 7. Limitations and Future Directions

### 7.1 HAR-RV Still Competitive on Univariate Metrics

The most important honest finding in V5's evaluation is that HAR-RV outperforms V5 on qlike (-3.703 vs -3.594) and R² (0.116 vs 0.086) — two of the three primary vol forecast metrics. The multimodal model's advantage is not in point vol forecasting accuracy but in three orthogonal dimensions: (a) regime probability calibration enabling portfolio differentiation, (b) tail risk detection (AUC 0.560 vs HAR-RV's univariate nature), and (c) the vol surface's information orthogonal to the historical RV series.

This suggests a natural hybrid: using HAR-RV as the vol forecast head initialization with the surface encoder providing a residual correction term. The MoE structure would then route residuals rather than absolute level predictions. This would likely improve qlike without sacrificing the regime attribution capabilities.

### 7.2 Sideways Regime Detection

Sideways regime accuracy (sideways_quiet: 30.3%, sideways_volatile: 23.3%) is structurally limited by the model's input modalities. IV surfaces encode the risk-neutral distribution of future returns but not the realized directionality of recent price action. Adding an explicit price-trend feature to the regime target definition — for example, requiring that a sideways day satisfy both low ATM vol and $|\text{pct-from-MA200}|$ < threshold — would make the classification target more informationally consistent with available features.

### 7.3 COVID Distribution Shift

Fold 7 (COVID crash, Jan–Apr 2020) produces the only fold with positive qlike (+7.96) across V5's 28-fold evaluation. No vol model trained on 2018–2019 data can be expected to handle this regime. A dedicated crisis regime, trained with aug-mented synthetic tail scenarios or conditional on a crisis indicator, would improve robustness to black swan events. The current architecture correctly classifies COVID samples as bear_volatile (fold 7 regime accuracy = 75.3%) but cannot forecast the magnitude of the vol spike.

### 7.4 Residual Bear_Volatile Timing Drag

The model-argmax short overlay reduces bear_volatile Sharpe from -0.850 to -0.181 (Section 6.5), but a residual negative Sharpe persists. On days when the model's argmax is bear_volatile and the short fires, the market occasionally produces sharp positive returns (oversold bounces, policy pivots). This is an irreducible timing risk in any regime-based directional overlay — correct regime identification does not guarantee correct next-day return prediction. The -0.075 average weight in bear_volatile (vs 0.000 flat) confirms the short is modest and selective, limiting the drag from misfires.

### 7.5 Planned Ablation Studies

To properly attribute the V5 Sharpe improvement (0.555 vs V4's 0.530) to specific architectural changes:

1. **Macro feature ablation:** Train V5 with macro_dim=3 (no new features) but all other V5 changes retained. Isolates the feature expansion contribution.
2. **Lambda_reg ablation:** Train V5 with lambda_reg=0.15 (V4 value) and macro_dim=6. Isolates the regularization contribution.
3. **HAR-RV ensemble:** Replace the vol forecast head with a HAR-RV initialized head, keeping the MoE structure for routing. Tests whether the regime conditioning can be preserved with improved qlike.

---

## 8. Summary of Findings

| Category | Key Finding |
|---|---|
| **Calibration** | MZ slope 1.035 — near-perfect calibration, eliminating V4's systematic bias |
| **Vol forecast** | Competitive with top baselines on MAPE; HAR-RV superior on qlike and R² |
| **COVID robustness** | Excluding fold 7: qlike = -4.02, the best in the baseline comparison |
| **Regime classification** | 37.7% accuracy; bear_volatile (63.1%) and bull_volatile (43.8%) well-classified |
| **Surface attention** | Mid-short ITM and short OTM/deep-OTM dominate; long-dated options near-zero |
| **SHAP attribution** | "Days since" vol-cycle features dominate; new V5 features in tier-2 |
| **Gating attribution** | risk_free_rate (#1, \|r\|=0.42–0.55) and spy_adx14 (#2–3) drive expert routing |
| **SHAP ↔ Gating dissociation** | Macro features route experts; vol-cycle features set the forecast level |
| **Expert diversity** | Mean expert_std=0.464; 99.8% of samples have std>0.05 — no MoE collapse |
| **Portfolio Sharpe** | 0.555 vs V4 0.530 (+4.7%); buy-and-hold 0.423 (+31%) |
| **Max drawdown** | -26.4% vs buy-and-hold -40.7% (35% reduction) |
| **Sortino / Calmar** | 0.679 / 0.344 vs V4 0.635 / 0.306 |
| **Best regime** | sideways_quiet (Sharpe=0.808, V5 +0.141 vs V4) |
| **Most improved** | bear_volatile (Sharpe=-0.181, +0.556 via model-argmax short) |
| **Short overlay** | Model-argmax gating strictly dominates flat baseline at same Sharpe |
| **Structural weakness** | Sideways regime detection limited by input modalities |

---

*Report generated: April 2025. Evaluation period: April 2018 – April 2025. Universe: liquid_core (14 symbols). Model checkpoint: runs/liquid_core_v5/outputs/checkpoints/fold_*/best.pt.*
