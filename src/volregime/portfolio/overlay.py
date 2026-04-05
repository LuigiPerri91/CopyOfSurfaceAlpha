"""
Portfolio overlay: converts model predictions into a position size w ∈ [w_min, w_max].

Strategy logic (per backtest.yaml / PDR):
    1. Vol-targeting:  w = sigma_target / sigma_hat_annualised
    2. Regime sizing:  w = w * regime_rules[regime_name].position_size
    3. Crisis gating:  w = w * (1 - p_crisis)
                       p_crisis = P(bear_volatile) + P(sideways_volatile)
    4. ADX override:   if ADX < 20 (no trend): w = w * (1 - size_reduction)
    5. Confidence:     w = w / (1 + ensemble_std)   [Phase 7, disabled by default]
    6. Clip:           w = clip(w, w_min, w_max)
    7. VIX breaker:    if VIX >= threshold → w = 0.0  (circuit breaker, applied last)

Config keys (cfg["backtest"]):
    vol_targeting.sigma_target / w_max / w_min
    regime_gating.enabled / crisis_regimes
    regime_rules.{name}.position_size
    adx_override.enabled / adx_threshold / size_reduction
    confidence_scaling.enabled
    vix_circuit_breaker.enabled / threshold
    costs.transaction_cost_bps / slippage_bps
    rebalance.min_trade_threshold
"""
import logging
import numpy as np 

from .regime_rules import RegimeRules, REGIME_INT_TO_NAME

logger = logging.getLogger(__name__)

class PortfolioOverlay:

    def __init__(self, cfg: dict):
        bt_cfg = cfg.get('backtest', cfg)
        vt_cfg = bt_cfg.get('vol_targeting', {})

        self.sigma_target = float(vt_cfg.get('sigma_target', 0.10))
        self.w_max = float(vt_cfg.get('w_max', 1.5))
        self.w_min = float(vt_cfg.get('w_min', 0.0))

        self.rules = RegimeRules(cfg)

        cs_cfg = bt_cfg.get('confidence_scaling', {})
        self.use_confidence_scaling = bool(cs_cfg.get('enabled', False))

        rb_cfg = bt_cfg.get('rebalance', {})
        self.min_trade_threshold = float(rb_cfg.get('min_trade_threshold',0.02))

        costs = bt_cfg.get('costs',{})
        self.cost_bps = (float(costs.get('transaction_cost_bps', 5)) + float(costs.get('slippage_bps',2)))/ 10_000

        vix_cb = bt_cfg.get('vix_circuit_breaker', {})
        self.vix_cb_enabled = bool(vix_cb.get('enabled', False))
        self.vix_cb_threshold = float(vix_cb.get('threshold', 40.0))

    def compute(
        self,
        log_rv_pred: float,
        regime_probs: np.ndarray,
        signals: dict | None = None,
        ensemble_std: float | None = None,
        macro_regime_name: str | None = None,
    ) -> dict:
        """
        Compute the target position size for a single date.

        Args:
            log_rv_pred:   model's log(RV_21d) prediction (denormalised)
            regime_probs:  (6,) softmax probabilities from regime head
            signals:       dict from identify_regime() — for ADX override
            ensemble_std:  std of rv_forecast across ensemble members (Phase 7)

        Returns:
            dict: weight (final), plus intermediate values for logging/analysis
        """
        # annualize predicted vol
        # rv_21d is the 21-day ahead realized vol (raw scale)
        # annualize: sigma_ann = rv_21d * sqrt(252/21)
        rv_21d = float(np.exp(log_rv_pred))
        sigma_hat = rv_21d * np.sqrt(252.0 / 21.0)
        sigma_hat = np.maximum(sigma_hat, 1e-4)

        # regime sizing — prefer SPY macro regime (rule-based) if provided;
        # fall back to model argmax so standalone calls still work
        regime_idx = int(np.argmax(regime_probs))
        regime_name = macro_regime_name if macro_regime_name else REGIME_INT_TO_NAME.get(regime_idx, "bull_quiet")
        regime_size = self.rules.get_position_size(regime_name)

        if regime_size < 0:
            # Short regime: bypass vol targeting entirely.
            # Vol targeting is wrong-direction for shorts — high vol produces a tiny
            # vol-target weight which when multiplied by a negative size gives a tiny
            # short, but high-vol bear is exactly when you want maximum short exposure.
            # Use the fixed short size from regime_rules directly.
            w = float(regime_size)
        else:
            # Long regime: vol targeting then regime sizing
            w = self.sigma_target / sigma_hat
            w *= regime_size

        # crisis gating
        p_crisis = 0.0
        if self.rules.gating_enabled:
            # probability mass on crisis regimes: bear_volatile = 3, sideways_volatile = 5
            p_crisis = float(regime_probs[3] + regime_probs[5])
            p_crisis = np.clip(p_crisis, 0.0, 1.0)
            w *= (1.0 - p_crisis)

        # ADX override
        adx_override_applied = False
        adx_14 = float(signals.get('adx_14', 99)) if signals else 99.0
        w, adx_override_applied = self.rules.apply_adx_override(w, adx_14)

        # confidence scaling
        conf_applied = False
        if self.use_confidence_scaling and ensemble_std is not None:
            w /= (1.0 + float(ensemble_std))
            conf_applied = True

        # clip
        weight = float(np.clip(w, self.w_min, self.w_max))

        # VIX circuit breaker — applied last, overrides everything
        vix_breaker_fired = False
        if self.vix_cb_enabled and signals is not None:
            current_vix = float(signals.get('vix', 0.0))
            if current_vix >= self.vix_cb_threshold:
                weight = 0.0
                vix_breaker_fired = True

        return {
            "weight": weight,
            "w_pre_clip": float(w),
            "w_vol_target": float(self.sigma_target / sigma_hat),
            "sigma_hat_ann": sigma_hat,
            "regime_name": regime_name,
            "regime_idx": regime_idx,
            "regime_size": regime_size,
            "p_crisis": p_crisis,
            "adx_override": adx_override_applied,
            "confidence_scaling": conf_applied,
            "vix_breaker": vix_breaker_fired,
        }

    def compute_batch(
        self,
        log_rv_preds: np.ndarray,
        regime_probs_batch: np.ndarray,
        signals_list: list[dict] | None = None,
        ensemble_stds: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Vectorized version — compute position sizes for N dates.

        Args:
            log_rv_preds:       (N,)
            regime_probs_batch: (N, 6)
            signals_list:       list of N signal dicts (optional)
            ensemble_stds:      (N,) optional
        """
        results = []
        for i in range(len(log_rv_preds)):
            sig = signals_list[i] if signals_list is not None else None
            std = float(ensemble_stds[i]) if ensemble_stds is not None else None
            results.append(self.compute(
                log_rv_pred= float(log_rv_preds[i]),
                regime_probs=regime_probs_batch[i],
                signals = sig,
                ensemble_std=std
            ))
        return results
    
    def transaction_cost(self, w_prev: float, w_next: float) -> float:
        """Round-trip cost in decimal for a position change."""
        return abs(w_next - w_prev) * self.cost_bps

    def should_rebalance(self, w_prev: float, w_next:float) -> bool:
        """Skip rebalancing if position change is below threshold (save costs)."""
        return abs(w_next - w_prev) >= self.min_trade_threshold
