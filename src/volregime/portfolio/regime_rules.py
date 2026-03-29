"""
Regime identification + position-sizing rules from backtest.yaml.

identify_regime():
    Rule-based regime classifier from OHLCV data.
    Matches the same logic as data/targets.py (compute_regime_label) but
    operates on a streaming OHLCV window for live/backtest use.

RegimeRules:
    Parses backtest.yaml regime_rules into a usable structure.
    Provides get_position_size(regime_name) and ADX override logic.

Six regimes (integer -> string):
    0 bull_quiet        1 bull_volatile
    2 bear_quiet        3 bear_volatile
    4 sideways_quiet    5 sideways_volatile
"""

import numpy as np 
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REGIME_INT_TO_NAME = {
    0: "bull_quiet",
    1: "bull_volatile",
    2: "bear_quiet",
    3: "bear_volatile",
    4: "sideways_quiet",
    5: "sideways_volatile",
}
REGIME_NAME_TO_INT = {v: k for k, v in REGIME_INT_TO_NAME.items()}

# OHLCV indicators
def _wilder_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Wilder's exponential smoothing — used by ATR and ADX."""
    n = len(arr)
    out = np.zeros(n)
    if n < window:
        return out
    out[window - 1] = arr[:window].mean()
    alpha = 1.0 / window
    for i in range(window, n):
        out[i] = (1 - alpha) * out[i - 1] + alpha * arr[i]
    return out


def compute_atr(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, window: int) -> np.ndarray:
    """Average True Range with Wilder smoothing."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _wilder_smooth(tr, window)


def compute_adx(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Average Directional Index (ADX) with Wilder smoothing.
    ADX < 20  -> no trend (sideways)
    ADX 25-50 -> trending
    ADX > 50  -> extreme trend
    """
    n   = len(high)
    tr  = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)

    for i in range(1, n):
        tr[i]  = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i]  - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up if (up > down and up > 0) else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0

    str_ = _wilder_smooth(tr,  window)
    spdm = _wilder_smooth(pdm, window)
    sndm = _wilder_smooth(ndm, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(str_ > 0, 100 * spdm / str_, 0.0)
        ndi = np.where(str_ > 0, 100 * sndm / str_, 0.0)
        dx = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)

    adx = np.zeros(n)
    start = 2 * window
    if n > start:
        adx[start] = dx[window:start + 1].mean()
        alpha = 1.0 / window
        for i in range(start + 1, n):
            adx[i] = (1 - alpha) * adx[i - 1] + alpha * dx[i]
    return adx

# regime identification
def identify_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    cfg: dict,
    vix: np.ndarray | None = None,
) -> tuple[int, dict]:
    """
    Rule-based regime identification for the LAST date in the OHLCV window.

    Args:
        high, low, close: arrays ending on today; min length ≈ 250
        cfg:              full merged config dict (reads cfg["backtest"])
        vix:              VIX close values (same length), optional

    Returns:
        regime_label: int 0-5
        signals:      dict of raw indicator values (for overlay + logging)
    """
    bt = cfg.get("backtest", cfg)
    ri = bt.get("regime_identification", {})

    ma_period = int(ri.get("ma_period", 200))
    adx_period = int(ri.get("adx_period", 14))
    adx_thr = ri.get("adx_thresholds", {})
    no_trend_thr = float(adx_thr.get("no_trend", 20))

    atr_s = int(ri.get("atr_short_period", 10))
    atr_l = int(ri.get("atr_long_period", 50))
    atr_thr = ri.get("atr_ratio_thresholds", {})
    quiet_thr = float(atr_thr.get("quiet", 0.75))
    volatile_thr = float(atr_thr.get("volatile", 1.25))

    vix_thr = ri.get("vix_thresholds", {})
    vix_extreme = float(vix_thr.get("extreme", 30))

    #  Direction: close vs 200-day MA
    ma_w = min(ma_period, len(close))
    sma = close[-ma_w:].mean()
    above_ma = bool(close[-1] > sma)

    #  Trend strength: ADX(14) 
    adx_vals = compute_adx(high, low, close, adx_period)
    adx_14 = float(adx_vals[-1])
    is_sideways = adx_14 < no_trend_thr

    #  Volatility state: ATR(10) / ATR(50) 
    atr_short_vals = compute_atr(high, low, close, atr_s)
    atr_long_vals = compute_atr(high, low, close, atr_l)
    atr_ratio = float(atr_short_vals[-1] / max(atr_long_vals[-1], 1e-10))
    is_quiet = atr_ratio < quiet_thr
    is_volatile = atr_ratio > volatile_thr

    #  VIX override 
    current_vix = float(vix[-1]) if vix is not None else float("nan")
    if vix is not None and current_vix > vix_extreme:
        is_volatile = True
        is_quiet = False

    #  Map to regime integer 
    if is_sideways:
        label = 5 if is_volatile else 4
    elif above_ma:
        label = 1 if is_volatile else 0
    else:
        label = 3 if is_volatile else 2

    signals = {
        "regime_label": label,
        "regime_name": REGIME_INT_TO_NAME[label],
        "above_200ma": above_ma,
        "adx_14": round(adx_14, 2),
        "atr_ratio": round(atr_ratio, 4),
        "is_sideways": is_sideways,
        "is_volatile": is_volatile,
        "is_quiet": is_quiet,
        "vix": round(current_vix, 2),
    }
    return label, signals

# Regime rules
@dataclass
class _RegimeRule:
    position_size: float
    atr_stop_multiplier: float
    strategy_note: str = ""

class RegimeRules:
    """
    Parses backtest.yaml regime_rules into callable sizing logic.

    Usage:
        rules = RegimeRules(cfg)
        size  = rules.get_position_size("bear_volatile")   # → 0.375
    """

    def __init__(self, cfg: dict):
        bt_cfg = cfg.get('backtest', cfg)
        raw = bt_cfg.get('regime_rules', {})

        self._rules: dict[str, _RegimeRule] = {}
        defaults = {
            "bull_quiet": _RegimeRule(1.0, 2.0),
            "bull_volatile": _RegimeRule(0.625, 2.5),
            "bear_quiet": _RegimeRule(1.0, 2.0),
            "bear_volatile": _RegimeRule(0.375, 3.0),
            "sideways_quiet": _RegimeRule(0.625,1.5),
            "sideways_volatile": _RegimeRule(0.0, 0.0),
        }
        for name , default in defaults.items():
            entry = raw.get(name, {})
            self._rules[name] = _RegimeRule(
                position_size= float(entry.get('position_size', default.position_size)),
                atr_stop_multiplier= float(entry.get('atr_stop_multiplier', default.atr_stop_multiplier)),
                strategy_note= str(entry.get('strategy_note', default.strategy_note))
            )
        
        # ADX override
        adx_cfg = bt_cfg.get('adx_override', {})
        self.adx_enabled = bool(adx_cfg.get("enabled", True))
        self.adx_threshold = float(adx_cfg.get("adx_threshold", 20))
        self.size_reduction = float(adx_cfg.get("size_reduction", 0.5))

        # Regime gating
        rg_cfg = bt_cfg.get("regime_gating", {})
        self.gating_enabled = bool(rg_cfg.get("enabled", True))
        self.crisis_regimes = set(rg_cfg.get("crisis_regimes", ["bear_volatile", "sideways_volatile"]))

    def get_position_size(self, regime_name: str) -> float:
        """Return the base position size for a given regime name."""
        if regime_name not in self._rules:
            logger.warning("Unknown regime '%s', defaulting to 0.5", regime_name)
            return 0.5
        return self._rules[regime_name].position_size

    def apply_adx_override(self, size: float, adx_14: float) -> tuple[float, bool]:
        """
        If ADX < threshold (weak trend with expanding vol), reduce position.
        Returns: (adjusted_size, override_applied)
        """
        if self.adx_enabled and adx_14 < self.adx_threshold:
            return size * (1.0 - self.size_reduction), True
        return size, False

    def is_crisis_regime(self, regime_name: str) -> bool:
        return regime_name in self.crisis_regimes