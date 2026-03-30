try:
    from .dolt_client import DoltClient
except ImportError:
    DoltClient = None  # doltpy not installed; data fetching unavailable
from .symbol_map import SymbolMap
from .underlying import fetch_underlying, compute_log_returns
from .market_state import fetch_market_state
from .vol_history import compute_vol_history_features
from .cleaning import (standardize_call_put, rename_iv_column, filter_quality,
                       filter_moneyness, filter_maturity, detect_obs_frequency, detect_gaps)
try:
    from .surface_builder import build_surface
except (FileNotFoundError, KeyError):
    build_surface = None  # configs not found at relative path; run from project root
try:
    from .feature_eng import build_returns_tensor, build_vol_history_vector, build_market_state_vector
except (FileNotFoundError, KeyError):
    build_returns_tensor = build_vol_history_vector = build_market_state_vector = None
from .targets import compute_forward_rv, compute_tail_indicator, compute_regime_label
from .dataset import SurfaceAlphaDataset