"""
Gradient-boosted tree baseline (LightGBM / XGBoost).
Strongest tabular baseline — same information as the deep model
but in hand-engineered form. If the deep model beats this, the
ViT surface encoding is genuinely adding value.
"""

import numpy as np 
import pandas as pd
import logging
import lightgbm as lgb 
import xgboost as xgb

logger = logging.getLogger(__name__)

def extract_surface_features(surface_tensor: np.ndarray, n_moneyness_bins: int =20) -> dict:
    """
    Hand-engineer scalar features from a (6, H, W) surface tensor.
    Channels: iv(0), spread_norm(1), mask(2), staleness(3), delta(4), vega(5)
    """
    iv = surface_tensor[0]
    mask = surface_tensor[2]
    H, W = iv.shape
    atm_col = W // 2
    otm_put_col = W // 4
    otm_call_col = 3 * W // 4

    iv_masked = np.where(mask > 0, iv, np.nan)

    def col_mean(col_idx):
        col = iv_masked[:, col_idx]
        valid = col[~np.isnan(col)]
        return float(vaild.mean()) if len(valid) >0 else 0.0

    def row_mean(row_idx):
        row = iv_masked[row_idx, :]
        valid = row[~np.isnan(row)]
        return float(valid.mean()) if len(valid) >0 else 0.0

    atm_vol = col_mean(atm_col)
    put_25d_vol = col_mean(otm_put_col)
    call_25d_vol = col_mean(otm_call_col)
    short_term_vol = row_mean(0)
    long_term_vol = row_mean(H - 1)

    return {
        "sf_atm_vol":        atm_vol,
        "sf_put_25d":        put_25d_vol,
        "sf_call_25d":       call_25d_vol,
        "sf_skew":           put_25d_vol - call_25d_vol,
        "sf_term_slope":     long_term_vol - short_term_vol,
        "sf_butterfly":      (put_25d_vol + call_25d_vol) / 2 - atm_vol,
        "sf_short_term_vol": short_term_vol,
        "sf_long_term_vol":  long_term_vol,
        "sf_obs_density":    float(mask.mean()),
    }

def build_boosting_features(surface_tensor: np.ndarray, returns_tensor: np.ndarray, vol_history_vector: np.ndarray, 
                            market_state_vector: np.ndarray, vol_history_feature_names: list[str] | None = None, market_state_feature_names: list[str] | None = None) -> dict:
    """Combine surafce stats + returns stats + vol-history + market state"""
    features = {}
    features.update(extract_surface_features(surface_tensor))

    log_returns = returns_tensor[:,0]
    rolling_std = float(log_returns[-21:].std()) if len(log_returns) >= 21 else 0.0
    jump_count = int((np.abs(log_returns) > 2.5 * rolling_std).sum()) if rolling_std > 0 else 0

    features['ret_rv_5d'] = log_returns.rolling(5, min_periods=1).std()
    features['ret_rv_10d'] = log_returns.rolling(10, min_periods=1).std()
    features['ret_rv_21d'] = log_returns.rolling(21, min_periods=1).std()
    features['ret_trailing_21d'] = log_returns.rolling(21, min_periods=1).sum()
    features['ret_jump_count_21d'] = float(jump_count)

    vh_names = vol_history_feature_names or [
        "iv_rank", "hv_rank", "vol_risk_premium",
        "iv_momentum_short", "iv_momentum_medium",
        "hv_momentum_short", "hv_momentum_medium",
        "days_since_iv_year_high", "days_since_iv_year_low",
        "days_since_hv_year_high", "days_since_hv_year_low",
    ]

    for i, name in enumerate(vh_names):
        features[f'vh_{name}'] = float(vol_history_vector[i]) if i < len(vol_history_vector) else np.nan

    mkt_names = market_state_feature_names or ["vix", "spy_return", "risk_free_rate"]
    for i, name in enumerate(mkt_names):
        features[f"mkt_{name}"] = float(market_state_vector[i]) if i < len(market_state_vector) else np.nan
    
    return features

class BoostingBaseline:
    """LightGBM or XGBoost trained on build_boosting_features() output."""
    def __init__(self, model_type:str = "lightgbm", n_estimators: int =500, max_depth:int =6, lr :float = 0.05):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.model = None
        self.feature_names = None

    def fit(self, features_df: pd.DataFrame, targets: np.ndarray) -> None:
        self.feature_names = list(features_df.columns)
        X = np.nan_to_num(features_df.values.astype(np.float32), nan=0.0)
        y = targets.astype(np.float32)

        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.lr, n_jobs=-1, verbose=-1)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.lr, n_jobs=-1, verbose=0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X,y)
        logger.info("Boosting fit: %d samples, %d features", len(y), X.shape[1])

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X = np.nan_to_num(features_df.valeus.astype(np.float32),nan=0.0)
        return self.model.predict(X).astype(np.float32)

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None or self.feature_names is None:
            return {}
        pairs = sorted(zip(self.feature_names, self.model.feature_importances_.tolist()), key=lambda x: x[1], reverse=True)
        return dict(pairs)