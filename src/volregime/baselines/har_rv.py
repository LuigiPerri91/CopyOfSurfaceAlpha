"""
Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009).
RV_{t+h} = c + β_d·RV_t + β_w·RV_{t-5:t} + β_m·RV_{t-21:t} + ε
Simple OLS — deceptively strong benchmark.
"""

from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score

class HARRVBaseline:
    """Heterogeneours Autoregressive RV: regress forward RV on daily, weekly, monthly trailing RV."""
    def __init__(self):
        self.reg = LinearRegression()
        self.coef_ = None
        self.intercept_ = None
        self.train_r2_ = None

    def _build_features(self, rv_series: np.ndarray | pd.Series) -> np.ndarray:
        s = pd.Series(rv_series) if not isinstance(rv_series, pd.Series) else rv_series
        rv_1d = s.to_numpy()
        rv_5d = s.rolling(5, min_periods=1).mean().to_numpy()
        rv_21d = s.rolling(21, min_periods=1).mean().to_numpy()
        return np.stack([rv_1d, rv_5d, rv_21d], axis=1)

    def fit(self, rv_series: np.ndarray, forward_rv:np.ndarray) -> None:
        """
        Args:
            rv_series:  1-D trailing realized vol, length N.
            forward_rv: 1-D forward realized vol targets, length N.
        """
        X = self._build_features(rv_series)[20:]
        y = forward_rv[20:]
        self.reg.fit(X,y)
        self.coef_ = self.reg.coef_
        self.intercept_ = self.reg.intercept_
        self.train_r2_ = r2_score(y, self.reg.predict(X))

    def predict(self, rv_1d: float, rv_5d: float, rv_21d: float) -> float:
        pred = self.reg.predict([[rv_1d,rv_5d,rv_21d]])[0]
        return float(max(pred,0.0))

    def predict_series(self,rv_series: np.ndarray) -> np.ndarray:
        """First 20 values will be NaN (insufficient warmup history)."""
        X = self._build_features(rv_series)
        preds = np.clip(self.reg.predict(X), 0.0, None)
        result = np.full(len(rv_series), np.nan, dtype = np.float32)
        result[20:] = preds[20:].astype(np.float32)
        return result

