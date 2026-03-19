from yaml import warnings
import numpy as np 
from arch import arch_model
import warnings

class GARCHBaseline:
    """GARCH(1,1) — produces h-step-ahead annualized vol forecasts."""
    def __init__(self, p:int =1, q:int =1, horizon:int =21):
        self.p = p
        self.q = q
        self.horizon = horizon
        self.result = None

    def fit(self, returns: np.ndarray) -> None:
        """
        Args:
            returns: 1-D array of daily log returns.
        """
        returns_pct = returns * 100
        model = arch_model(returns_pct, vol='Garch', p=self.p,q=self.q, dist='normal',rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.result = model.fit(disp='off', show_warning=False)
    
    def predict(self, n_ahead: int | None = None) -> float:
        if self.result is None:
            raise RuntimeError('Call fit() before predict().')
        h = n_ahead or self.horizon
        forecast = self.result.forecast(horizon=h, reindex=False)
        # return sqrt of h-step-ahead variance sum
        variance = forecast.variance.iloc[-1].values[:h]
        return float(np.sqrt(variance.sum()) / 100)

    def predict_series(self, trailing_returns_list: list[np.ndarray]) -> np.ndarray:
        return np.array([self.predict() for _ in trailing_returns_list], dtype=np.float32)

class EGARCHBaseline:
    """EGARCH - captures asymmetric leverage effect."""
    def __init__(self, p:int =1, q:int =1, horizon:int =21):
        self.p = p
        self.q = q
        self.horizon = horizon
        self.result = None

    def fit(self, returns: np.ndarray) -> None:
        """
        Args:
            returns: 1-D array of daily log returns.
        """
        returns_pct = returns * 100
        model = arch_model(returns_pct, vol='EGARCH', p=self.p,q=self.q, dist='normal')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.result = model.fit(disp='off', show_warning=False)
    
    def predict(self, n_ahead: int | None = None) -> float:
        if self.result is None:
            raise RuntimeError('Call fit() before predict().')
        h = n_ahead or self.horizon
        forecast = self.result.forecast(horizon=h, reindex=False)
        # return sqrt of h-step-ahead variance sum
        variance = forecast.variance.iloc[-1].values[:h]
        return float(np.sqrt(variance.sum()) / 100)

    def predict_series(self, trailing_returns_list: list[np.ndarray]) -> np.ndarray:
        return np.array([self.predict() for _ in trailing_returns_list], dtype=np.float32)
