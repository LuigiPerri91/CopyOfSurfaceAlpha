import numpy as np 

class PersistanceBaseline:
    """RV_{t+h} = RV_t (trailing realized vol carried forward)."""
    def fit(self, *args, **kwargs):
        pass

    def predict(self,trailing_rv: np.ndarray) -> np.ndarray: 
        return np.asarray(trailing_rv, dtype= np.float32) # identity

class ATMIVCarryForward:
    """
    Use ATM implied vol as the forward RV forecast.
    Extracts from channel 0 (iv), center moneyness column, shortest valid maturity.
    """
    def __init__(self, n_moneyness_bins: int =20, atm_bin_idx: int | None = None):
        self.atm_bin_idx = atm_bin_idx if atm_bin_idx is not None else n_moneyness_bins // 2

    def fit(self, *args, **kwargs):
        pass

    def predict(self, surface_tensors: np.ndarray) -> np.ndarray:
        """
        Args:
            surface_tensors: (N, 6, H, W)
        Returns:
            (N,) ATM IV from shortest observed maturity
        """
        iv_channel = surface_tensors[:, 0, :, :]
        mask_channel = surface_tensors[:,2,:,:]
        atm_col = self.atm_bin_idx

        forecasts = []
        for i in range(len(surface_tensors)):
            iv_col = iv_channel[i, :, atm_col]
            mask_col = mask_channel[i, :, atm_col]
            valid = np.where(mask_col > 0)[0]
            if len(valid) > 0:
                forecasts.append(float(iv_col[valid[0]]))
            else:
                nonzero = iv_channel[i][mask_channel[i] > 0]
                forecasts.append(float(nonzero.mean()) if len(nonzero) > 0 else 0.0)
        
        return np.array(forecasts, dtype=np.float32)

