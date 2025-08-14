from typing import Optional

import numpy as np
import pandas as pd


class CovarianceEstimator:
    def __init__(self) -> None:
        self._sigma: Optional[pd.DataFrame] = None

    def fit(self, returns: pd.DataFrame, **kwargs) -> "CovarianceEstimator":
        raise NotImplementedError("fit() must be implemented in subclasses.")

    def get_covariance(self) -> pd.DataFrame:
        raise NotImplementedError("get_covariance() must be implemented in subclasses.")
    
    def get_correlation(self) -> pd.DataFrame:
        if self._sigma is None:
            raise RuntimeError("Call fit() before get_correlation().")

        std = np.sqrt(np.diag(self._sigma))
        corr = self._sigma.divide(std, axis=0).divide(std, axis=1)
        return corr

    def get_scaler(self, horizon: int) -> float:
        return float(horizon)