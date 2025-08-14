import pandas as pd
import numpy as np

from .base import CovarianceEstimator

class SampleCovariance(CovarianceEstimator):

    def fit(self, returns: pd.DataFrame, horizon: int = 1, **kwargs) -> "SampleCovariance":
        if returns.isnull().values.any():
            returns = returns.dropna(how = "any")

        daily_cov = returns.cov()

        daily_cov = daily_cov.fillna(0.0)
        daily_cov = (daily_cov + daily_cov.T) / 2
        np.fill_diagonal(daily_cov.values, daily_cov.values.diagonal() + 1e-8)

        scale = self.get_scaler(horizon)
        self._sigma = daily_cov * scale
        return self

    def get_covariance(self) -> pd.DataFrame:
        if self._sigma is None:
            raise RuntimeError("Must call fit() before get_covariance().")
        return self._sigma