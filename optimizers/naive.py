import pandas as pd
import numpy as np

from portfolio_optimizer.optimizers.base import BaseOptimizer, OptimizerRole

class EqualWeightOptimizer(BaseOptimizer):
    ROLE = OptimizerRole.NAIVE
    def fit(self, mu, cov):
        asset_names = list(cov.columns)
        n = len(asset_names)
        w = np.ones(n) / n
        return pd.Series(w, index = asset_names)

class VolatilityParityOptimizer(BaseOptimizer):
    ROLE = OptimizerRole.NAIVE
    def fit(self, mu, cov):
        asset_names = list(cov.columns)
        std_devs = np.sqrt(np.diag(cov.values))
        inv_vol = 1.0 / std_devs
        w = inv_vol / inv_vol.sum()
        return pd.Series(w, index = asset_names)