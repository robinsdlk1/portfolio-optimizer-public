from typing import Any, Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_optimizer.optimizers.base import BaseOptimizer, OptimizerRole, Constraints

class _BaseFactorOptimizer(BaseOptimizer):
    ROLE = OptimizerRole.FACTOR

    def __init__(self, betas: pd.DataFrame, intercept: pd.Series | float, preprocess: Callable[[pd.DataFrame], pd.DataFrame], *, risk_aversion: float = 1.0, constraints: Constraints | None = None):
        BaseOptimizer.__init__(self, constraints)

        self.betas = betas
        self.intercept = intercept
        self.preprocess = preprocess
        self.risk_aversion = float(risk_aversion)

    def _compute_mu_hat(self, X: pd.DataFrame) -> pd.Series:
        X_proc = self.preprocess.transform(X)
        mu_vec = (X_proc @ self.betas.values).ravel()
        mu_hat = pd.Series(mu_vec, index = X.index)

        if isinstance(self.intercept, pd.Series):
            mu_hat = mu_hat.add(self.intercept.reindex(X.index).fillna(0.0))
        else:
            mu_hat += float(self.intercept)

        return mu_hat

    def fit(self, X: pd.DataFrame, cov: pd.DataFrame, *, groups: pd.Series | None = None,) -> pd.Series:
        raise NotImplementedError

class FactorSharpeOptimizer(_BaseFactorOptimizer):
    ROLE = OptimizerRole.FACTOR

    def __init__(self, betas: pd.DataFrame, intercept: pd.Series | float, preprocess: Any, *, risk_aversion: float = 1.0, constraints: Constraints | None = None):
        _BaseFactorOptimizer.__init__(self, betas, intercept, preprocess, risk_aversion = risk_aversion, constraints = constraints)

    def fit(self, X: pd.DataFrame, cov: pd.DataFrame, *, groups: pd.Series | None = None, w_prev: pd.Series | None = None) -> pd.Series:
        print("Fitting FactorSharpeOptimizer.")

        mu = self._compute_mu_hat(X)
        n = len(mu)
        w = cp.Variable(n)

        lam = self.risk_aversion
        objective = cp.Minimize(lam * cp.quad_form(w, cov.values) - mu.values @ w)

        cons = self.constraints.build(w, mu.index, groups = groups, w_prev = w_prev)
        prob = cp.Problem(objective, cons)
        chosen = "OSQP" if self.constraints.max_assets is None else "ECOS_BB"
        prob.solve(solver = chosen)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solver status: {prob.status}")

        w_val = np.asarray(w.value).ravel()
        if self.constraints.long_only:
            w_val = np.maximum(w_val, 0.0)
        tgt = self.constraints.weight_sum
        s = w_val.sum()
        if s > 0 and abs(s - tgt) > 1e-8:
            w_val *= tgt / s

        self.weights_ = pd.Series(w_val, index = mu.index)
        return self.weights_

class FactorTargetReturnOptimizer(_BaseFactorOptimizer):
    ROLE = OptimizerRole.FACTOR

    def __init__(self, betas: pd.DataFrame, intercept: pd.Series | float, preprocess: Any, *, target_return: float = 0.05, risk_aversion: float = 1.0, constraints: Constraints | None = None):
        _BaseFactorOptimizer.__init__(self, betas, intercept, preprocess, risk_aversion  = risk_aversion, constraints = constraints)
        self.target_return = float(target_return)

    def fit(self, X: pd.DataFrame, cov: pd.DataFrame, *, groups: pd.Series | None = None, w_prev: pd.Series | None = None) -> pd.Series:
        print("Fitting FactorTargetReturnOptimizer.")

        mu = self._compute_mu_hat(X)
        n = len(mu)
        w = cp.Variable(n)

        lam = self.risk_aversion
        objective = cp.Minimize(lam * cp.quad_form(w, cov.values))

        cons = self.constraints.build(w, mu.index, groups = groups, w_prev = w_prev)
        cons.append(mu.values @ w >= self.target_return)

        prob = cp.Problem(objective, cons)
        chosen = "OSQP" if self.constraints.max_assets is None else "ECOS_BB"
        prob.solve(solver = chosen)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solver status: {prob.status}")

        w_val = np.asarray(w.value).ravel()
        if self.constraints.long_only:
            w_val = np.maximum(w_val, 0.0)
        tgt = self.constraints.weight_sum
        s = w_val.sum()
        if s > 0 and abs(s - tgt) > 1e-8:
            w_val *= tgt / s

        self.weights_ = pd.Series(w_val, index = mu.index)
        return self.weights_