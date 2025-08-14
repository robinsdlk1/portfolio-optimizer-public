import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_optimizer.optimizers.base import BaseOptimizer, OptimizerRole, Constraints

class MinVolOptimizer(BaseOptimizer):
    ROLE = OptimizerRole.CLASSIC

    def fit(self, cov: pd.DataFrame, *, groups: pd.Series | None = None, w_prev: pd.Series | None = None) -> pd.Series:
        print("Fitting MinVolOptimizer (classical).")

        n = cov.shape[0]
        w = cp.Variable(n)
        
        objective = cp.Minimize(cp.quad_form(w, cov.values))
        constraints = self.constraints.build(w, cov.columns, groups = groups, w_prev = w_prev)
        
        prob = cp.Problem(objective, constraints)
        chosen_solver = "OSQP" if self.constraints.max_assets is None else "ECOS_BB"
        prob.solve(solver = chosen_solver)
        
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solver status: {prob.status}")

        w_val = np.asarray(w.value).flatten()
        if self.constraints.long_only:
            w_val = np.maximum(w_val, 0.0)
        tgt = self.constraints.weight_sum
        s = w_val.sum()
        if s > 0 and abs(s - tgt) > 1e-8:
            w_val *= tgt / s

        return pd.Series(w_val, index = cov.columns)

class MaxSharpeOptimizer(BaseOptimizer):
    ROLE = OptimizerRole.CLASSIC

    def __init__(self, *, risk_aversion: float = 1.0, constraints: Constraints | None = None):
        BaseOptimizer.__init__(self, constraints)
        self.risk_aversion = float(risk_aversion)

    def fit(self, mu: pd.Series, cov: pd.DataFrame, *, groups: pd.Series | None = None, w_prev: pd.Series | None = None) -> pd.Series:
        print("Fitting MaxSharpeOptimizer (classical).")
        
        n = cov.shape[0]
        w = cp.Variable(n)
        
        lam = self.risk_aversion
        objective = cp.Minimize(lam * cp.quad_form(w, cov.values) - mu.values @ w)
        constraints = self.constraints.build(w, cov.columns, groups = groups, w_prev = w_prev)

        prob = cp.Problem(objective, constraints)
        chosen_solver = "OSQP" if self.constraints.max_assets is None else "ECOS_BB"
        prob.solve(solver=chosen_solver)
        
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solver status: {prob.status}")
        
        w_val = np.asarray(w.value).flatten()

        if self.constraints.long_only:
            w_val = np.maximum(w_val, 0)

        target_sum = self.constraints.weight_sum
        s = w_val.sum()
        if s > 0 and abs(s - target_sum) > 1e-8:
            w_val *= target_sum / s

        return pd.Series(w_val, index = mu.index)

class TargetReturnOptimizer(BaseOptimizer):
    ROLE = OptimizerRole.CLASSIC

    def __init__(self, target_return: float = 0.05, constraints: Constraints | None = None):
        BaseOptimizer.__init__(self, constraints)
        self.target_return = float(target_return)

    def fit(self, mu: pd.Series, cov: pd.DataFrame, groups: pd.Series | None = None, w_prev: pd.Series | None = None) -> pd.Series:
        print(f"Fitting TargetReturnOptimizer (classical) with target mu = {self.target_return:.2f}")
        
        n = cov.shape[0]
        w = cp.Variable(n)
        
        objective = cp.Minimize(cp.quad_form(w, cov.values))
        constraints = self.constraints.build(w, cov.columns, groups = groups, w_prev = w_prev)
        constraints.append(mu.values @ w >= self.target_return)
        
        prob = cp.Problem(objective, constraints)
        chosen_solver = "OSQP" if self.constraints.max_assets is None else "ECOS_BB"
        prob.solve(solver = chosen_solver)
        
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP solver status: {prob.status}")
        
        w_val = np.asarray(w.value).flatten()
        if self.constraints.long_only:
            w_val = np.maximum(w_val, 0.0)
        tgt = self.constraints.weight_sum
        s = w_val.sum()
        if s > 0 and abs(s - tgt) > 1e-8:
            w_val *= tgt / s

        return pd.Series(w_val, index = mu.index)