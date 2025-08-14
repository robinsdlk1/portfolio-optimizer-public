from typing import Callable, Any, Dict, Optional, Type
import inspect

import numpy as np
import pandas as pd

from portfolio_optimizer.optimizers.base import OptimizerRole
from portfolio_optimizer.risk_models.sample import SampleCovariance

def make_optimizer_strategy(*, optimizer_cls: Type, window: int, X_source: Optional[pd.DataFrame] = None, **opt_kwargs: Any) -> Callable[[pd.DataFrame, Optional[pd.Series]], pd.Series]:

    role: OptimizerRole = getattr(optimizer_cls, "ROLE", OptimizerRole.NAIVE)
    fit_sig = inspect.signature(optimizer_cls.fit)
    accepts = set(fit_sig.parameters)

    def _strategy(history: pd.DataFrame, w_prev: Optional[pd.Series]) -> pd.Series:

        window_hist = history.iloc[-window:] if window > 0 else history
        window_hist = window_hist.fillna(0.0)        # closed‑day patch

        groups: Optional[pd.Series] = None
        X_sub: Optional[pd.DataFrame] = None
        if X_source is not None:
            X_sub = X_source.reindex(window_hist.columns)
            if "Sector" in X_sub.columns:
                groups = X_sub["Sector"].astype(str)

        opt = optimizer_cls(**opt_kwargs)

        risk_model = SampleCovariance()
        cov = risk_model.fit(window_hist).get_covariance()
        mu = window_hist.mean().fillna(0.0) # closed‑day patch

        kw: Dict[str, Any] = {}
        if "R" in accepts:
            kw["R"] = window_hist
        if "mu" in accepts:
            kw["mu"] = mu
        if "cov" in accepts:
            kw["cov"] = cov
        if "risk_model" in accepts:
            kw["risk_model"] = risk_model
        if "groups" in accepts:
            kw["groups"] = groups
        if "w_prev" in accepts:
            kw["w_prev"] = w_prev
        if "X" in accepts and X_sub is not None:
            kw["X"] = X_sub

        w_next: pd.Series = opt.fit(**kw)

        w_next = w_next.reindex(window_hist.columns).fillna(0.0)

        target_sum = 1.0
        constraints = getattr(opt, "constraints", None)
        if constraints is not None and hasattr(constraints, "weight_sum"):
            target_sum = float(constraints.weight_sum)

        s = float(w_next.sum())
        if not np.isfinite(s) or s == 0.0:
            w_next = pd.Series(np.full(len(window_hist.columns), 1.0 / len(window_hist.columns)),
                            index=window_hist.columns)
        else:
            if abs(s - target_sum) > 1e-8:
                w_next *= (target_sum / s)

        return w_next

    return _strategy

def equal_weight_strategy(history: pd.DataFrame, _: pd.Series) -> pd.Series:
    n = history.shape[1]
    return pd.Series(np.full(n, 1.0 / n), index=history.columns)