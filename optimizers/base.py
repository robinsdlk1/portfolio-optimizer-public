import pandas as pd
import numpy as np
import cvxpy as cp
from enum import Enum, auto

class OptimizerRole(Enum):
    FACTOR = auto()
    NAIVE = auto()
    CLASSIC = auto()
    
class Constraints:
    def __init__(self, *, long_only: bool = True, min_weight: float | pd.Series = 0.0, max_weight: float | pd.Series = 1.0, weight_sum: float = 1.0, group_min: dict[str, float] | None = None, group_max: dict[str, float] | None = None, gross_cap: float | None = None, turnover_cap: float | None = None, max_assets: int | None = None, group_mode: str = "net", mask_caps: list[tuple[np.ndarray, float, str]] | None = None):
        self.long_only = bool(long_only)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_sum = float(weight_sum)
        self.group_min = group_min or {}
        self.group_max = group_max or {}
        self.gross_cap = float(gross_cap) if gross_cap is not None else None
        self.max_assets = int(max_assets) if max_assets is not None else None
        self.turnover_cap = (float(turnover_cap) if turnover_cap is not None else None)
        if group_mode not in {"net", "long_only"}:
            raise ValueError("group_mode must be 'net' or 'long_only'")
        self.group_mode = group_mode
        self.mask_caps = list(mask_caps or []) 

    def build(self, w: cp.Variable, asset_index: pd.Index, *, groups: pd.Series | None = None, w_prev: pd.Series | None = None) -> list[cp.Constraint]:
        cons: list[cp.Constraint] = []

        if self.long_only:
            cons.append(w >= 0)
        if isinstance(self.min_weight, pd.Series):
            cons.append(w >= self.min_weight.loc[asset_index].values)
        else:
            cons.append(w >= self.min_weight)

        if isinstance(self.max_weight, pd.Series):
            cons.append(w <= self.max_weight.loc[asset_index].values)
        else:
            cons.append(w <= self.max_weight)

        cons.append(cp.sum(w) == self.weight_sum)

        if self.gross_cap is not None:
            cons.append(cp.norm1(w) <= self.gross_cap)

        if self.group_max and groups is not None:
            for g, cap in self.group_max.items():
                mask = (groups == g).values.astype(float)
                if self.group_mode == "long_only":
                    cons.append(mask @ cp.pos(w) <= cap)
                else:
                    cons += [mask @ w <= cap, mask @ w >= -cap]

        if self.group_min and groups is not None:
            for lbl, lo in self.group_min.items():
                if lbl not in groups.values:
                    continue
                mask = (groups == lbl).astype(float).values
                if self.group_mode == "long_only":
                    cons.append(mask @ cp.pos(w) >= lo)
                else:
                    cons.append(mask @ w >= lo)

        if self.max_assets is not None:
            n = len(asset_index)
            z = cp.Variable(n, boolean = True)

            if isinstance(self.max_weight, pd.Series):
                ub = self.max_weight.loc[asset_index].values
            else:
                ub = np.full(n, self.max_weight)

            if isinstance(self.min_weight, pd.Series):
                lb = self.min_weight.loc[asset_index].values
            else:
                lb = np.full(n, float(self.min_weight))

            cons += [w <= cp.multiply(ub, z), w >= cp.multiply(lb, z), cp.sum(z) <= self.max_assets]

        if self.turnover_cap is not None and w_prev is not None:
            if isinstance(w_prev, pd.Series):
                prev_vals = w_prev.loc[asset_index].values
            else:
                prev_vals = np.asarray(w_prev)
            cons.append(cp.norm1(w - prev_vals) <= self.turnover_cap)

        if self.mask_caps:
            for mask, cap, mode in self.mask_caps:
                cap = float(cap)
                if mode == "long_only":
                    cons.append(mask @ cp.pos(w) <= cap)
                else:
                    cons += [mask @ w <= cap, mask @ w >= -cap]

        return cons
    
class BaseOptimizer:
    def __init__(self, constraints = None):
        if constraints is None:
            self.constraints = Constraints()
        else:
            self.constraints = constraints

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the fit method.")