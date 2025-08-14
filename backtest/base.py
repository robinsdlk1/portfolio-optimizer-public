import numpy as np
import pandas as pd
from typing import Callable, Dict, Any

class BacktestEngine:
    def __init__(self, returns: pd.DataFrame, initial_weights: pd.Series, strategy: Callable[[pd.DataFrame, pd.Series], pd.Series], schedule: pd.DatetimeIndex, cost_bps: float = 0.0, initial_capital: float = 1.0):
        self.returns = returns.copy().fillna(0.0) # closed‑day patch
        self.cost = cost_bps / 10_000
        self.schedule = pd.DatetimeIndex(schedule)
        self.strategy = strategy
        self.capital_0 = initial_capital
        self.w0 = initial_weights.reindex(returns.columns).fillna(0.0)

        self._rp: list[float] = []
        self._w_path: list[np.ndarray] = []

    def run(self) -> Dict[str, Any]:
        w = self.w0.values.copy()
        capital = self.capital_0

        for t, date in enumerate(self.returns.index):
            if t > 0 and date in self.schedule:
                target = self.strategy(self.returns.iloc[:t], pd.Series(w, index=self.returns.columns))
                target = target.reindex(self.returns.columns).fillna(0.0).values
                turnover = np.abs(target - w).sum()
                capital -= capital * turnover * self.cost
                w = target

            self._w_path.append(w.copy())

            r_ptf = np.dot(w, self.returns.iloc[t].values)
            self._rp.append(r_ptf)
            capital *= 1 + r_ptf

        rp = pd.Series(self._rp, index=self.returns.index, name="portfolio_return")
        w_path = pd.DataFrame(self._w_path, index=self.returns.index, columns=self.returns.columns)
        value = (1 + rp).cumprod() * self.capital_0

        return {"returns": rp, "value": value, "weights": w_path}