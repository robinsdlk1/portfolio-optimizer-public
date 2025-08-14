import pandas as pd
import numpy as np

def compute_realized_metrics(returns: pd.Series, freq: int = 252) -> dict:
    mu = returns.mean() * freq
    sigma = returns.std() * np.sqrt(freq)
    sharpe = mu / sigma if sigma > 0 else np.nan

    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = 1 - cum / roll_max
    max_dd = drawdown.max()

    return {
        "Annualized Return": mu,
        "Annualized Volatility": sigma,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

def compare_realized_metrics(returns_matrix: pd.DataFrame, weights: pd.Series, freq: int = 252) -> pd.DataFrame:
    port_ret = (returns_matrix @ weights).dropna()
    ew_ret = (returns_matrix @ (np.ones(len(weights)) / len(weights))).dropna()

    port_metrics = compute_realized_metrics(port_ret, freq)
    ew_metrics = compute_realized_metrics(ew_ret, freq)

    df = pd.DataFrame([port_metrics, ew_metrics], index=["Optimized", "Equal Weighted"]).T
    return df