import pandas as pd
import numpy as np

def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame):
    return np.sqrt(weights.T @ cov @ weights)

def portfolio_return(weights: pd.Series, mu: pd.Series):
    return weights.T @ mu

def portfolio_sharpe_ratio(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.0):
    port_ret = portfolio_return(weights, mu)
    port_vol = portfolio_volatility(weights, cov)
    return (port_ret - risk_free_rate) / port_vol if port_vol > 0 else np.nan

def herfindahl_index(weights: pd.Series):
    return np.sum(np.square(weights))

def effective_number_of_assets(weights: pd.Series):
    hhi = herfindahl_index(weights)
    return 1.0 / hhi if hhi > 0 else np.nan

def weight_dispersion(weights: pd.Series):
    return np.std(weights)

def concentration_ratio(weights: pd.Series, top_n = 5):
    abs_weights = weights.abs().sort_values(ascending = False)
    return abs_weights.iloc[:top_n].sum()