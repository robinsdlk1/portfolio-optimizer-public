import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Callable, Union

def compute_cumulative_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    opt_series = (asset_returns @ weights).dropna()
    ew_weights = np.full(len(asset_returns.columns), 1 / len(asset_returns.columns))
    ew_series = (asset_returns @ ew_weights).dropna()

    aligned = pd.concat({"Optimised": opt_series, "Equal-Weight": ew_series}, axis = 1).dropna()
    return (1 + aligned).cumprod()

def compute_cagr(portfolio_returns: pd.Series, freq: int = 252) -> float:
    if portfolio_returns.empty:
        return np.nan
    cum = (1 + portfolio_returns).cumprod()
    n_years = len(cum) / freq
    return cum.iat[-1] ** (1 / n_years) - 1

def plot_cumulative_returns(cum_returns: pd.DataFrame, title: str = "Cumulative Portfolio Returns") -> go.Figure:
    fig = go.Figure()
    for col in cum_returns.columns:
        fig.add_trace(go.Scatter(x = cum_returns.index, y = cum_returns[col], name = col))
    fig.update_layout(title = title, xaxis_title = "Date", yaxis_title = "Cumulative Return", template = "plotly_white")
    return fig

def plot_drawdowns(portfolio_returns: pd.Series, title: str = "Portfolio Drawdown") -> go.Figure:
    cum = (1 + portfolio_returns).cumprod()
    roll_max = cum.cummax()
    drawdown = 1 - cum / roll_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy"))
    fig.update_layout(title = title, xaxis_title = "Date", yaxis_title = "Drawdown", template = "plotly_white")
    return fig

def plot_rolling_sharpe(portfolio_returns: pd.Series, window: int = 63, title: str = "Rolling Sharpe Ratio") -> go.Figure:
    mean_ret = portfolio_returns.rolling(window).mean() * 252
    vol_ret = portfolio_returns.rolling(window).std() * np.sqrt(252)
    sharpe = mean_ret / vol_ret

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = sharpe.index, y = sharpe, name = "Rolling Sharpe"))
    fig.update_layout(title = title, xaxis_title = "Date", yaxis_title = "Sharpe Ratio", template = "plotly_white")
    return fig

def plot_return_histogram(portfolio_returns: pd.Series) -> go.Figure:
    fig = px.histogram(portfolio_returns, nbins = 50, marginal = "box", title = "Daily Return Distribution", labels = {"value": "Return"}, opacity = 0.75)
    fig.update_layout(template = "plotly_white")
    return fig

def plot_correlation_heatmap(asset_returns: pd.DataFrame) -> go.Figure:
    corr = asset_returns.corr()
    fig = px.imshow(corr, text_auto = ".2f", title = "Asset Correlation Matrix", aspect = "auto", color_continuous_scale = "RdBu", zmin = -1, zmax = 1)
    fig.update_layout(template = "plotly_white")
    return fig

def plot_rolling_metric(ret_series: pd.Series, window: int, metric_name: str, func: Callable[[Union[pd.Series, np.ndarray]], float], freq: int = 252) -> go.Figure:
    annualise = metric_name.lower().startswith(("return", "sharpe"))
    rolled = ret_series.rolling(window).apply(func)
    if annualise:
        rolled *= freq

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = rolled.index, y = rolled.values, name = metric_name))
    fig.update_layout(title = f"{metric_name} (Rolling {window}d)", xaxis_title = "Date", yaxis_title = metric_name, template = "plotly_white")
    return fig

def rolling_sharpe(window_returns: Union[pd.Series, np.ndarray]) -> float:
    series = pd.Series(window_returns)
    vol = series.std()
    return series.mean() / vol if vol > 0 else np.nan


def rolling_max_dd(window_returns: Union[pd.Series, np.ndarray]) -> float:
    cumulative = (1 + pd.Series(window_returns)).cumprod()
    dd = 1 - cumulative / cumulative.cummax()
    return dd.max() if not dd.empty else np.nan

def plot_portfolio_composition_over_time(weights_df: pd.DataFrame) -> go.Figure:
    df = weights_df.copy()
    row_sums = df.sum(axis = 1)
    df = df.div(row_sums, axis=0).fillna(0.0)

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x = df.index, y = df[col], mode = 'lines', stackgroup = 'one', name = col, hoverinfo = 'x+y+name'))

    fig.update_layout(title = "Portfolio Composition Over Time", xaxis_title = "Date", yaxis_title = "Portfolio Weight", hovermode = "x unified", template = "plotly_white", legend_title = "Asset")

    return fig