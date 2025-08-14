from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

def plot_weights_bar(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    ordered = weights.sort_values(ascending = False)
    fig = go.Figure(go.Bar(x = ordered.index, y = ordered.values))
    fig.update_layout(title = title, xaxis_title = "Assets", yaxis_title = "Weight", template = "plotly_white")
    return fig

def plot_weight_pie(weights: pd.Series, title: str = "Portfolio Allocation") -> go.Figure:
    fig = go.Figure(go.Pie(labels=weights.index, values=weights.values))
    fig.update_layout(title=title, template="plotly_white")
    return fig

def compute_risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    port_var = float(weights.T @ cov.values @ weights)
    if port_var == 0:
        return pd.Series(0.0, index = weights.index)
    marginal = cov @ weights
    contrib = weights * marginal
    return contrib / port_var

def plot_risk_contributions(weights: pd.Series, cov: pd.DataFrame, title: str = "Risk Contributions") -> go.Figure:
    rc = compute_risk_contributions(weights, cov).sort_values(ascending = False)
    fig = go.Figure(go.Bar(x = rc.index, y=rc.values))
    fig.update_layout(title = title, xaxis_title = "Assets", yaxis_title = "% of Total Portfolio Risk", template = "plotly_white")
    return fig

DEFAULT_GRAPH_TYPES: tuple[str, ...] = ("weights", "pie", "risk")

def build_optimizer_graphs(weights_dict: Dict[str, pd.Series], *, cov: pd.DataFrame | None, factors: pd.DataFrame) -> Dict[str, List[Tuple[str, go.Figure]]]:
    graphs: Dict[str, List[Tuple[str, go.Figure]]] = {g: [] for g in DEFAULT_GRAPH_TYPES}

    for label, w in weights_dict.items():
        graphs["weights"].append((label, plot_weights_bar(w, title=f"{label} - Weights")))
        graphs["pie"].append((label, plot_weight_pie(w, title=f"{label} - Allocation")))
        if cov is not None:
            graphs["risk"].append((label, plot_risk_contributions(w, cov, title=f"{label} - Risk Contributions")))
    return graphs