import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import statsmodels.api as sm 

def build_reg_summary(X_design, y) -> str:
    return sm.OLS(y, sm.add_constant(X_design, has_constant = "add")).fit(cov_type = "HC1").summary().as_text()

def build_residual_plot(preprocessor, betas: pd.Series, intercept: float, X: pd.DataFrame, y: pd.Series) -> go.Figure:
    X_proc = preprocessor.transform(X)
    y_hat = intercept + X_proc @ betas.loc[preprocessor.get_feature_names_out()].values
    resid = (y - y_hat).values

    fig = go.Figure()
    fig.add_trace(go.Histogram(x = resid, nbinsx = 30, name = "Residuals"))
    fig.update_layout(title = "Residual distribution", template = "plotly_dark")
    return fig

def build_beta_bar(betas):
    s = betas if isinstance(betas, pd.Series) else pd.Series(betas)
    s = s.reindex(s.abs().sort_values(ascending = False).index)

    df = (s.reset_index().rename(columns = {"index": "Feature", 0: "Beta"}).assign(ID = lambda d: range(len(d))))

    fig = px.bar(df, x = "ID", y = "Beta", hover_data = ["Feature", "Beta"], labels = {"ID": "Feature", "Beta": "Recovered Beta"}, title = "Recovered Coefficients (Beta)")
    fig.update_layout(template = "plotly_dark", xaxis = dict(tickmode = "array", tickvals = df["ID"], ticktext = df["Feature"].str.slice(0, 15) + "…"), xaxis_tickangle = -65)
    return fig, s

def build_beta_table(beta_sorted, title = "Recovered Beta") -> go.Figure:
    factors = beta_sorted.index.tolist()
    betas = [f"{v:.4f}" for v in beta_sorted.values]

    row_colors = ["rgba(245,245,245,1)" if i % 2 == 0 else "rgba(235,235,235,1)" for i in range(len(factors))]

    fig = go.Figure(
        go.Table(header = dict(values = ["<b>Factor</b>", "<b>Beta</b>"], fill_color = "#1f2c56", font = dict(color = "white", size = 13), align = "left", height = 28),
                 cells = dict(values = [factors, betas], fill_color = [row_colors, row_colors], align = "left", font = dict(size = 12), height = 24))
        )
    fig.update_layout(title = title, margin = dict(l = 0, r = 0, t = 35, b = 0), template = "plotly_white", height = min(400, 28 + 24 * len(factors)))
    return fig

def plot_pca_loading_bar(pca, pc_index: int, feature_names: list) -> go.Figure:
    loading = pca.components_[pc_index]
    df = pd.DataFrame({"Factor": feature_names, "Loading": loading}).sort_values("Loading", key = abs, ascending = False)

    fig = px.bar(df, x = "Factor", y = "Loading", title = f"Loadings for PC{pc_index + 1}")
    fig.update_layout(template = "plotly_white")
    return fig

def plot_pca_heatmap(pca, feature_names: list):
    heatmap_data = pca.components_
    fig = go.Figure(data = go.Heatmap(z = heatmap_data, x = feature_names, y = [f"PC{i+1}" for i in range(len(heatmap_data))], colorscale = "RdBu", zmid = 0))

    fig.update_layout(title = "PCA Loadings Heatmap", xaxis_title = "Original Features", yaxis_title = "Principal Components", template = "plotly_white")
    return fig

def build_elbow_figure(cum_var: np.ndarray, k: int | None = None, target_var: float | None = None) -> go.Figure:
    x = np.arange(1, len(cum_var) + 1)
    unexplained = 1 - cum_var

    fig = go.Figure(
        go.Scatter(x = x, y = unexplained, mode = "lines+markers", name = "Unexplained variance")
    )

    if target_var is not None:
        fig.add_shape(type = "line", x0 = 1, x1 = len(cum_var), y0 = 1 - target_var, y1 = 1 - target_var, line = dict(color = "red", width = 1.5, dash = "dash"))

    if k is not None:
        fig.add_shape(type = "line", x0 = k, x1 = k, y0 = 0, y1 = max(unexplained), line = dict(color = "green", width = 1.5, dash = "dot"))

    fig.update_layout(title = "PCA elbow curve", xaxis_title = "Principal components (k)", yaxis_title = "Unexplained variance", template = "plotly_white", height = 350)
    return fig