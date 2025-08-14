from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _format_column(col: pd.Series) -> List[str]:
    if pd.api.types.is_numeric_dtype(col):
        return [f"{x:,.2f}" if pd.notna(x) else "" for x in col]
    return col.astype(str).tolist()

def _build_sample_table(X: pd.DataFrame, n_rows: int = 0) -> go.Figure:
    if n_rows == 0:
        n_rows = len(X)
    row_colors = ["#f8fafc" if i % 2 == 0 else "#e2e8f0" for i in range(n_rows)]
    tbl_fig = go.Figure(
        go.Table(
            columnwidth=[120] + [95] * (len(X.columns) - 1),
            header = dict(values = [f"<b>{c}</b>" for c in X.columns], fill_color = "#1f2937", font = dict(color = "white", size = 13), align = "left", height = 32),
            cells = dict(values = [_format_column(X[c].head(n_rows)) for c in X.columns], fill_color = [row_colors] * len(X.columns), align = "left", font = dict(size = 12), height = 26),
        )
    )
    tbl_fig.update_layout(title = f"Factor sample - {n_rows} rows", margin = dict(l = 0, r = 0, t = 38, b = 0), height = 420)
    return tbl_fig

def _build_overview_figures(X: pd.DataFrame, y: pd.Series, horizon: int) -> Tuple[go.Figure, go.Figure, go.Figure]:
    num_corr = X.select_dtypes("number").corr().round(2)
    heatmap = px.imshow(num_corr, text_auto = True, aspect = "auto", title = "Factor correlation matrix")
    hist = px.histogram(y, nbins = 30, title = f"Distribution of {horizon}-day forward returns")
    table = _build_sample_table(X)
    return heatmap, hist, table