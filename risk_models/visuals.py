import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Tuple

def plot_covariance_heatmap(sigma: pd.DataFrame, title: str = "Covariance matrix"):
    fig, ax = plt.subplots(figsize = (6, 5))
    im = ax.imshow(sigma, cmap = "viridis")
    fig.colorbar(im, ax = ax, fraction = 0.046)
    ax.set_title(title)
    ax.set_xticks(range(len(sigma.columns)), sigma.columns, rotation = 90, fontsize = 6)
    ax.set_yticks(range(len(sigma.index)), sigma.index, fontsize = 6)
    fig.tight_layout()

    fig_px = px.imshow(sigma, x = sigma.columns, y = sigma.index, color_continuous_scale = "Viridis", title = title)
    return fig, fig_px

def plot_eigen_spectrum(sigma: pd.DataFrame, title: str = "Eigenvalue spectrum") -> Tuple[plt.Figure, plt.Axes]:
    values = np.linalg.eigvalsh(sigma.values)
    values_sorted = np.sort(values)[::-1]

    fig, ax = plt.subplots(figsize = (5, 3))
    ax.bar(range(1, len(values_sorted) + 1), values_sorted)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax