import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.io as pio

from typing import Mapping, Optional, Any, Tuple
from portfolio_optimizer.optimizers.base import Constraints

_CAT_LIMIT = 50 # max distinct values for a column to count as "categorical"

def _make_plotly_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs = False, full_html = False)

def _to_float(val: str | None, *, default: Optional[float] = None) -> Optional[float]:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def parse_constraints(form: Mapping[str, str]) -> Constraints:
    long_only = form.get("long_only", "on") == "on"
    min_w = _to_float(form.get("min_weight"), default = 0.0)
    max_w = _to_float(form.get("max_weight"), default = 1.0)

    if min_w is not None and max_w is not None and min_w > max_w:
        print("Min weight exceeds max weight; reverting to defaults (0 / 1).", "warning")
        min_w, max_w = 0.0, 1.0

    if not long_only and (min_w is None or min_w >= 0):
        min_w = -max_w 
    
    weight_sum = _to_float(form.get("weight_sum"), default = 1.0) or 1.0

    gross_cap = _to_float(form.get("gross_cap"))
    turnover_cap = _to_float(form.get("turnover_cap"))

    max_assets_raw = form.get("max_assets")
    max_assets: int | None = None
    if max_assets_raw:
        try:
            max_val = int(float(max_assets_raw))
            if max_val > 0:
                max_assets = max_val
            else:
                print("Max assets must be a positive integer.", "warning")
        except ValueError:
            print("Invalid max-assets value; ignoring.", "warning")

    if gross_cap is not None and gross_cap < 1 and (min_w or 0) < 0:
        print("Gross-exposure cap < 1 forbids any short positions.", "error")
    
    return Constraints(long_only = long_only, min_weight = min_w, max_weight = max_w, weight_sum = weight_sum, gross_cap = gross_cap, turnover_cap = turnover_cap, max_assets = max_assets, group_mode = "net")

def build_acd_metadata(X: pd.DataFrame) -> dict[str, str]:

    group_cols: list[str] = [c for c in X.columns if (X[c].dtype == "object" or X[c].dtype.name == "category") and X[c].nunique(dropna=True) <= max(_CAT_LIMIT, len(X) // 10)]

    if "Sector" in X.columns and "Sector" not in group_cols:
        group_cols.append("Sector")

    numeric_cols: list[dict[str, Any]] = [
        {
            "name": c,
            "min": float(X[c].min(skipna=True)),
            "max": float(X[c].max(skipna=True)),
        }
        for c in X.select_dtypes(include=["number"]).columns
    ]

    group_values_map = {
        col: X[col].value_counts(dropna=True).to_dict()
        for col in group_cols
    }

    tickers_list = list(map(str, X.index))
    return {
        "group_candidates_json": json.dumps(group_cols),
        "numeric_candidates_json": json.dumps(numeric_cols),
        "group_values_map_json": json.dumps(group_values_map),
        "tickers_simple_json": json.dumps(tickers_list),
    }

def apply_acd_rules(rules_json: str, X: pd.DataFrame, constraints: Constraints, *, default_group_mode: str = "long_only") -> Tuple[Constraints, pd.Series | None]:
    if not rules_json:
        return constraints, None

    try:
        rules = json.loads(rules_json)
    except json.JSONDecodeError:
        return constraints, None

    group_by = rules.get("group_by")
    groups: pd.Series | None = None
    if group_by in X.columns:
        groups = X[group_by].astype(str)

    constraints.group_max.update(rules.get("group_caps", {}))
    constraints.group_min.update(rules.get("group_min", {}))

    for rc in rules.get("range_caps", []):
        col = rc.get("column")
        cap = rc.get("max")
        if col not in X.columns or cap is None:
            continue

        low = rc.get("low", -np.inf)
        high = rc.get("high", np.inf)
        mode = rc.get("mode", default_group_mode)

        mask = ((X[col] >= low) & (X[col] <= high)).astype(float).values
        if not mask.any():
            continue
        constraints.mask_caps.append((mask, float(cap), mode))

    return constraints, groups