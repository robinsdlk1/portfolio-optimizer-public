import pandas as pd
import json

from typing import Dict, Any
from flask import Blueprint, render_template, request, flash

from portfolio_optimizer.interface.state_manager import get_X, get_y, get_R, get_factor_model, get_horizon, set_portfolio_weights, set_optimizer_specs
from portfolio_optimizer.interface.helpers import _make_plotly_html, parse_constraints, build_acd_metadata, apply_acd_rules

from portfolio_optimizer.optimizers.naive import EqualWeightOptimizer, VolatilityParityOptimizer
from portfolio_optimizer.optimizers.classic import MinVolOptimizer, MaxSharpeOptimizer, TargetReturnOptimizer
from portfolio_optimizer.optimizers.factor import FactorSharpeOptimizer, FactorTargetReturnOptimizer
from portfolio_optimizer.optimizers.visuals import build_optimizer_graphs

from portfolio_optimizer.risk_models.sample import SampleCovariance

bp_optimizers = Blueprint("optimizers", __name__)

@bp_optimizers.route("/optimizers_step", methods = ["GET", "POST"])
def optimizers_step():
    X, y, R, factor_model = get_X(), get_y(), get_R(), get_factor_model()

    group_candidates = "[]"
    numeric_candidates = "[]"
    group_values_map  = "{}"
    tickers_simple_json    = "[]"

    if X is None or y is None or R is None:
        flash("Please complete the factor model step before optimising.", "error")
        return render_template("optimizers.html", weights_ready = False, graphs_ready = False, graphs = {}, group_candidates = group_candidates, numeric_candidates = numeric_candidates, group_values_map = group_values_map, tickers_simple_json = tickers_simple_json)
    
    if not isinstance(factor_model, dict) or any(k not in factor_model for k in ("betas", "intercept", "preprocessor")):
        flash("Invalid factor model. Please refit the factor step.", "error")
        return render_template("optimizers.html", weights_ready = False, graphs_ready = False, graphs = {}, group_candidates = group_candidates, numeric_candidates = numeric_candidates, group_values_map = group_values_map, tickers_simple_json = tickers_simple_json)

    betas = factor_model["betas"]
    intercept = factor_model["intercept"]
    preprocess = factor_model["preprocessor"]

    if "ticker" in "".join(X.columns).lower() and X.index.name is None:
        tic_col = next(c for c in X.columns if "ticker" in c.lower())
        X = X.set_index(tic_col, drop=True)

    if X is not None:
        meta = build_acd_metadata(X)
        group_candidates = json.loads(meta["group_candidates_json"])
        numeric_candidates = json.loads(meta["numeric_candidates_json"])
        tickers_simple_json = meta["tickers_simple_json"]
        group_values_map = json.loads(meta["group_values_map_json"])

    if request.method == "POST":
        selected_factor_opts = request.form.getlist("factor_optimizers")
        selected_classic_opts = request.form.getlist("classic_optimizers")
        selected_naive_opts = request.form.getlist("naive_optimizers")

        if not (selected_factor_opts or selected_classic_opts or selected_naive_opts):
            flash("Please select at least one optimizer.", "error")
            return render_template("optimizers.html", weights_ready = False, graphs_ready = False, graphs = {}, group_candidates = group_candidates, numeric_candidates = numeric_candidates, group_values_map = group_values_map, tickers_simple_json = tickers_simple_json)
        
        try:
            risk_lambda = float(request.form.get("risk_aversion", "1.0"))
            if not (0 < risk_lambda < 1e6):
                raise ValueError
        except ValueError:
            flash("Invalid risk-aversion value; using 1.0", "warning")
            risk_lambda = 1.0

        try:
            target_ret = float(request.form.get("target_return", "0.05"))
            if not (-0.5 < target_ret < 1.0):
                raise ValueError
        except ValueError:
            flash("Invalid target return; using 0.05", "warning")
            target_ret = 0.05

        constraints = parse_constraints(request.form)

        rules_raw = request.form.get("constraint_json", "").strip()
        constraints, groups = apply_acd_rules(rules_raw, X, constraints)

        horizon = get_horizon() or 1
        cov = SampleCovariance().fit(R, horizon = horizon).get_covariance()
        mu_hat = y

        weights_dict: Dict[str, pd.Series] = {}
        specs_dict: Dict[str, tuple[type, Dict[str, Any]]] = {}

        common_f_kwargs = dict(betas = betas, intercept = intercept, preprocess = preprocess, constraints = constraints)

        for opt in selected_factor_opts:
            try:
                if opt == "sharpe":
                    optimizer = FactorSharpeOptimizer(**common_f_kwargs, risk_aversion = risk_lambda)
                    full_kwargs = {**common_f_kwargs, "risk_aversion": risk_lambda}
                elif opt == "target_return":
                    optimizer = FactorTargetReturnOptimizer(**common_f_kwargs, target_return = target_ret, risk_aversion = risk_lambda)
                    full_kwargs = {**common_f_kwargs, "target_return": target_ret, "risk_aversion": risk_lambda}
                else:
                    raise ValueError(f"Unknown factor optimizer: {opt}")

                weights = optimizer.fit(X = X, cov = cov, groups = groups)
                label = f"Factor - {opt}"
                weights_dict[label] = weights
                specs_dict[label] = (optimizer.__class__, full_kwargs)

            except Exception as e:
                flash(f"Error with factor optimizer '{opt}': {e}", "error")

        for opt in selected_classic_opts:
            try:
                if opt == "min_vol":
                    optimizer = MinVolOptimizer(constraints = constraints)
                    fit_kwargs = dict(cov = cov, groups = groups)
                elif opt == "max_sharpe":
                    optimizer = MaxSharpeOptimizer(constraints = constraints, risk_aversion = risk_lambda)
                    fit_kwargs = dict(mu = mu_hat, cov = cov, groups = groups)
                elif opt == "target_return":
                    optimizer = TargetReturnOptimizer(constraints = constraints, target_return = target_ret)
                    fit_kwargs = dict(mu = mu_hat, cov = cov, groups = groups)
                else:
                    raise ValueError(f"Unknown classic optimizer: {opt}")

                weights = optimizer.fit(**fit_kwargs)
                label = f"Classic - {opt}"
                weights_dict[label] = weights
                specs_dict[label] = (optimizer.__class__, {"constraints": constraints})

            except Exception as e:
                flash(f"Error with classic optimizer '{opt}': {e}", "error")

        for opt in selected_naive_opts:
            try:
                if opt == "equal":
                    optimizer = EqualWeightOptimizer()
                elif opt == "inv_vol":
                    optimizer = VolatilityParityOptimizer()
                else:
                    raise ValueError(f"Unknown naive optimizer: {opt}")

                weights = optimizer.fit(mu = mu_hat, cov = cov)
                label = f"Naive - {opt}"
                weights_dict[label] = weights
                specs_dict[label] = (optimizer.__class__, {})

            except Exception as e:
                flash(f"Error with naive optimizer '{opt}': {e}", "error")

        if not weights_dict:
            flash("No portfolios could be constructed. Check logs or try different optimisers.", "error")
            return render_template("optimizers.html", weights_ready = False, graphs_ready = False, graphs = {}, group_candidates = group_candidates, numeric_candidates = numeric_candidates, group_values_map = group_values_map, tickers_simple_json = tickers_simple_json)

        graphs = build_optimizer_graphs(weights_dict, cov = cov, factors = X)
        graphs_html = {gtype: [(lbl, _make_plotly_html(fig)) for lbl, fig in figs] for gtype, figs in graphs.items()}

        set_portfolio_weights(weights_dict)
        set_optimizer_specs(specs_dict)

        flash(f"{len(weights_dict)} portfolios computed successfully.", "success")
        return render_template("optimizers.html", weights_ready = True, graphs_ready = True, graphs = graphs_html, group_candidates = group_candidates, numeric_candidates = numeric_candidates, group_values_map = group_values_map, tickers_simple_json = tickers_simple_json)

    return render_template("optimizers.html", weights_ready = False, graphs_ready = False, graphs = {}, group_candidates = group_candidates, numeric_candidates = numeric_candidates, group_values_map = group_values_map, tickers_simple_json = tickers_simple_json)