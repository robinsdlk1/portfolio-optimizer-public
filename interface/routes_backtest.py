import pandas as pd

from typing import Dict, Any
from flask import Blueprint, render_template, request, flash

from portfolio_optimizer.interface.state_manager import get_X, get_R, get_portfolio_weights, get_optimizer_specs
from portfolio_optimizer.interface.helpers import _make_plotly_html

from portfolio_optimizer.backtest.visuals import plot_cumulative_returns, plot_drawdowns, plot_return_histogram, plot_portfolio_composition_over_time
from portfolio_optimizer.backtest.metrics import compute_realized_metrics
from portfolio_optimizer.backtest.base import BacktestEngine

from portfolio_optimizer.strategy.wrappers import make_optimizer_strategy

from portfolio_optimizer.optimizers.base import OptimizerRole
from portfolio_optimizer.optimizers.naive import VolatilityParityOptimizer

bp_backtest = Blueprint("backtest", __name__)

@bp_backtest.route("/backtest_step", methods=["GET", "POST"])
def backtest_step():
    R = get_R()
    weights_dict = get_portfolio_weights()
    specs = get_optimizer_specs() or {}

    if R is None or weights_dict is None:
        flash("Run the optimisation step first.", "error")
        return render_template("backtest.html", pnl_ready=False)

    if request.method == "POST":
        chosen = request.form.get("which_portfolio")
        if chosen not in weights_dict:
            flash("Unknown portfolio selection.", "error")
            return render_template("backtest.html", pnl_ready = False, portfolios = list(weights_dict.keys()))

        w0: pd.Series = weights_dict[chosen]
        opt_cls, opt_kwargs = specs.get(chosen, (VolatilityParityOptimizer, {}))
        role = getattr(opt_cls, "ROLE", OptimizerRole.NAIVE)

        freq_ui = request.form.get("rebalance_freq", "W-FRI")
        win_ui = int(request.form.get("window_len", "60"))
        cost_ui = float(request.form.get("cost_bps", "0.0"))

        schedule = pd.date_range(R.index[0], R.index[-1], freq = freq_ui).intersection(R.index) if freq_ui else pd.DatetimeIndex([])

        X_full = get_X()
        strat_kwargs: Dict[str, Any] = dict(optimizer_cls = opt_cls, window = win_ui, X_source = X_full, **opt_kwargs)

        rolling_strategy = make_optimizer_strategy(**strat_kwargs)

        engine = BacktestEngine(R, w0, rolling_strategy, schedule, cost_bps = cost_ui)
        res = engine.run()

        port_ret = res["returns"]
        port_weights = res["weights"]
        cum_df = (res["value"] / res["value"].iloc[0]).to_frame("portfolio_value")

        graphs: Dict[str, Any] = {
            "performance": [
                (
                    "Cumulative",
                    plot_cumulative_returns(cum_df, title = f"Cumulative - {chosen}"),
                ),
                (
                    "Drawdowns",
                    plot_drawdowns(port_ret, title = f"Drawdowns - {chosen}"),
                ),
            ],
            "allocation": [
                (
                    "Weights over time",
                    plot_portfolio_composition_over_time(port_weights),
                ),
            ],
            "distribution": [
                ("Return histogram", plot_return_histogram(port_ret)),
            ],
            "risk": [],
        }

        graphs_html = {
            cat: [(lbl, _make_plotly_html(fig)) for lbl, fig in figs]
            for cat, figs in graphs.items()
        }

        metrics = compute_realized_metrics(port_ret)

        return render_template("backtest.html", pnl_ready = True, graphs = graphs_html, metrics = metrics, portfolios = list(weights_dict.keys()), selected = chosen)

    return render_template("backtest.html", pnl_ready = False, portfolios = list(weights_dict.keys()))