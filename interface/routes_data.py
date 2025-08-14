import pandas as pd
import tempfile

from typing import List
from flask import Blueprint, flash, redirect, render_template, request, url_for

from portfolio_optimizer.interface.helpers import _make_plotly_html
from portfolio_optimizer.interface.state_manager import get_X, get_y, get_R, get_horizon, set_data_state

from portfolio_optimizer.data.get_returns import convert_to_yahoo_ticker
from portfolio_optimizer.data.returns import ReturnsDataset
from portfolio_optimizer.data.cleaner import clean_X
from portfolio_optimizer.data.visuals import _build_overview_figures

bp_data = Blueprint("data", __name__)

@bp_data.route("/data_step", methods = ["GET", "POST"])
def data_step():
    if request.method == "POST":
        print("[data_step] Enter POST")

        csv_file = request.files.get("csv_file")
        json_file = request.files.get("returns_json")
        print(f"[data_step] csv_file = {bool(csv_file)} name = {(csv_file.filename if csv_file else None)}")
        print(f"[data_step] json_file = {bool(json_file)} name = {(json_file.filename if json_file else None)}")

        if not csv_file or csv_file.filename == "":
            print("[data_step] No CSV provided -> redirect")
            flash("Please choose a CSV file first.", "error")
            return redirect(url_for("data.data_step"))

        factor_date_raw = request.form.get("factor_date")
        returns_end_raw = request.form.get("returns_end")
        horizon = int(request.form.get("horizon", 63))
        print(f"[data_step] factor_date_raw = {factor_date_raw} returns_end_raw = {returns_end_raw} horizon = {horizon}")

        if not factor_date_raw or not returns_end_raw:
            print("[data_step] Missing dates -> redirect")
            flash("Please provide both Factor date and Returns until.", "error")
            return redirect(url_for("data.data_step"))

        X_raw = pd.read_csv(csv_file.stream)
        print(f"[data_step] X_raw shape = {X_raw.shape} cols = {list(X_raw.columns)}")

        ticker_col = next((c for c in X_raw.columns if "ticker" in c.lower()), None)
        print(f"[data_step] detected ticker_col = {ticker_col}")
        if ticker_col is None:
            flash("Could not find a Ticker column.", "error")
            return redirect(url_for("data.data_step"))

        raw_tickers: List[str] = X_raw[ticker_col].dropna().astype(str).unique().tolist()
        print(f"[data_step] raw_tickers n = {len(raw_tickers)} sample = {raw_tickers[:10]}")

        yf_tickers: List[str] = [convert_to_yahoo_ticker(t) for t in raw_tickers]
        print(f"[data_step] mapped yf_tickers n = {len(yf_tickers)} sample = {yf_tickers[:10]}")

        if json_file and json_file.filename:
            tmp_path = tempfile.NamedTemporaryFile(delete = False, suffix = ".json").name
            json_file.save(tmp_path)
            print(f"[data_step] Using JSON returns at {tmp_path}")
            returns_ds = ReturnsDataset(tickers = None, start_date = None, end_date = None, source = "json", json_path = tmp_path, fwd_horizons = [horizon], save = False)
        else:
            print(f"[data_step] Using yahooquery returns start={factor_date_raw} end={returns_end_raw}")
            returns_ds = ReturnsDataset(tickers = yf_tickers, start_date = factor_date_raw, end_date = returns_end_raw, source = "yahooquery", fwd_horizons = [horizon])

        fwd = returns_ds.compute_forward_return(horizon)
        fwd.index = pd.to_datetime(fwd.index).normalize()
        print(f"[data_step] fwd shape = {fwd.shape}")
        if len(fwd.index):
            print(f"[data_step] fwd index range: {fwd.index.min()} -> {fwd.index.max()} (len = {len(fwd.index)})")
        else:
            print("[data_step] fwd index is EMPTY")

        end_dt_req = pd.to_datetime(returns_end_raw).normalize()
        idx = fwd.index

        if end_dt_req in idx:
            pos = idx.get_loc(end_dt_req)
        else:
            pos = idx.searchsorted(end_dt_req, side = "right") - 1

        if pos < 0:
            flash("Returns-until date precedes the first trading day in the dataset.", "error")
            return redirect(url_for("data.data_step"))

        req_cols = [f"{t}_Fwd{horizon}d" for t in yf_tickers]
        have_cols = [c for c in req_cols if c in fwd.columns]
        if not have_cols:
            flash("No forward-return columns available at the requested horizon.", "error")
            return redirect(url_for("data.data_step"))

        scan_limit = min(horizon * 2, len(idx))
        resolved_pos = None
        for k in range(pos, max(-1, pos - scan_limit), -1):
            row = fwd.iloc[k][have_cols]
            if row.notna().any():
                resolved_pos = k
                break

        if resolved_pos is None:
            flash("Could not find any non-empty forward returns at or before the requested returns_end.", "error")
            return redirect(url_for("data.data_step"))

        resolved_end = idx[resolved_pos]
        print(f"[data_step] resolved_end (non-NaN row) = {resolved_end}")

        y = fwd.iloc[resolved_pos][have_cols]
        y.index = [c.removesuffix(f"_Fwd{horizon}d") for c in have_cols]
        y.name = "FwdReturn"
        print(f"[data_step] y built at {resolved_end}: len = {len(y)} NaNs = {int(y.isna().sum())}")
        print(f"[data_step] y head = \n{y.head(10)}")

        nan_tickers = y.index[y.isna()]
        if len(nan_tickers) > 0:
            preview = ", ".join(nan_tickers[:10]) + (" …" if len(nan_tickers) > 10 else "")
            print(f"[data_step] y NaNs for {len(nan_tickers)} tickers: {preview}")
            flash(f"Forward return missing for {len(nan_tickers)} tickers: {preview}", "warning")

        X_raw[ticker_col] = X_raw[ticker_col].apply(convert_to_yahoo_ticker)
        X = X_raw.set_index(ticker_col, drop=True).reindex(y.index)
        print(f"[data_step] X shape = {X.shape} X NaNs = {int(X.isna().sum().sum())}")
        print(f"[data_step] X.index == y.index? {X.index.equals(y.index)}")

        R = returns_ds.get_return_matrix()
        print(f"[data_step] R shape = {getattr(R, 'shape', None)}")

        assert X.index.equals(y.index), "X and y are not aligned!"

        set_data_state(X, y, R, horizon)
        print("[data_step] set_data_state done")

        heatmap_fig, hist_fig, table_fig = _build_overview_figures(X, y, horizon)
        heatmap_html = _make_plotly_html(heatmap_fig)
        hist_html = _make_plotly_html(hist_fig)
        table_html = _make_plotly_html(table_fig)

        has_nans = X.isna().any().any()
        flash(f"Loaded {len(X)} tickers and {X.shape[1]} factors.", "success")
        print("[data_step] render_template(data.html)")

        return render_template("data.html", heatmap = heatmap_html, hist = hist_html, table = table_html, has_nans = has_nans, horizon = horizon)

    print("[data_step] Enter GET -> render empty data.html")
    return render_template("data.html")

@bp_data.route("/clean", methods = ["POST"])
def clean_step():
    method = request.form.get("clean_method", "drop")
    drop_thresh = float(request.form.get("drop_thresh", 0.10))
    horizon = int(request.form.get("horizon_current", 63))

    X, y, R = get_X(), get_y(), get_R()
    if X is None or y is None:
        flash("Nothing to clean - load a dataset first.", "error")
        return redirect(url_for("data.data_step"))

    X_clean = clean_X(X, method = method, drop_thresh = drop_thresh)
    horizon = get_horizon()
    set_data_state(X_clean, y, R, horizon)

    heatmap_fig, hist_fig, table_fig = _build_overview_figures(X_clean, y, horizon)
    heatmap_html = _make_plotly_html(heatmap_fig)
    hist_html = _make_plotly_html(hist_fig)
    table_html = _make_plotly_html(table_fig)

    remaining = int(X_clean.isna().sum().sum())
    if remaining:
        flash(f"{remaining:,} missing values remain after cleaning.", "warning")
    else:
        flash( f"All NaNs removed, {len(X_clean)} tickers and {X_clean.shape[1]} factors remaining.", "success")

    return render_template("data.html", heatmap = heatmap_html, hist = hist_html, table = table_html, has_nans = remaining > 0, horizon = horizon)