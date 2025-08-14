from flask import Blueprint, flash, redirect, render_template, request, url_for

from portfolio_optimizer.interface.helpers import _make_plotly_html

from portfolio_optimizer.interface.state_manager import get_X, get_y, set_factor_model

from portfolio_optimizer.factor_models.ols import OLSFactorModel
from portfolio_optimizer.factor_models.pca import FactorPCA
from portfolio_optimizer.factor_models.visuals import build_elbow_figure, build_beta_bar, build_beta_table, build_reg_summary, build_residual_plot


bp_factors = Blueprint("factors", __name__)

@bp_factors.route("/factors", methods=["GET", "POST"])
def factors_step():
    X, y = get_X(), get_y()
    print(X)
    if X is None or y is None:
        flash("Please load data in data before running the factor model.", "error")
        return redirect(url_for("data.data_step"))

    form = request.form
    model_type = form.get("model_type", "ols")
    action = form.get("action")

    pca_mode = form.get("pca_mode", "var")
    target_var = float(form.get("target_var", 95)) / 100.0
    pca_n = form.get("pca_n", type = int)

    elbow_html = None
    summary_html = None
    resid_html = None
    beta_chart = None
    factor_table = None

    if model_type == "pca":
        _pre = OLSFactorModel()._build_preprocessor(X).fit(X)
        p_encoded = len(_pre.get_feature_names_out())
        preview_k = min(X.shape[0], p_encoded, 30)
        cum = FactorPCA(n_components = preview_k).fit(X).explained_variance_ratio_.cumsum()
        elbow_html = _make_plotly_html(build_elbow_figure(cum, k = (pca_n if pca_mode == "manual" and pca_n else None), target_var = (target_var if pca_mode == "var" else None)))

    if action == "fit":

        if model_type == "ols":
            try:
                mdl = OLSFactorModel().fit(X, y)
            except ValueError as e:
                flash(str(e), "error")
                return render_template("factors.html", model_type = model_type, pca_mode = pca_mode, target_var = int(target_var * 100), pca_n = pca_n or "", std_flag = False, elbow_html = None, summary_html = None, resid_html = None, beta_chart = None, factor_table = None)
            betas = mdl.betas
            intercept = mdl.intercept_
            pre = mdl.preprocessor

            X_design = mdl.preprocessor.transform(X)
            summary_html = build_reg_summary(X_design, y)

            set_factor_model({"betas": mdl.betas,"intercept": mdl.intercept_,"preprocessor": mdl.preprocessor})

        else:
            if pca_mode == "var":
                pca = FactorPCA(target_variance = target_var)
            
            elif pca_mode == "manual":
                pca = FactorPCA(n_components = max(1, pca_n or 1))
            
            elif pca_mode == "cv":
                _selector = FactorPCA(n_components = 1, random_state = 40)
                n_splits = min(5, max(2, len(X) - 1))
                
                best_k, _cv_scores = _selector.select_k_via_cv(X, y, n_splits = n_splits, random_state = 40)
                pca = FactorPCA(n_components = best_k)
                
                best_mean = float(_cv_scores.loc[_cv_scores["k"] == best_k, "mean_r2"].iloc[0])
                flash(f"CV selected k = {best_k} (mean out-of-fold R^2 = {best_mean:.3f}).", "success")
            else:
                raise ValueError(f"Unknown pca_mode: {pca_mode}")
            
            pca = pca.fit(X)

            betas = pca.get_betas(X, y)
            intercept = pca.intercept_
            pre = pca.preprocessor

            _, _, Z_df, _ = pca.get_pc_regression(X, y)
            summary_html = build_reg_summary(Z_df, y)

            cum = pca.explained_variance_ratio_.cumsum()
            elbow_html = _make_plotly_html(build_elbow_figure(cum, k = len(pca.components_), target_var = target_var if pca_mode == "var" else None))
            
            set_factor_model({"betas": betas, "intercept": intercept, "preprocessor": pre})

        beta_fig, beta_sorted = build_beta_bar(betas)
        beta_chart = _make_plotly_html(beta_fig)
        factor_table = _make_plotly_html(build_beta_table(beta_sorted))

        resid_fig = build_residual_plot(pre, betas, intercept, X, y)
        resid_html = _make_plotly_html(resid_fig)

    return render_template(
        "factors.html",
        model_type = model_type,
        pca_mode = pca_mode,
        target_var = int(target_var * 100),
        pca_n = pca_n or "",
        std_flag = False,
        elbow_html = elbow_html,
        summary_html = summary_html,
        resid_html = resid_html,
        beta_chart = beta_chart,
        factor_table = factor_table
    )