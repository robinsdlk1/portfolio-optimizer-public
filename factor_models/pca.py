import numpy as np
import pandas as pd

from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from portfolio_optimizer.factor_models.ols import OLSFactorModel

class FactorPCA:
    def __init__(self, n_components: Optional[int] = None, target_variance: Optional[float] = None, *, random_state: int = 40) -> None:
        if (n_components is None) and (target_variance is None):
            raise ValueError("Specify either n_components or target_variance.")
        if (n_components is not None) and (target_variance is not None):
            raise ValueError("Choose either n_components or target_variance, not both.")

        self.n_components = n_components
        self.target_variance = target_variance
        self.random_state = random_state

        self._ols_helper = OLSFactorModel(random_state = random_state)

        self.preprocessor: ColumnTransformer | None = None
        self._pca: PCA | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

        self.betas: pd.Series | None = None
        self.intercept_: float | None = None
        self.is_fitted: bool = False

    def _build_preprocessor(self, X: pd.DataFrame, categories = None) -> ColumnTransformer:
        return self._ols_helper._build_preprocessor(X, categories = categories)

    def fit(self, X: pd.DataFrame) -> "FactorPCA":
        if X.empty: raise ValueError("Input dataframe X is empty.")

        self.preprocessor = self._build_preprocessor(X)
        X_proc = self.preprocessor.fit_transform(X)

        k = self.n_components if self.n_components is not None else self.target_variance
        if isinstance(k, (int, np.integer)):
            n_samples, n_features = X_proc.shape
            k_max = int(min(n_samples, n_features))
            if k < 1:
                raise ValueError("n_components must be >= 1.")
            if k > k_max:
                k = k_max
        else:
            if not (0 < float(k) <= 1.0):
                raise ValueError("target_variance must be in (0, 1].")
        self._pca = PCA(n_components = k, random_state = self.random_state).fit(X_proc)

        self.components_ = self._pca.components_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        Z = self._pca.transform(self.preprocessor.transform(X))
        cols = [f"PC{i+1}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, index = X.index, columns = cols)

    def inverse_transform_betas(self, beta_pc: np.ndarray) -> pd.Series:
        if self.components_ is None:
            raise RuntimeError("PCA not fitted.")
        beta_raw = beta_pc @ self.components_
        return pd.Series(beta_raw, index = self.feature_names_)

    def get_betas(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        Z = self.transform(X).values
        Z_aug = np.column_stack([np.ones(len(Z)), Z])
        coef = np.linalg.lstsq(Z_aug, y.values, rcond = None)[0]

        self.intercept_ = float(coef[0])
        beta_pc = coef[1:]
        self.betas = self.inverse_transform_betas(beta_pc)
        return self.betas

    def get_pc_regression(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, float, pd.DataFrame, object]:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        Z_df = self.transform(X)

        Z_aug = np.column_stack([np.ones(len(Z_df)), Z_df.values])
        coef = np.linalg.lstsq(Z_aug, y.values, rcond=None)[0]

        intercept_pc = float(coef[0])
        beta_pc = pd.Series(coef[1:], index = Z_df.columns)

        class _IdPre:
            def __init__(self, cols): self._cols = cols
            def transform(self, df):  return df.values
            def get_feature_names_out(self): return self._cols

        id_pre = _IdPre(list(Z_df.columns))

        return beta_pc, intercept_pc, Z_df, id_pre

    @property
    def n_components_(self) -> int:
        if self._pca is None:
            raise RuntimeError("PCA not fitted.")
        return self._pca.n_components_
    
    def select_k_via_cv(self, X: pd.DataFrame, y: pd.Series, k_grid: Optional[list[int]] = None, n_splits: int = 5, random_state: Optional[int] = None) -> tuple[int, pd.DataFrame]:
        n = len(X)
        if n != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if n_splits < 2 or n_splits > n:
            raise ValueError("n_splits must be in [2, n].")

        pre_full = self._build_preprocessor(X)
        pre_full.fit(X)
        p_encoded = len(pre_full.get_feature_names_out())

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        min_train = min(len(tr) for tr, _ in kf.split(X))
        k_max = int(min(min_train, p_encoded, 50))
        
        print(f"[CV] k_max = {k_max}, p_encoded = {p_encoded}, min_train = {min_train}, n_splits = {n_splits}")

        if k_max < 1:
            raise ValueError("Not enough data after preprocessing to run CV-PCR.")

        if k_grid is None:
            k_grid = list(range(1, k_max + 1))
        else:
            k_grid = sorted(set(int(k) for k in k_grid if 1 <= int(k) <= k_max))
            if not k_grid:
                raise ValueError(f"k_grid has no feasible values within [1, {k_max}].")
        
        cat = None
        try:
            cat = pre_full.named_transformers_["cat"].named_steps["encoder"].categories_
        except Exception:
            pass

        results = []
        splits = list(KFold(n_splits = n_splits, shuffle = True, random_state = random_state).split(X))
        for k in k_grid:
            fold_scores = []
            for train_idx, test_idx in splits:
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                pre = self._build_preprocessor(X_tr, categories = cat)
                X_tr_proc = pre.fit_transform(X_tr)
                X_te_proc = pre.transform(X_te)

                k_fold = min(k, X_tr_proc.shape[0], X_tr_proc.shape[1])
                if k_fold < 1:
                    continue

                pca = PCA(n_components = k_fold, random_state = self.random_state).fit(X_tr_proc)
                Z_tr = pca.transform(X_tr_proc)
                Z_te = pca.transform(X_te_proc)

                Z_tr_aug = np.column_stack([np.ones(len(Z_tr)), Z_tr])
                coef = np.linalg.lstsq(Z_tr_aug, y_tr.values, rcond = None)[0]
                y_hat = np.dot(np.column_stack([np.ones(len(Z_te)), Z_te]), coef)

                denom = np.sum((y_te.values - y_te.values.mean())**2)
                if denom <= 0:
                    fold_r2 = np.nan
                else:
                    fold_r2 = 1.0 - np.sum((y_te.values - y_hat)**2) / denom
                fold_scores.append(fold_r2)

            mean_r2 = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
            std_r2  = float(np.nanstd(fold_scores)) if fold_scores else float("nan")
            results.append((k, mean_r2, std_r2))

        scores = pd.DataFrame(results, columns = ["k", "mean_r2", "std_r2"]).sort_values("k")
        if scores["mean_r2"].isna().all():
            raise ValueError("All CV folds failed; cannot select k.")
        best_mean = scores["mean_r2"].max()
        best_k = int(scores.loc[scores["mean_r2"] == best_mean, "k"].min())
        return best_k, scores