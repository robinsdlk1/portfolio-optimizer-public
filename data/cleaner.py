import pandas as pd
import numpy as np
from typing import Literal, Any, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

CleanMethod = Literal["drop", "median", "mean", "knn", "iterative"]

class FeatureCleaner:

    def __init__(self, method: CleanMethod = "drop", drop_thresh: float = 0.4, **kwargs: Any):
        self.method: CleanMethod = method
        self.drop_thresh: float = drop_thresh
        self.kwargs: dict[str, Any] = kwargs
        self._imputer: Optional[Any] = None
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "FeatureCleaner":
        if self.method == "drop":
            self.keep_cols_ = X.columns[~X.isna().any(axis = 0)]
            self._is_fitted = True
            return self

        self.keep_cols_ = X.columns[X.isna().mean() <= self.drop_thresh]
        X_fit = X[self.keep_cols_]

        numeric = X_fit.select_dtypes(include = np.number)

        if self.method in {"median", "mean"}:
            strat = "median" if self.method == "median" else "mean"
            self._require("SimpleImputer", SimpleImputer)
            self._imputer = SimpleImputer(strategy = strat, **self.kwargs).fit(numeric)

        elif self.method == "knn": 
            self._require("KNNImputer", KNNImputer)
            self._imputer = KNNImputer(**self.kwargs).fit(numeric)

        elif self.method == "iterative":
            self._require("IterativeImputer", IterativeImputer)
            self._imputer = IterativeImputer(random_state = 40, **self.kwargs).fit(numeric)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Cleaner not fitted. Call 'fit' first.")

        X_out = X.copy()

        if self.method == "drop":
            return X_out[self.keep_cols_].copy()

        X_out = X_out[self.keep_cols_]

        numeric_cols = X_out.select_dtypes(include = np.number).columns
        if len(numeric_cols):
            X_out[numeric_cols] = self._imputer.transform(X_out[numeric_cols])

        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @staticmethod
    def _require(name: str, obj: Any):
        if obj is None:
            raise ImportError(f"{name} is required for this cleaning method. Install scikit-learn first.")

def clean_X(X: pd.DataFrame, method: CleanMethod = "drop", **kwargs: Any) -> pd.DataFrame:
    return FeatureCleaner(method = method, **kwargs).fit_transform(X)