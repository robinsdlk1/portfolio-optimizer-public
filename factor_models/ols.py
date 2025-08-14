import pandas as pd

from typing import Sequence, Optional

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


class OLSFactorModel:
    def __init__(self, random_state: int = 40):
        self.random_state = random_state
        self.preprocessor: ColumnTransformer | None = None
        self.model = LinearRegression()
        self.feature_names_: Sequence[str] | None = None
        self.betas: pd.Series | None = None
        self.intercept_: float | None = None
        self.is_fitted = False

    def _build_preprocessor(self, X: pd.DataFrame, categories: Optional[Sequence[Sequence[str]]] = None) -> ColumnTransformer:
        num_cols = X.select_dtypes(include = "number").columns
        cat_cols = X.select_dtypes(exclude = "number").columns
        
        num_steps = [("imputer", IterativeImputer(random_state = self.random_state)), ("scaler",  StandardScaler())]
        num_pipe = Pipeline(num_steps)
        
        cats = 'auto' if categories is None else categories
        try:
            encoder = OneHotEncoder(drop = "first", handle_unknown = "ignore", sparse_output = False, categories = cats)
        except TypeError:
            encoder = OneHotEncoder(drop = "first", handle_unknown = "ignore", sparse = False, categories = cats)
        
        cat_pipe = Pipeline([("encoder", encoder)])

        return ColumnTransformer(transformers = [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder = "drop", verbose_feature_names_out = False)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OLSFactorModel":
        self.preprocessor = self._build_preprocessor(X)
        X_proc = self.preprocessor.fit_transform(X)

        if X_proc.shape[0] < X_proc.shape[1]:
            raise ValueError("OLS ill-posed: fewer rows than columns after encoding. Switch to PCA or drop factors.")

        self.model.fit(X_proc, y)

        self.feature_names_ = self.preprocessor.get_feature_names_out()
        self.betas = pd.Series(self.model.coef_, index=self.feature_names_)
        self.intercept_ = float(self.model.intercept_)
        self.is_fitted = True
        return self

    def _transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.preprocessor.transform(X)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_proc = self._transform(X)
        return pd.Series(self.model.predict(X_proc), index=X.index)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.model.score(self._transform(X), y)