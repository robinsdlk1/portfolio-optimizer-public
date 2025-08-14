from portfolio_optimizer.data.factors import FactorDataset
from portfolio_optimizer.data.returns import ReturnsDataset

from typing import List, Optional, Tuple
import pandas as pd

class DataInterface:
    def __init__(self, factor_source: str, factor_path: str, tickers: Optional[List[str]], start_date: str, end_date: str, return_type: str = "log_ret", fwd_horizon: int = 63, save_returns: bool = True):
        self.factor_data = FactorDataset(source = factor_source, path = factor_path, tickers = tickers)

        self.returns_data = ReturnsDataset(tickers = self.factor_data.get_tickers().unique().tolist(), start_date = start_date, end_date = end_date, source = "yahooquery", return_type = return_type, fwd_horizons = [fwd_horizon], save = save_returns)
        self.fwd_horizon = fwd_horizon

    def get_clean_dataset(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        X_full = self.factor_data.get_raw_data()
        _, y = self.returns_data.align_forward_return(X_full, self.fwd_horizon, roll_next = True, strict = False)

        X_feat = self.factor_data.get_features()
        ticker_order = X_full.loc[y.index, "Ticker"].astype(str).values
        X_feat = X_feat.reindex(ticker_order)
        y.index = ticker_order

        R = self.returns_data.get_return_matrix()
        return X_feat, y, R