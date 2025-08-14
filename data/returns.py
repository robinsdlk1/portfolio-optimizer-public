import pandas as pd
import numpy as np
import bisect

from typing import List, Optional, Dict
from datetime import timedelta, date
from portfolio_optimizer.data.get_returns import load_returns_from_json, fetch_and_save_returns, convert_to_yahoo_ticker

class ReturnsDataset:
    def __init__(self, tickers: List[str] | None, start_date: str | None, end_date: str | None, source: str = "yahooquery", *, json_path: str | None = None, return_type: str = "ret", fwd_horizons: Optional[List[int]] = None, save: bool = True):
        self.source = source.lower()
        self.return_type = return_type
        self.horizons = fwd_horizons or [21, 63] # 1, 3 trading months
        self.save = save
        self.json_path = json_path

        if self.source == "json":
            if json_path is None:
                raise ValueError("json_path must be provided for source = 'json'.")

            self.returns_dict = load_returns_from_json(json_path)
            self.tickers = list(self.returns_dict.keys())

            idx = pd.concat(self.returns_dict.values()).index
            self.start = idx.min().strftime("%Y-%m-%d")
            self.end   = idx.max().strftime("%Y-%m-%d")

        else:
            self.tickers = tickers
            self.start = start_date
            max_h = max(self.horizons)
            end_dt = pd.to_datetime(end_date)
            pad_end_dt = end_dt + timedelta(days = int(max_h * 1.5))
            yesterday = pd.Timestamp(date.today() - timedelta(days = 1))
            if pad_end_dt > yesterday:
                pad_end_dt = yesterday
            self.end = pad_end_dt.strftime("%Y-%m-%d")

            self.returns_dict = self._load_returns()

        self.return_matrix = self._build_return_matrix()
        self.forward_returns = self._compute_forward_returns()

    def _load_returns(self) -> Dict[str, pd.DataFrame]:
        if self.source == "json":
            return self.returns_dict
        return fetch_and_save_returns(ticker_list = self.tickers, start_date = self.start, end_date = self.end, source = self.source, save = self.save)

    def _build_return_matrix(self) -> pd.DataFrame:
        series_list = []
        for ticker, df in self.returns_dict.items():
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([str(i) for i in col if i]) for col in df.columns]
            try:
                series = df[[self.return_type]].copy()
                series.columns = [ticker]
                series_list.append(series)
            except Exception as e:
                print(f"Skipping {ticker}: {e}")
        matrix = pd.concat(series_list, axis = 1)
        matrix.index.name = "Date"
        return matrix.sort_index()

    def _compute_forward_returns(self) -> pd.DataFrame:
        frames = []
        for h in self.horizons:
            if self.return_type == "log_ret":
                fwd = (self.return_matrix.shift(-1).rolling(window = h, min_periods = h).sum())                         
            else:  # "ret"
                fwd = ((1 + self.return_matrix.shift(-1)).rolling(window = h, min_periods = h).apply(np.prod, raw = True) - 1)
            fwd.columns = [f"{c}_Fwd{h}d" for c in fwd.columns]
            frames.append(fwd)
        return pd.concat(frames, axis = 1)
        
    def compute_forward_return(self, horizon: int) -> pd.DataFrame:
        if horizon not in self.horizons:
            raise ValueError(f"{horizon} not in initial horizon list {self.horizons}")
        return self.forward_returns.filter(like=f"_Fwd{horizon}d")

    def get_return_matrix(self) -> pd.DataFrame:
        return self.return_matrix.copy()

    def get_forward_returns(self) -> pd.DataFrame:
        return self.forward_returns.copy()

    @staticmethod
    def _next_trading_day(idx: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
        pos = bisect.bisect_right(idx, date)
        if pos < len(idx):
            return idx[pos]
        return None

    def align_forward_return(self, X: pd.DataFrame, horizon: int, *, ticker_col: str = "Ticker", date_col: str = "Date", roll_next: bool = True, strict: bool = False):
        fwd = self.compute_forward_return(horizon)

        def _find_column(tic: str) -> str | None:
            for alias in (tic, convert_to_yahoo_ticker(tic)):
                col = f"{alias}_Fwd{horizon}d"
                if col in fwd.columns:
                    return col
            return None

        if ticker_col not in X.columns or date_col not in X.columns:
            raise ValueError(f"align_forward_return: '{ticker_col}' or '{date_col}' column missing (have {list(X.columns)})")

        def _fetch(row) -> float:
            col = _find_column(str(row[ticker_col]))
            if col is None: return np.nan
            
            d = row[date_col]

            if d in fwd.index: return fwd.at[d, col]

            if roll_next:
                d2 = self._next_trading_day(fwd.index, d)
                if d2 is not None:
                    return fwd.at[d2, col]
            return np.nan

        y = X.apply(_fetch, axis = 1).rename("FwdReturn")

        if strict and y.isna().any():
            bad = X.loc[y.isna(), [ticker_col, date_col]]
            raise ValueError(f"{len(bad)} forward returns missing even after roll-forward; examples:\n{bad.head(5).to_string(index = False)}")

        return X, y