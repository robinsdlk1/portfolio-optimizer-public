import pandas as pd
from pathlib import Path
from typing import Union, Optional, List

class FactorDataset:
    def __init__(self, source: str, path: Union[str, Path], tickers: Optional[List[str]] = None):
        self.source = source.lower()
        self.path = Path(path)
        self.tickers = tickers

        self.data = self._load_data()
        self._clean_column_names()

    def _load_data(self) -> pd.DataFrame:
        if self.source == "manual":
            return pd.read_csv(self.path)

        elif self.source == "polygon":
            file = self._find_latest_polygon_csv()
            df = pd.read_csv(file)

            if self.tickers is not None:
                df = df[df["Ticker"].isin(self.tickers)].copy()
                if df.empty:
                    raise ValueError("No tickers matched in polygon dataset.")
            return df
        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'manual' or 'polygon'.")

    def _find_latest_polygon_csv(self) -> Path:
        files = list(self.path.glob("polygon_factor_dataset_*.csv"))
        if not files:
            raise FileNotFoundError("No polygon dataset found in path.")
        latest = max(files, key = lambda f: f.stat().st_mtime)
        return latest

    def _clean_column_names(self):
        self.data.columns = [col.strip() for col in self.data.columns]

    def get_raw_data(self) -> pd.DataFrame:
        return self.data.copy()

    def get_features(self, exclude: Optional[List[str]] = None) -> pd.DataFrame:
        df = self.data.copy()
        if exclude is None:
            exclude = ["Ticker", "Date"]
        features = df.drop(columns = [col for col in exclude if col in df.columns], errors = "ignore")

        if "Ticker" in df.columns:
            features.index = df["Ticker"].values
        return features

    def get_tickers(self) -> pd.Series:
        return self.data["Ticker"].copy() if "Ticker" in self.data.columns else None

    def get_tickers_and_dates(self) -> pd.DataFrame:
        cols = [col for col in ["Ticker", "Date"] if col in self.data.columns]
        return self.data[cols].drop_duplicates().reset_index(drop=True)