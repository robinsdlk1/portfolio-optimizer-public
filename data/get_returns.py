import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import json

from typing import List, Dict, Optional
from datetime import datetime as dt
from math import ceil
from yahooquery import Ticker as YQTicker

BASE_URL = "https://api.polygon.io"
API_KEY = os.getenv("POLYGON_API_KEY")

PG_YF_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "700.HK": "0700.HK",
}
# Someone should create the tickers mapping.

def convert_to_yahoo_ticker(ticker: str) -> str:
    return PG_YF_TICKER_MAP.get(ticker, ticker)

def request_with_retries(url: str, params: Optional[Dict] = None, max_retries: int = 5, backoff_factor: float = 60) -> Optional[requests.Response]:
    for _ in range(max_retries):
        response = requests.get(url, params = params)
        if response.status_code == 200:
            return response
        if response.status_code == 429:
            print("Rate limit hit. Retrying in %.0fs…", backoff_factor)
            time.sleep(backoff_factor)
        else:
            logging.error("Request failed (%s): %s", response.status_code, response.text)
            return None
    return None

def get_polygon_returns(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 10000,
        "apiKey": API_KEY,
    }

    response = request_with_retries(url, params)
    if response is None or "results" not in response.json():
        print("No return data for %s on Polygon", ticker)
        return None

    df = pd.DataFrame(response.json()["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit = "ms", utc = True).tz_localize(None)
    df = df.set_index("timestamp").rename(columns = {"c": "close"})[["close"]]
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()

def _yq_history_batch(tickers: List[str], start_date: str, end_date: str, interval: str = "1d", batch_size: int = 50) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    total_batches = ceil(len(tickers) / batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print("Yahooquery batch %d/%d - %s", i // batch_size + 1, total_batches, batch)

        try:
            yq = YQTicker(batch, asynchronous=False)
            hist = yq.history(start = start_date, end = end_date, interval = interval, adj_ohlc = True)
        except Exception as exc:
            logging.error("Batch failed for %s - %s: %s", batch, type(exc).__name__, exc)
            continue

        if hist.empty:
            print("Empty history for batch %s", batch)
            continue

        if isinstance(hist.index, pd.MultiIndex):
            tickers_in_response = hist.index.get_level_values(0).unique()
            for tic in tickers_in_response:
                sub = hist.xs(tic, level=0)
                df = _postprocess_yq_hist(sub)
                if df is not None:
                    out[tic] = df
        else:
            tic = batch[0]
            df = _postprocess_yq_hist(hist)
            if df is not None:
                out[tic] = df
        time.sleep(0.4)
    return out


def _postprocess_yq_hist(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    if raw.empty:
        return None

    raw.index = pd.to_datetime(raw.index, utc = True).tz_localize(None)

    if "close" not in raw.columns:
        if "adjclose" in raw.columns:
            raw = raw.rename(columns={"adjclose": "close"})
        else:
            return None

    df = raw[["close"]].copy()
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()

def get_all_returns(tickers: List[str], start_date: str, end_date: str, source: str = "yahooquery", batch_size: int = 50) -> Dict[str, pd.DataFrame]:
    if source not in {"yahooquery", "polygon"}:
        raise ValueError(f"Unknown source '{source}' - choose 'yahooquery' or 'polygon'.")

    if source == "polygon":
        result: Dict[str, pd.DataFrame] = {}
        for tic in tickers:
            print("Fetching Polygon returns for %s", tic)
            df = get_polygon_returns(tic, start_date, end_date)
            if df is not None:
                result[tic] = df
            time.sleep(1.25)
        return result

    return _yq_history_batch(tickers, start_date, end_date, batch_size = batch_size)


def validate_returns(returns_dict: Dict[str, pd.DataFrame], expected_tickers: Optional[List[str]] = None) -> None:
    empty, wrong_columns, missing = [], [], []

    if expected_tickers is not None:
        expected_set = set(expected_tickers)
        fetched_set = set(returns_dict.keys())
        missing = list(expected_set - fetched_set)

    for ticker, df in returns_dict.items():
        if df.empty:
            empty.append(ticker)
        elif not {"close", "ret", "log_ret"}.issubset(df.columns):
            wrong_columns.append(ticker)

    if missing:
        print("[MISSING] %d tickers with no data: %s", len(missing), missing)
    if empty:
        print("[EMPTY] %d tickers returned empty DF: %s", len(empty), empty)
    if wrong_columns:
        print("[BAD COLS] %d tickers missing expected columns: %s", len(wrong_columns), wrong_columns)

    if not (missing or empty or wrong_columns):
        print("All tickers passed integrity validation.")

def load_returns_from_json(path: str) -> Dict[str, pd.DataFrame]:
    with open(path, "r") as f:
        raw = json.load(f)

    out: Dict[str, pd.DataFrame] = {}
    for tic, rows in raw.items():
        df = pd.DataFrame(rows)

        ts_col = next(c for c in ("date", "Date_", "timestamp", "Date", "index") if c in df.columns)
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.rename(columns={f"close_{tic}": "close", "ret_": "ret", "log_ret_": "log_ret"})
        out[tic] = df.set_index(ts_col).sort_index()[["close", "ret", "log_ret"]].copy()

    return out

def save_returns_to_json(data: Dict[str, pd.DataFrame], prefix: str = "returns") -> None:
    os.makedirs("portfolio_optimizer/db", exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    filename = f"portfolio_optimizer/db/{prefix}_{timestamp}.json"

    json_data: Dict[str, List[Dict]] = {}
    for ticker, df in data.items():
        df_copy = df.copy()
        if df_copy.index.name is None:
             df_copy.index.name = "Date"
        df_copy = df_copy.reset_index()

        df_copy.columns = ["_".join(map(str, col)) if isinstance(col, tuple) else str(col)for col in df_copy.columns]

        for col in df_copy.columns:
            if np.issubdtype(df_copy[col].dtype, np.datetime64):
                df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d")

        json_data[ticker] = df_copy.to_dict(orient = "records")

    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)

    print("Saved returns to %s", filename)

def fetch_and_save_returns(ticker_list: List[str], start_date: str, end_date: str, source: str = "yahooquery", save: bool = True, batch_size: int = 50) -> Dict[str, pd.DataFrame]:
    print("Starting return fetching from source: %s", source)

    if source == "yahooquery":
        ticker_list = [PG_YF_TICKER_MAP.get(t, t) for t in ticker_list]

    returns = get_all_returns(ticker_list, start_date, end_date, source = source, batch_size = batch_size)
    validate_returns(returns, expected_tickers=ticker_list)

    if save:
        save_returns_to_json(returns, prefix = f"returns_{source}")

    return returns