import pandas as pd
from typing import Optional, Dict, Any

_state = {"X": None, "y": None, "R": None, "factor_model": None, "horizon": None, "portfolio_weights": None, "optimizer_specs": None, "factor_date": None, "returns_end": None}

def init_state():
    _state["X"] = None
    _state["y"] = None
    _state["R"] = None
    _state["factor_model"] = None
    _state["portfolio_weights"] = None
    _state["optimizer_specs"] = None
    _state["factor_date"] = None
    _state["returns_end"] = None

def set_data_state(X: pd.DataFrame, y: pd.Series, R: pd.DataFrame, horizon: int):
    _state["X"] = X
    _state["y"] = y
    _state["R"] = R
    _state["horizon"] = horizon

def set_factor_model(model: Dict[str, Any]):
    _state["factor_model"] = model

def get_factor_model() -> Optional[Dict[str, Any]]:
    return _state.get("factor_model")

def set_X(X) -> Optional[pd.DataFrame]:
    _state["X"] = X

def get_X() -> Optional[pd.DataFrame]:
    return _state["X"]

def set_y(y) -> Optional[pd.DataFrame]:
    _state["y"] = y

def get_y() -> Optional[pd.Series]:
    return _state["y"]

def set_R(R) -> Optional[pd.DataFrame]:
    _state["R"] = R

def get_R() -> Optional[pd.DataFrame]:
    return _state["R"]

def set_horizon(horizon: int):
    _state["horizon"] = horizon

def set_factor_date(d):
    _state["factor_date"] = pd.to_datetime(d)

def get_factor_date():
    return _state.get("factor_date")

def set_returns_end(d):
    _state["returns_end"] = pd.to_datetime(d)

def get_returns_end():
    return _state.get("returns_end")

def get_horizon():
    return _state.get("horizon")

def set_portfolio_weights(weights_dict):
    _state["portfolio_weights"] = weights_dict

def get_portfolio_weights():
    return _state.get("portfolio_weights")

def set_optimizer_specs(specs: dict):
    _state["optimizer_specs"] = specs

def get_optimizer_specs():
    return _state.get("optimizer_specs")

def clear_state():
    init_state()