from flask import Blueprint

from portfolio_optimizer.interface.routes_home import bp_home
from portfolio_optimizer.interface.routes_data import bp_data
from portfolio_optimizer.interface.routes_factors import bp_factors
from portfolio_optimizer.interface.routes_optimizers import bp_optimizers
from portfolio_optimizer.interface.routes_backtest import bp_backtest

bp = Blueprint("ui", __name__)
bp.register_blueprint(bp_home)
bp.register_blueprint(bp_data)
bp.register_blueprint(bp_factors)
bp.register_blueprint(bp_optimizers)
bp.register_blueprint(bp_backtest)