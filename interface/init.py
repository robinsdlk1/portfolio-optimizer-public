from flask import Flask

from portfolio_optimizer.interface.routes_data import bp_data
from portfolio_optimizer.interface.routes_factors import bp_factors
from portfolio_optimizer.interface.routes_optimizers import bp_optimizers
from portfolio_optimizer.interface.routes_backtest import bp_backtest
from portfolio_optimizer.interface.routes_home import bp_home

def create_app():
    app = Flask(__name__)
    app.secret_key = "dev"

    app.register_blueprint(bp_home)
    app.register_blueprint(bp_data)
    app.register_blueprint(bp_factors)
    app.register_blueprint(bp_optimizers)
    app.register_blueprint(bp_backtest)

    return app

if __name__ == "__main__":
    create_app().run(debug = True, port = 8080)