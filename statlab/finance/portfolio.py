from typing import List

import numpy as np
import pandas as pd
import scipy.stats as sts
from scipy.optimize import minimize

from risk_metrics.simple_var import ValueAtRiskParametric
from stats_utils import NumpyDataFrame


class PortfolioCreator:

    shape = None
    portfolio_var = None
    portfolo_std = None
    portfolio_expected_yield = None
    weights = None
    opt_results = None
    eq_cons = None
    bounds = None

    def __init__(
        self,
        min_bound: float = 0.0,
        max_bound: float = 1.0,
        opt_function: str = "markovitz_min_var",
    ):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.opt_function = opt_function

    @property
    def get_optimization_function(self):
        self.create_constraints_and_bounds()

        funcs = {
            "markovitz_min_var": self.calculate_var,
            "sharpe_ratio": self.calculate_neg_sharp_ratio,
        }

        return funcs[self.opt_function]

    def create_constraints_and_bounds(self):
        m = self.shape
        self.eq_cons = {
            "type": "eq",
            "fun": lambda x: 1 - np.sum(x),
            "jac": lambda x: [-1.0 for _ in range(m)],
        }
        self.bounds = [(self.min_bound, self.max_bound) for _ in range(m)]

    def make_optimization(
        self,
        func: callable,
        covariance: np.ndarray,
        expected_yields_assets: np.ndarray,
        options: dict,
    ):
        initial_weights = self.generate_initial_weights(self.shape)

        res = minimize(
            func,
            bounds=self.bounds,
            method="SLSQP",
            constraints=[self.eq_cons],
            x0=initial_weights,
            args=(covariance, expected_yields_assets),
            options=options,
        )

        return res

    def optimize_portfolio(
        self,
        covariance: np.ndarray,
        expected_yields_assets: np.ndarray = None,
        epochs: int = 5,
        options: dict = {"maxiter": 1000}
    ):

        self.shape = covariance.shape[0]
        func = self.get_optimization_function
        prev_res = None

        for _ in range(epochs):
            res = self.make_optimization(
                func=func,
                covariance=covariance,
                expected_yields_assets=expected_yields_assets,
                options=options
            )

            if prev_res is None:
                prev_res = res
                continue

            if res.success and res.fun < prev_res.fun:
                prev_res = res

        self.portfolio_var = self.calculate_var(prev_res.x, covariance)
        self.portfolio_std = np.sqrt(self.portfolio_var)
        self.weights = prev_res.x
        self.portfolio_expected_yield = self.calculate_expected_yield(
            self.weights, expected_yields_assets
        )
        self.opt_results = prev_res

    def generate_initial_weights(self, m: int):
        initial_weights = np.random.uniform(
            self.min_bound, self.max_bound, size=m)
        return initial_weights / sum(initial_weights)

    @staticmethod
    def calculate_var(weight: np.ndarray, covariance: np.ndarray, *_):
        return weight.dot(covariance).dot(weight.T)

    @staticmethod
    def calculate_expected_yield(
        weight: np.ndarray, expected_yields_assets: np.ndarray
    ):
        return weight.dot(expected_yields_assets.T)

    def calculate_neg_sharp_ratio(
        self,
        weight: np.ndarray,
        covariance: np.ndarray,
        expected_yields: np.ndarray
    ):
        var = self.calculate_var(weight, covariance)
        port_yield = self.calculate_expected_yield(weight, expected_yields)

        return -port_yield / var


class ComponentValueAtRiskPortfolio:

    def __init__(
        self,
        covariance_matrix_assets: np.ndarray,
        value_at_risk_assets: np.ndarray,
        portfolio_weights: np.ndarray,
        portfolio_var: float,
        portfolio_value_at_risk: float,
        portfolio_value: float = 1.0,
        assets_name: List[str] = None
    ):
        self.portfolio_value_at_risk = portfolio_value_at_risk
        self.covariance_matrix_assets = covariance_matrix_assets
        self.value_at_risk_assets = value_at_risk_assets
        self.portfolio_value = portfolio_value
        self.portfolio_weights = portfolio_weights
        self.portfolio_var = portfolio_var
        self.shape = covariance_matrix_assets.shape[0]
        self.assets_name = assets_name

        if assets_name is None:
            self.assets_name = [f'asset_{i}' for i in range(self.shape)]

    @property
    def diversified_risk(self) -> float:
        return self.portfolio_value_at_risk * self.portfolio_value

    @property
    def undiversified_risk(self) -> float:
        return np.sum(self.portfolio_value * self.portfolio_weights
                      * self.value_at_risk_assets)

    @property
    def betas(self) -> np.ndarray:
        return self.portfolio_weights.dot(self.covariance_matrix_assets) \
            / self.portfolio_var

    @property
    def marginal_value_at_risk(self) -> np.ndarray:
        return self.portfolio_value_at_risk * self.portfolio_value \
            / self.portfolio_value * self.betas

    @property
    def component_value_at_risk(self) -> np.ndarray:
        return self.portfolio_weights * self.portfolio_value \
            * self.marginal_value_at_risk

    @property
    def risk_value_assets(self):
        return self.portfolio_value * self.portfolio_weights \
            * self.value_at_risk_assets

    def fit(self) -> dict:
        result = {}
        result['diversified_risk'] = self.diversified_risk
        result['undiversified_risk'] = self.undiversified_risk
        result['betas'] = NumpyDataFrame.from_numpy(
            self.betas.reshape(1, -1),
            columns=self.assets_name
        )
        result['marginal_value_at_risk'] = NumpyDataFrame.from_numpy(
            self.marginal_value_at_risk.reshape(1, -1),
            columns=self.assets_name
        )
        result['component_value_at_risk'] = NumpyDataFrame.from_numpy(
            self.component_value_at_risk,
            columns=self.assets_name
        )

        return result


class PortfolioOptimization(PortfolioCreator):

    prices: NumpyDataFrame = None
    log_yields: NumpyDataFrame = None
    distribution_assets: dict = None
    value_at_risk_assets: dict = None
    expected_yields_assets: dict = None
    portfolio_value_at_risk: float = None
    portfolio_yields: NumpyDataFrame = None  # TODO
    portfolio_distribution: dict = None
    component_value_at_risk: dict = None

    def __init__(
        self,
        min_bound: float = 0.0,
        max_bound: float = 1.0,
        opt_function: str = "markovitz_min_var",
        alpha: float = 0.05,
        alpha_dist: float = 0.1,
        dist_list: List[str] = None,
        portfolio_value: float = 1.0

    ):
        self.alpha = alpha
        self.alpha_dist = alpha_dist
        self.dist_list = dist_list
        self.portfolio_value = portfolio_value

        super().__init__(
            min_bound=min_bound,
            max_bound=max_bound,
            opt_function=opt_function
        )

    def calculate_value_at_risk_asset(self, log_yields: np.ndarray):
        return ValueAtRiskParametric(
            log_yields,
            self.alpha_dist,
            self.dist_list,
            is_log_yields=True
        ).fit(self.alpha)

    def fit(self, prices: pd.DataFrame):
        self.prices = NumpyDataFrame(prices)
        self.log_yields = self.prices.apply(np.log).diff
        self.distribution_assets = dict.fromkeys(self.prices.columns)

        self.covariance = self.log_yields.cov()
        self.shape = self.log_yields.shape[1]
        empty_array = np.ones((1, self.shape))

        self.value_at_risk_assets = NumpyDataFrame.from_numpy(
            empty_array,
            columns=self.log_yields.columns
        )
        self.expected_yields_assets = NumpyDataFrame.from_numpy(
            empty_array,
            columns=self.log_yields.columns
        )

        for col in self.log_yields.columns:
            var = self.calculate_value_at_risk_asset(self.log_yields[col])
            self.distribution_assets[col] = var.dist
            self.value_at_risk_assets[:, col] = var.current_risk
            self.expected_yields_assets[:, col] = var.dist['loc']

        self.optimize_portfolio(
            self.covariance.data,
            self.expected_yields_assets.data
        )

        self.weights = NumpyDataFrame.from_numpy(
            self.weights.reshape(1, -1),
            columns=self.prices.columns
        )

        self.portfolio_yields = self.log_yields.data.dot(
            self.weights.data.T
        ).ravel()

        portfolio_value_at_risk = ValueAtRiskParametric(
            self.portfolio_yields,
            self.alpha_dist,
            self.dist_list,
            is_log_yields=True
        ).fit(self.alpha)

        self.portfolio_distribution: dict = portfolio_value_at_risk.dist
        self.portfolio_value_at_risk: float = portfolio_value_at_risk.current_risk
        self.component_value_at_risk: dict = self.get_component_value_at_risk()

        return self

    def calculate_portfolio_yields(self, x, is_log_yields: bool = False):
        pass

    def calculate_portfolio_value(self):
        pass

    def get_component_value_at_risk(self) -> dict:
        cvar = ComponentValueAtRiskPortfolio(
            covariance_matrix_assets=self.covariance.data,
            value_at_risk_assets=self.value_at_risk_assets.data,
            portfolio_weights=self.weights.data,
            portfolio_var=self.portfolio_var,
            portfolio_value_at_risk=self.portfolio_value_at_risk,
            portfolio_value=self.portfolio_value,
            assets_name=self.prices.columns
        ).fit()

        return cvar


if __name__ == "__main__":
    X = sts.norm(loc=10000, scale=20000).rvs(size=(100, 5))
    X += 10000000
    cols = ['a', 'b', 'c', 'q', 'n']
    X = pd.DataFrame(X, columns=cols)
    port = PortfolioOptimization(
        opt_function='sharpe_ratio',
        max_bound=0.4,
        portfolio_value=15000
    ).fit(X)

    print(port.covariance)
    print(port.shape)
    print(port.expected_yields_assets)
    print(port.weights)
    print(port.opt_results)

    print(port.component_value_at_risk['diversified_risk'])
    print(port.component_value_at_risk['undiversified_risk'])
    print(port.component_value_at_risk['marginal_value_at_risk'])
    print(port.component_value_at_risk['component_value_at_risk'])
