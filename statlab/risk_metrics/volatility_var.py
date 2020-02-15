import numpy as np
import scipy.stats as sts

from graphs.plot import plot_lines
from distributions import ContinousDistributionEstimator
from risk_metrics.simple_var import ValueAtRisk
from volatility_models.ewma import EWMAVolatilityModels


class ValueAtRiskEWMA(ValueAtRisk):
    """
    This class allows to assess risks using parametric Value-at-Risk
    methodology with consideration of volatility prediction.
    The most simple example of volatility predictions can be
    EWMA model estimations.
    """
    def __init__(self, prices, dist: tuple = None):
        super(ValueAtRiskEWMA, self).__init__(prices=prices)
        self.dist = dist
        self.optimized_dist = self.distribution_optimize
        self.top_dist = self.optimized_dist.dist
        self.function = self.optimized_dist.function

    def fit(self, fitted_sigma: np.ndarray, alpha: float = 0.05):

        estimated_distribution = ContinousDistributionEstimator(self.dist)
        estimated_distribution.fit(self.log_ret)

        self.historical_risks[:] = self.function.mean() + fitted_sigma * sts.norm.ppf(alpha)
        self.current_risk = self.historical_risks[-1]

        return self

    @property
    def distribution_optimize(self):
        dist = ContinousDistributionEstimator(self.dist)
        dist.fit(self.log_ret)

        return dist


if __name__ == '__main__':
    np.random.seed(10)
    y = np.random.normal(loc=0, scale=1, size=100).cumsum() + 1100
    simple_ewma = EWMAVolatilityModels(y).ewma_simple()
    fitted_sigma = simple_ewma.fitted_vol**0.5
    ewma_var = ValueAtRiskEWMA(y).fit(fitted_sigma, 0.05)
    plot_lines((ewma_var.log_ret, ewma_var.historical_risks), title='Log ret VS EWMA Value-at-Risk')
