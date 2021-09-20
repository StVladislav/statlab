import numpy as np

from distributions import ContinousDistributionEstimator, DiscreteDistributionEstimator
from risk_metrics.metrics import points_of_crossing, get_drawdown_time


class StochasticDrawdownApproach:
    def __init__(
        self, price, trend,
        alpha: float = 0.05,
        dist_continuous: list = ('gumbel_r', 'genhalflogistic', 'genextreme', 'gamma'),
        dist_discrete: list = ('poisson', 'planck', 'dlaplace')
    ):
        self.price = price
        self.trend = trend
        self.dist_continuous = ContinousDistributionEstimator(dist_continuous)
        self.dist_discrete = DiscreteDistributionEstimator(dist_discrete)
        self.crosses = points_of_crossing(price, trend)
        self.alpha = alpha
        self.current_risk = {}
        self.summary = {}

    def _dist_fit(self, cont_vals, disc_val):
        discrete = self.dist_discrete.fit(disc_val)
        continuous = self.dist_continuous.fit(cont_vals)

        return continuous, discrete

    def fit(self):
        drawdowns_time = np.array(get_drawdown_time(self.price, self.trend), dtype=np.int64)
        drowdowns_trend = np.diff(np.log(self.trend[self.crosses['all']]))
        drowdowns_trend = drowdowns_trend[drowdowns_trend < 0]
        dist_of_risks, dist_of_time = self._dist_fit(drowdowns_trend, drawdowns_time)
        self.summary['dist_of_risks'] = dist_of_risks
        self.summary['dist_of_time'] = dist_of_time
        self.current_risk['yield_risk'] = dist_of_risks.function.ppf(self.alpha)
        self.current_risk['time_risk'] = dist_of_time.function.ppf(1 - self.alpha)
        self.summary = {**self.summary, **self.current_risk}
