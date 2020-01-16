import numpy as np

from distributions import ContinousDistributionEstimator
from stochastic_models import GeometricBrownianMotion

from errors import IncorrectValue


class ValueAtRisk:

    def __init__(self, prices):
        self.prices = np.array(prices, dtype=np.float32)
        self.log_ret = self.get_log_ret
        self.current_risk = None
        self.historical_risks = self.create_nan_array

    @property
    def get_log_ret(self):

        if any(self.prices < 0):
            raise IncorrectValue('ValueAtRisk - init prices must be greater than 0')

        return np.diff(np.log(self.prices)).astype(np.float32)

    @property
    def get_price_from_ret(self):
        return np.append(np.nan, self.prices[:-1] * np.exp(self.log_ret))

    @property
    def get_risk_prices(self):

        if self.current_risk is None:
            raise ValueError('ValueAtRisk First you need calculate risk')

        return np.append(np.nan, np.exp(self.historical_risks) * self.prices[1:])

    @property
    def create_nan_array(self):
        return np.array([None for _ in range(len(self.log_ret))], dtype=np.float32)



class ValueAtRiskHistorical(ValueAtRisk):

    def __init__(self, prices):
        super(ValueAtRiskHistorical, self).__init__(prices=prices)

    def fit(self, alpha: float = 0.05):
        alpha *= 100
        self.current_risk = np.percentile(self.log_ret, alpha)
        self.historical_risks[:] = self.current_risk

        return self


class ValueAtRiskParametric(ValueAtRisk):

    def __init__(self, prices, dist: tuple = None):
        super(ValueAtRiskParametric, self).__init__(prices=prices)
        self.dist = dist
        self.optimized_dist = self.distribution_optimize
        self.top_dist = self.optimized_dist.dist
        self.function = self.optimized_dist.function

    def fit(self, alpha: float = 0.05):
        self.current_risk = self.function.ppf(alpha)
        self.historical_risks[:] = self.current_risk

        return self

    @property
    def distribution_optimize(self):
        dist = ContinousDistributionEstimator(self.dist)
        dist.fit(self.log_ret)

        return dist


class ValueAtRiskMonteCarlo(ValueAtRisk):

    def __init__(
            self,
            prices,
            dist: tuple = None,
            init_price_index: int = -1,
            prediction_horizon: int = 1,
            n_iter: int = 10000
    ):
        super(ValueAtRiskMonteCarlo, self).__init__(prices=prices)
        self.init_price_index = init_price_index
        self.prediction_horizon = prediction_horizon
        self.dist = dist
        self.optimized_dist = self.distribution_optimize
        self.top_dist = self.optimized_dist.dist
        self.function = self.optimized_dist.function
        self.samples = self.specify_mc_generator(
            init_price_index=init_price_index,
            prediction_horizon=prediction_horizon,
            n_iter=n_iter
        )

    def fit(self, alpha: float = 0.05):
        samples_yield = np.log(self.samples[-1, :] / self.samples[0, 0])
        self.current_risk = np.percentile(samples_yield, alpha * 100)
        risk_samples = self.current_risk

        if self.init_price_index > 0:
            risk_samples = []

            for i in range(self.prediction_horizon - self.init_price_index):
                log_ret = np.log(self.samples[i+1,:] / self.samples[0, 0])
                risk_samples.append(np.percentile(log_ret, alpha * 100))

        risk_samples = np.array(risk_samples, dtype=np.float32)
        self.historical_risks[-(self.prediction_horizon - self.init_price_index):] = risk_samples

        return self

    @property
    def distribution_optimize(self):
        dist = ContinousDistributionEstimator(self.dist)
        dist.fit(self.log_ret)

        return dist

    def specify_mc_generator(self, init_price_index, prediction_horizon, n_iter):
        gbm = GeometricBrownianMotion(
            So=self.prices[init_price_index],
            T=prediction_horizon,
            dt=1,
            n_iter=n_iter,
            mu=self.function.mean(),
            sigma=self.function.std()
        )

        return gbm.sample_simulations()


if __name__ == '__main__':
    np.random.seed(10)
    y = np.random.normal(loc=0, scale=1, size=100).cumsum() + 1100
    historical_var = ValueAtRiskHistorical(y).fit(0.05)
    parametric_var = ValueAtRiskParametric(y).fit(0.05)
    montecarlo_var = ValueAtRiskMonteCarlo(y, init_price_index=-1, prediction_horizon=1).fit(0.05)
    montecarlo_var2 = ValueAtRiskMonteCarlo(y, init_price_index=20, prediction_horizon=len(y)).fit(0.05)
    print(f'Hisorical VaR is {historical_var.current_risk}\nParametric VaR is {parametric_var.current_risk}\n'
          f'Monte Carlo VaR is {montecarlo_var.current_risk}\nMonte Carlo long period is'
          f' {montecarlo_var2.historical_risks}')