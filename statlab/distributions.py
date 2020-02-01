import warnings

import numpy as np
import scipy.stats as sts

from scipy.optimize import minimize


class DiscreteDistributionsParams:

    @staticmethod
    def initialize_params(dist: str, x: np.ndarray) -> tuple:
        distribution_params = dict(
            poisson=(np.mean(x),),
            dlaplace=(np.random.exponential(0.5),),
            bernoulli=(sum(x == 1) / len(x),),
            planck=(np.random.uniform(0.2, 0.7),),
            binom=(len(x), max(np.unique(x)))
        )

        return distribution_params[dist]


class DiscreteDistributionEstimator(DiscreteDistributionsParams):

    DEFAULT_DISTRIBUTIONS = (
        'poisson',
        'dlaplace',
        'binom',
        'planck',
    )

    def __init__(self, dist: str or tuple = None):

        self.loglikelihood = None
        self.function = None
        self.params = None

        if dist is not None:
            self.dist = (dist,) if isinstance(dist, str) else dist
        else:
            self.dist = self.DEFAULT_DISTRIBUTIONS

    @staticmethod
    def calc_negloglike(params: tuple, x: np.ndarray, dist: str) -> float:
        return -getattr(sts, dist)(*params).logpmf(x).sum()

    def optimize_loglike(self, dist: str, params: tuple, x: np.ndarray):

        params = np.array(params)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            optimization_result = minimize(self.calc_negloglike, x0=params, args=(x, dist), method='SLSQP')

        return optimization_result

    def fit(self, x: np.ndarray):

        current_loglike = -np.inf
        current_dist = None

        for i in self.dist:
            init_params = self.initialize_params(i, x)
            opt_negloglike = self.optimize_loglike(i, init_params, x)
            function = getattr(sts, i)(*opt_negloglike.x)
            loglikelihood = -opt_negloglike.fun

            if loglikelihood > current_loglike:
                current_loglike = loglikelihood
                self.params = opt_negloglike.x
                self.function = function
                self.loglikelihood = loglikelihood
                current_dist = i

        self.dist = current_dist

        return self.function


class ContinousDistributionEstimator:
    DEFAULT_DISTRIBUTIONS = (
        'norm',
        't',
        'laplace',
        'genlogistic',
        'gennorm',
    )

    def __init__(self, dist: str or tuple = None):
        self.loglikelihood = None
        self.function = None
        self.params = None

        if dist is not None:
            self.dist = (dist,) if isinstance(dist, str) else dist
        else:
            self.dist = self.DEFAULT_DISTRIBUTIONS

    @staticmethod
    def calc_loglike(dist: str, params: tuple, x: np.ndarray) -> float:
        return getattr(sts, dist)(*params).logpdf(x).sum()

    def fit(self, x: np.ndarray):

        current_loglike = -np.inf
        current_dist = None

        for i in self.dist:
            params = getattr(sts, i).fit(x)
            function = getattr(sts, i)(*params)
            loglikelihood = self.calc_loglike(i, params, x)

            if loglikelihood > current_loglike:
                current_loglike = loglikelihood
                self.params = params
                self.function = function
                self.loglikelihood = loglikelihood
                current_dist = i

        self.dist = current_dist

        return self.function


if __name__ == '__main__':
    y_c = sts.norm(loc=0, scale=1).rvs(size=100).cumsum()
    y_d = sts.poisson(10).rvs(size=100)
    fit_dist = ContinousDistributionEstimator()
    fit_dist.fit(y_c)
    print(fit_dist.dist)  # best distribution using negative loglikelihood
    print(fit_dist.function.mean())  # expeted mean of best distribution
    print(fit_dist.function.interval(0.95))  # 95% interval of dist

    fit_dist_d = DiscreteDistributionEstimator()
    fit_dist_d.fit(y_d)
    print(fit_dist_d.dist)  # best distribution using negative loglikelihood
    print(fit_dist_d.function.mean())  # expeted mean of best distribution
    print(fit_dist_d.function.interval(0.95))  # 95% interval of dist
