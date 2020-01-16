import numpy as np
import scipy.stats as sts

from stats_utils import mean_absolute_percentage_error


class GeometricBrownianMotion:
    """This class is used to generate stochastic process of geometric brownian motion
     for simulation of dynamics of given random variable
     (for example price of a financial asset)

    So - price at time period t0
    T - length of simulated values (prediction horizon)
    dt - time step
    mu - mean (for example of log. returns)
    sigma - standard deviation (for example of log returns)
    n_iter - number of simulations

    Note
    ______________
    For example we study daily close prices of a given stock.
    It is required to conduct simulations for determining of possible borders
    for 2 years ahead. Then dt will be 1/365, and T (prediction horizon) - 2.
    """
    def __init__(self, So: float, T: int, dt: float, mu: float, sigma: float, n_iter: int):
        self.initial_price = So
        self.prediction_horizon = T
        self.dt = dt
        self.mean = mu
        self.std = sigma
        self.var = sigma**2
        self.n_iter = n_iter
        self.n_points = int(T / dt)
        self.time = np.linspace(0, self.prediction_horizon, self.n_points)

    @property
    def wiener_process(self):
        z = np.random.randn(self.n_points) * np.sqrt(self.dt)
        w = np.cumsum(z)

        return w

    @property
    def gbm(self):
        drift = (self.mean - 0.5 * self.var) * self.time
        diffusion = self.std * self.wiener_process

        return self.initial_price * np.exp(drift + diffusion)

    def sample_simulations(self, seed: bool = True):
        """Generates simulations using given parameters.
        """
        sample_matrix = np.empty((self.n_points + 1, self.n_iter))
        sample_matrix[0, :] = self.initial_price

        for i in range(sample_matrix.shape[1]):
            if seed:
                np.random.seed(i)
            sample_matrix[1:, i] = self.gbm

        return sample_matrix


class CorrelatedGeometricBrownianMotion(GeometricBrownianMotion):
    """This class is used to generate stochastic process of geometric brownian motion
     for simulation of dynamics of given random variable
     (for example price of a financial asset), but selects only simulations that are
     most close to the given time series (judging by Pearson correlation).
     There is also a method that selects simulations on the basis of mean absolute percentage error (mae)


    So - price at time period t0
    T - length of simulated values (prediction horizon)
    dt - time step
    mu - mean (for example of log. returns)
    sigma - standard deviation (for example of log returns)
    n_iter - number of simulations

    Note
    ______________
    For example we study daily close prices of a given stock.
    It is required to conduct simulations for determining of possible borders
    for 2 years ahead. Then dt will be 1/365, and T (prediction horizon) - 2.
    """

    def __init__(self, T: int, dt: float, mu: float, sigma: float, n_iter: int):
        super(CorrelatedGeometricBrownianMotion, self).__init__(0, T, dt, mu, sigma, n_iter)

    def correlated_gbm(self, x: np.ndarray, required_corr: float = 0.5) -> np.ndarray:

        self.initial_price = x[0]
        result_matrix = []
        shape_result_matrix = 0
        max_iter = self.n_iter * 2
        current_iter = 0

        while shape_result_matrix != self.n_iter:
            if current_iter == max_iter:
                break
            result_gbm = self.gbm

            if sts.pearsonr(x, result_gbm[:len(x)])[0] >= required_corr:
                result_matrix.append(result_gbm)
                shape_result_matrix += 1
            current_iter += 1

        return np.transpose(np.array(result_matrix))

    def mae_gbm(self, x: np.ndarray, required_precision: float) -> np.ndarray:

        self.initial_price = x[0]
        result_matrix = []
        shape_result_matrix = 0
        max_iter = 100000
        current_iter = 0

        while shape_result_matrix != self.n_iter:
            if current_iter == max_iter:
                break
            result_gbm = self.gbm

            if mean_absolute_percentage_error(x, result_gbm[:len(x)]) <= required_precision:
                result_matrix.append(result_gbm)
                shape_result_matrix += 1
            current_iter += 1

        return np.transpose(np.array(result_matrix))


class MultivariateGeometricBrownianMotion:
    pass


if __name__ == '__main__':
    pass
