import numpy as np

from preprocessing.timeseries import ewma_calc
from errors import IncorrectValue


class EWMAVolatilityModels:

    def __init__(
            self,
            prices,
            alpha: float = 0.72,
    ):

        if any(prices < 0):
            raise IncorrectValue('EWMAVolatility model - init prices must be greater than 0')

        self.prices = np.array(prices, dtype=np.float32)
        self.alpha = alpha
        self.log_ret = self.get_log_ret
        self.fact_vol = self.log_ret**2
        self.fitted_vol = self.create_nan_array

    @property
    def create_nan_array(self):
        return np.array([None for _ in range(len(self.log_ret))], dtype=np.float32)

    @property
    def get_log_ret(self):
        return np.diff(np.log(self.prices)).astype(np.float32)

    def ewma_simple(self):

        if not all(np.isnan(self.fitted_vol)):
            self.fitted_vol = self.create_nan_array

        volatility = [self.fact_vol[0]]

        for i in range(1, len(self.fact_vol)):
            volatility.append(ewma_calc( self.fact_vol[i-1], volatility[-1], self.alpha))

        self.fitted_vol[:] = np.array(volatility, dtype=np.float32)

        return self

    @staticmethod
    def fisher_index(array: np.ndarray):
        return [1 if array[i] < 0 else 0 for i in range(len(array))]

    def neg_ewma(self, period: int = 20):

        if len(self.log_ret) < period:
            raise IncorrectValue('Incorrect shapes of prices array and period')

        if not all(np.isnan(self.fitted_vol)):
            self.fitted_vol = self.create_nan_array

        a0 = np.var(self.fact_vol[:period])
        neg_sigmas = self.fisher_index(self.log_ret) * self.fact_vol
        volatility = [self.fact_vol[period - 1]]

        for i in range(period, len(self.log_ret)):
            volatility.append(a0 + ewma_calc(neg_sigmas[i-1], volatility[-1], self.alpha))

        self.fitted_vol[period - 1:] = np.array(volatility, dtype=np.float32)

        return self

    def mean_ewma(self, period: int = 20):

        if not all(np.isnan(self.fitted_vol)):
            self.fitted_vol = self.create_nan_array

        if not all(np.isnan(self.fitted_vol)):
            self.fitted_vol = self.create_nan_array

        a0 = np.mean(self.fact_vol[:period])
        neg_sigmas = self.fisher_index(self.log_ret) * self.fact_vol
        volatility = [self.fact_vol[period - 1]]

        for i in range(period, len(self.log_ret)):
            volatility.append(a0 + ewma_calc(neg_sigmas[i-1], volatility[-1], self.alpha))

        self.fitted_vol[period - 1:] = np.array(volatility, dtype=np.float32)

        return self

    def realized_volatility(self, period: int = 20):

        volatility = [self.fact_vol[period - 1]]

        for i in range(period, len(self.log_ret)):
            volatility.append(np.sum(self.fact_vol[i-period : i]))

        self.fitted_vol[period - 1:] = np.array(volatility, dtype=np.float32)

        return self

    def neg_realized_volatility(self, period: int = 20):

        volatility = [self.fact_vol[period - 1]]
        neg_sigmas = self.fisher_index(self.log_ret) * self.fact_vol

        for i in range(period, len(self.log_ret)):
            if not all(neg_sigmas[i - period : i] == 0):
                volatility.append(np.sum(neg_sigmas[i - period : i]))
            else:
                volatility.append(np.mean(self.fact_vol[i - period : i]))

        self.fitted_vol[period - 1:] = np.array(volatility, dtype=np.float32)

        return self


class EWMAVolatilityCCM:
    pass


if __name__ == '__main__':
    np.random.seed(10)
    y = np.random.normal(loc=0, scale=1, size=100).cumsum() + 1100
    simple_ewma = EWMAVolatilityModels(y).ewma_simple()
    negative_ewma = EWMAVolatilityModels(y).neg_ewma()
    mean_ewma = EWMAVolatilityModels(y).mean_ewma()
    realized_vol = EWMAVolatilityModels(y).realized_volatility(period=5)
    neg_realized = EWMAVolatilityModels(y).realized_volatility(period=5)
