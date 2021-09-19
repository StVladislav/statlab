import numpy as np
import scipy.stats as sts

from scipy.signal import periodogram

from stats_utils import array_fill_nan
from errors import LengthCriteria, IncorrectType, IncorrectValue


def points_of_crossing(price, trend):
    trend_gap = price - trend
    cross_points = {
        'up': [],
        'down': [],
        'all': []
    }

    for i in range(1, len(trend_gap)):
        if trend_gap[i] > 0 >= trend_gap[i - 1]:
            cross_points['up'].append(i - 1)
            cross_points['all'].append(i - 1)
        elif trend_gap[i] < 0 <= trend_gap[i - 1]:
            cross_points['down'].append(i - 1)
            cross_points['all'].append(i - 1)

    return cross_points


def ewma_calc(x: float or int, prev: float or int, alpha: float = 0.4) -> float:
    """Calculates EWMA with slope alpha at time t of
    any time series

    Returns EWMA(x, alpha) value at moment t.

    Arguments
    --------------
    x - float or int (including numpy type); current point
    prev - float or int (including numpy type); previous point
    alpha - float: slope parameter of EWMA function

    Return
    EWMA value float
    """
    if not isinstance(alpha, float):
        raise IncorrectType('Incorrect type of prev for ewma_calc')

    return alpha * x + (1 - alpha) * prev


def ewma_trend(x: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """EWMA filtering of time series x with
    slope alpha

    Arguments
    --------------
    x - np.ndarray: series with values (e.g. prices) for which its necessary to calculate ewma
    alpha - float: slope parameter of EWMA function

    EXAMPLE:
        >>> np.random.seed(10)
        >>> series = sts.norm.rvs(size=100).cumsum()
        >>> trend = ewma_trend(series, 0.28)
    --------------
    Return np.ndarray vector of ewma-filtring given by series x
    """
    if not isinstance(x, np.ndarray):
        raise IncorrectType('Incorrect type of x for ewma_trend')

    if not isinstance(alpha, float):
        raise IncorrectType('Incorrect type of alpha for ewma_trend')

    trend = [x[0]]

    for i in range(1, len(x)):
        trend.append(ewma_calc(x[i], trend[-1], alpha=alpha))

    return np.array(trend)


def ma_trend(x: np.ndarray, window_size: int = 21) -> np.ndarray:
    """Calculates moving average by series

    Arguments
    --------------
    x - np.ndarray: series with values (e.g. prices) for which it is necessary to calculate moving average
    window_size - int: window for calculating moving average

    EXAMPLE:
        >>> np.random.seed(10)
        >>> series = sts.norm.rvs(size=100).cumsum()
        >>> trend = ma_trend(series, 10)
    --------------
    Returns np.ndarray vector of moving average given by series x
    """
    if not isinstance(x, np.ndarray):
        raise IncorrectType('Incorrect type of x for ma_trend')

    if not isinstance(window_size, int):
        raise IncorrectType('Incorrect type of prev for ma_trend')

    ma = [None for _ in range(window_size)]

    for i in range(window_size, len(x)):
        ma.append(np.mean(x[i-window_size : i+1]))

    return np.array(ma)


def std_rolling(x: np.ndarray, window_size: int = 21) -> np.ndarray:
    """Calculates moving std from series

    Arguments
    --------------
    x - np.ndarray: series with values (e.g. prices) for which its necessary to calculate moving std
    window_size - int: window for calculating moving std

    EXAMPLE:
        >>> np.random.seed(10)
        >>> series = sts.norm.rvs(size=100).cumsum()
        >>> std = std_rolling(series, 10)
    --------------
    Returns np.ndarray vector of moving std given by series x
    """
    if not isinstance(x, np.ndarray):
        raise IncorrectType('Incorrect type of x for std_rolling')

    if not isinstance(window_size, int):
        raise IncorrectType('Incorrect type of prev for std_rolling')

    std = [None for _ in range(window_size)]

    for i in range(window_size, len(x)):
        std.append(np.std(x[i-window_size : i+1]))

    return np.array(std)


def standard_scaler_rolling(x: np.ndarray, window_size: int = 21) -> np.ndarray:
    """Conversion (scaling) of time series using rolling mean and std

    Arguments
    --------------
    x - np.ndarray: series with values (e.g. prices) for which it is necessary to scaling
    window_size - int: window for calculating rolling scale

    Note:
    --------------
    scale_x = rolling ma(xt-window size: xt) / rolling std(xt-window size: xt)

    EXAMPLE:
        >>> np.random.seed(10)
        >>> series = sts.norm.rvs(size=100).cumsum()
        >>> scale_series = standard_scaler_rolling(series, 10)
    --------------
    Return np.ndarray vector of scaled series x
    """
    if not isinstance(x, np.ndarray):
        raise IncorrectType('Incorrect type of x for standard_scaler_rolling')

    if not isinstance(window_size, int):
        raise IncorrectType('Incorrect type of prev for standard_scaler_rolling')

    ma, std = ma_trend(x, window_size), std_rolling(x, window_size)
    scale_x = [None for _ in range(window_size)]

    for i in range(window_size, len(x)):
        scale_x.append((x[i] - ma[i]) / std[i])

    return np.array(scale_x)


def finding_seasonal(series: np.ndarray) -> int:
    """Find seasonal of time series
    using spectral density
    """
    hertz, power = periodogram(series)

    return round(1 / hertz[np.argmax(power)])


def lagged_feature(series: np.ndarray, lagg: int = 1, endog: bool = False, dropna: bool = False) -> np.ndarray:
    """Create lag of feature

    EXAMPLE:
        >>> t = np.array([1,2,3,4,5,6,7,8,9])
        >>> lagged_feature(t, lagg=2, endog=False)
        >>> lagged_feature(t, lagg=2, endog=True)
        >>> lagged_feature(t, lagg=2, endog=False, dropna=True)
    """
    if len(series) <= lagg:
        raise LengthCriteria

    lagged = np.empty(len(series))
    lagged[lagg:] = series[:-lagg]
    lagged[:lagg] = None

    if endog:
        lagged = np.column_stack((series, lagged))

    if dropna:
        lagged = lagged[~np.isnan(lagged).any(axis=1)]

    return lagged


def lagged_matrix(
        series,
        fill = None,
        lagg: int = 1,
        endog: bool = False,
        full: bool = True,
        exog: np.ndarray = None,
        dropna: bool = False
) -> np.ndarray:
    """Create lagged matrix from input array

    Arguments
    --------------
    series - np.ndarray: series with values (e.g. prices)
    fill - value of callable function: how to fill nan values
    lagg - int: number of laggs
    endog - bool: if True included endog variable at the first column
    full - bool: include all laggs in return matrix
    exog - array like:  if exist added to return matrix
    dropna - bool: if True all nan values will be droped from returned matrix

    EXAMPLE:
        >>> t = np.array([1,2,3,4,5,6,7,8,9])
        >>> exog = np.array([11,12,13,14,15,16,17,18,19])
        >>> lagged_matrix(t, lagg=2, endog=False)
        >>> lagged_matrix(t, lagg=2, endog=True)
        >>> lagged_matrix(t, lagg=2, endog=False, dropna=True)
        >>> lagged_matrix(t, lagg=2, endog=False, exog=exog)
        >>> lagged_matrix(t, lagg=2, endog=True, exog=exog, dropna=True)

    """
    final_matrix = None

    if full:
        for i in range(1, lagg + 1):
            final_matrix = lagged_feature(series, lagg=i) \
                if final_matrix is None \
                else np.column_stack((final_matrix, lagged_feature(series, lagg=i)))
    else:
        final_matrix = lagged_feature(series, lagg=lagg)

    if exog is not None:
        assert len(series) == len(exog), 'Length must be equal of series and exog'
        final_matrix = np.column_stack((final_matrix, exog))

    if endog:
        final_matrix = np.column_stack((series, final_matrix))

    if fill is not None: # FIXME add check shape
        final_matrix = final_matrix if len(final_matrix.shape) == 2 else final_matrix.reshape(-1, 1)
        for i in range(final_matrix.shape[1]):
            final_matrix[:, i] = array_fill_nan(final_matrix[:, i], fill=fill).ravel()

    if dropna:
        final_matrix = final_matrix[~np.isnan(final_matrix).any(axis=1)]

    return final_matrix


def time_split_generator(x: np.ndarray, length_train: int, length_test: int):
    """

    Example:
         >>> x = np.arange(1, 21)
         >>> for i,j in time_split_generator(x, 10, 3): print(f"train is {i} and test is {j}")
    """
    start = 0
    delta = length_test + length_train
    end = start + delta

    while len(x) >= end:
        sample = x[start:end]
        train, test = sample[:length_train], sample[length_train:]
        start += length_test
        end = start + delta

        yield train, test


class TimeSeriesCv:

    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def split(self, X: np.ndarray):
        indexes = np.arange(len(X))

        for i in range(self.n_splits, X.shape[0], self.n_splits):
            yield indexes[i - self.n_splits: i], indexes[i - self.n_splits: i]

    def get_n_splits(self):
        return self.n_splits


def time_series_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 2) -> GeneratorExit:

    if len(X.shape) <= 1:
        raise IncorrectValue

    if X.shape[0] != len(y):
        raise IncorrectValue

    for i in range(k_folds, X.shape[0], k_folds):
        yield X[i-k_folds : i, :], y[i-k_folds : i]


if __name__ == '__main__':
    pass
