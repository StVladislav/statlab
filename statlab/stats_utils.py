import datetime

import numpy as np
import scipy.stats as sts


def array_drop_nan(y, axis: int = 0) -> np.ndarray:
    """
    Returns numpy-ndarray without NaN.

    Parameters
    ----------
    x : array_like or pandas DataFrame/Series with nan-values
        The array which contains or does not contain nan-values.
    axis : integer (default = 0)
        This parameter determines by which axis nan-values are dropped.
        If axis = 0 than columns which contain nan-values will be dropped.
        If axis = 1 rows which contain nan-values will be dropped,

    Returns
    -------
    y : ndarray with float32 dtype
        An array of the same shape as `y` without nan-values.
        If all columns/rows contain nan-values then an
        empty array will be returned.

    Examples
    --------
    >>> array_drop_nan(np.array([1,2,3,4, None]), axis=0)
    array([1., 2., 3., 4.], dtype=float32)
    >>> array_drop_nan(np.array([[1,2, None], [4,5,6]]), axis=1)
    array([[4., 5., 6.]], dtype=float32)
    >>> array_drop_nan(np.array([[1,2, None], [4,5,6]]), axis=0)
    array([[1., 2.],
          [4., 5.]], dtype=float32)
    """

    y = np.array(y, dtype=np.float32)

    if len(y.shape) == 1:
        y = y[~np.isnan(y)]
    elif axis == 0:
        y = y[:, ~np.any(np.isnan(y), axis=0)]
    elif axis == 1:
        y = y[~np.any(np.isnan(y), axis=1), :]

    return y


def array_fill_nan(y, fill, **_) -> np.ndarray:

    y = np.array(y, dtype=np.float32)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if callable(fill):
        for j in range(y.shape[1]):
            value = fill(array_drop_nan(y[:, j]))
            y[:, j] = array_fill_nan(y[:, j], value).ravel()
    else:
        y[np.isnan(y)] = fill

    return y


def share_missing(y, axis: int = 0):

    if axis not in (0, 1):
        raise ValueError('Axis must be in interval (0, 1) share_missing')

    y = np.array(y, dtype=np.float32)

    return np.isnan(y).sum(axis=axis) / y.shape[axis]


def timer(func):
    """Decorator for speed measurement of function
    This simple decorator adds print of
    spent time on execution of functions
    with args and kwargs

    Example:
        @timer
        def amount(a: float, b: float):
            return a + b

    ------------
    Return wrapper of function
    """

    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        stop = datetime.datetime.now()
        print(f'Time on function is: {stop - start}')

        return result

    return wrapper


def mean_absolute_percentage_error(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_fit) / y_true)) * 100


def sample_entropy(x, bins: int = 10):
    """Calculate sample entropy
    using frequency distribution of data x
    """
    return sts.entropy(sts.relfreq(x, numbins=bins)[0])


def correlation_tolerance(matrix, tol: float = 0.5, labels: list = None):
    matrix = np.array(matrix, dtype=np.float32)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Matrix must be semtric')

    tol = -np.inf if tol is None else tol
    upper_triu = np.triu_indices(matrix.shape[0])
    labels = labels if labels is not None else upper_triu[1]
    correlation = dict.fromkeys(labels)

    for i, j in zip(upper_triu[0], upper_triu[1]):
        if i == j:
            continue

        current_value = matrix[i, j]

        if correlation[labels[i]] is None:
            correlation[labels[i]] = dict()

        if np.abs(current_value) >= tol:
            correlation[labels[i]][labels[j]] = current_value

    return correlation


def top_correlation(array, count: int = 10, labels: list = None):

    array = np.array(array, dtype=np.float32)
    labels = labels if labels is not None else np.arange(len(array))
    indexies = np.argsort(array)

    return array[indexies][-count:], labels[indexies]


class Counter:
    """Realized counter of smth
    This class may be used if need to create custom
    counter using given conditions.
    """

    def __init__(self):
        self.current = 0

    def update(self):
        self.current = 0

    def increase(self):
        self.current += 1


if __name__ == '__main__':
    pass
