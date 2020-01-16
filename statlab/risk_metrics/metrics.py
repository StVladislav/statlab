import numpy as np
import scipy.stats as sts


def maxdrawdown(series: np.ndarray) -> tuple:
    """Calculate maximum drawdowns and their time of input series

    Arguments:
    series - np.ndarray of prices
    ----------------

    EXAMPLE:
       >>> y = sts.norm.rvs(size=100).cumsum()
       >>> dd = maxdrawdown(y)
       >>> dd[0] # This is calculated maxdrawdown

    Return tuple; 0 - value of maxdrawdown; 1 - dict: drowdowns - all drawdowns;
    drawdowns_time - time of each drawdown; 2 - list of event start indices
    ----------------
    P.S. If series at last iteration has drawdown
    algorithm will notify about it.
    """
    assert isinstance(series, np.ndarray), 'Incorrect type of series for maxdrawdown. Its must be np.ndarray'

    drawdowns = []
    drawdowns_time = []
    drawdowns_begin = []

    current_dd = None
    start_time_dd = None
    possible_dd = None

    for i in range(1, len(series)):
        if current_dd is None:
            if series[i] < series[i - 1]:
                current_dd = series[i - 1]
                possible_dd = series[i]
                start_time_dd = i - 1
                drawdowns_begin.append(start_time_dd)
        elif series[i] < current_dd:
            if series[i] < possible_dd:
                possible_dd = series[i]
        elif series[i] > current_dd:
            drawdowns.append(possible_dd - current_dd)
            drawdowns_time.append(i - start_time_dd - 1)
            current_dd = None
            start_time_dd = None
            possible_dd = None

    max_drawdown = np.min(drawdowns)

    if current_dd is not None:
        max_drawdown = possible_dd - current_dd
        print(f'Drawdown is not over yet! Current max drawdown is {possible_dd - current_dd}')

    return max_drawdown, dict(drawdowns=np.array(drawdowns), drawdowns_time=np.array(drawdowns_time)), drawdowns_begin


def kupiec_test():
    """Kupiec test for evaluation of VaR estimations
    """
    pass


if __name__ == '__main__':
    pass