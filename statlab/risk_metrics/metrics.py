import numpy as np
import scipy.stats as sts
from preprocessing.timeseries import points_of_crossing


def maxdrawdown_by_trend(close, trend):
    cross = points_of_crossing(close, trend)
    drawdowns_yields = []
    drawdowns_indices = []
    drawdowns_time = []
    for i in range(len(cross['all'])-1):
        start = cross['all'][i]
        end = cross['all'][i+1]

        if start - end == 0:
            if close[start] > close[end]:
                drawdowns_indices.append(end)
                drawdowns_yields.append(np.log(close[end] / close[start]))
                drawdowns_time.append(1)
                continue
        dd = np.min(np.log(close[start : end + 1] / close[start]))
        dd_ind = np.argmin(np.log(close[start : end + 1] / close[start]))
        if dd < 0:
            drawdowns_yields.append(dd)
            drawdowns_indices.append(cross['all'][i] + dd_ind)
            drawdowns_time.append(drawdowns_indices[-1] - start)
    res = {
        'drawdowns_yields': np.array(drawdowns_yields),
        'drawdowns_time': np.array(drawdowns_time),
        'drawdowns_indices': np.array(drawdowns_indices)
    }

    return res


def maxdrawdown(series: np.ndarray) -> tuple:
    """Calculate maximum drawdowns and their time of input series

    Arguments:
    series - np.ndarray of prices
    ----------------

    EXAMPLE:
       >>> y = sts.norm.rvs(size=100).cumsum()
       >>> dd = maxdrawdown(y)
       >>> dd[0] # This is calculated maxdrawdown

    Return tuple;
    0 - value of maxdrawdown;
    1 - dict:
        drowdowns - all drawdowns;
        drawdowns_time - time of each drawdown;
        drawdowns_yield - all drawdowns as pct changes
    2 - list of event start indices
    ----------------
    P.S. If series at last iteration has drawdown
    algorithm will notify about it.
    """
    assert isinstance(series, np.ndarray), 'Incorrect type of series for maxdrawdown. Its must be np.ndarray'

    drawdowns = []
    drawdowns_time = []
    drawdowns_begin = []
    drawdowns_yield = []

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
            drawdowns_yield.append(possible_dd / current_dd)
            drawdowns_time.append(i - start_time_dd - 1)
            current_dd = None
            start_time_dd = None
            possible_dd = None

    max_drawdown = np.min(drawdowns)

    if current_dd is not None:
        max_drawdown = possible_dd / current_dd
        print(f'Drawdown is not over yet! Current max drawdown is {max_drawdown}')

    to_ret = (
        max_drawdown,
        dict(
            drawdowns=np.array(drawdowns),
            drawdowns_yield=np.array(drawdowns_yield),
            drawdowns_time=np.array(drawdowns_time)
        ),
        drawdowns_begin
    )

    return to_ret


def kupiec_test():
    """Kupiec test for evaluation of VaR estimations
    """
    pass


class AltmanZscore:
    pass


if __name__ == '__main__':
    pass