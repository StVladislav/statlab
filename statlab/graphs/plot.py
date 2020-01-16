import pandas as pd
import numpy as np

import scipy.stats as sts
from bokeh.transform import transform
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.nonparametric.api as smnp

from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
from bokeh import palettes


COLORS = ('blue', 'red', 'black', ' aqua', ' green', ' orange', ' yellow', 'purple', ' magenta')


def create_figure(
        title: str,
        x_axis_label: str = 't',
        y_axis_label: str = 'y',
        plot_width: int = 800,
        plot_height: int = 600,
        **kwargs
) -> figure:
    """
    Create bokeh figure with params
    """
    return figure(title=title, x_axis_label=x_axis_label,
                  y_axis_label=y_axis_label, plot_width=plot_width,
                  plot_height=plot_height, **kwargs)


def plot_line(
        y: np.ndarray,
        x: np.ndarray = None,
        points: bool = True,
        title: str = 'y',
        x_axis_label: str = 'Index',
        y_axis_label: str = 'Value',
        plot_width: int = 1000,
        plot_height: int = 500,
        color: str = 'blue',
        legend: str = 'Line',
        show_graph: bool = True
) -> figure:
    """
    Plot simple line where y by y axis and x by x axis. If
    x is not existed x will be equal range(0, len(y))
    """
    if x is None:
        x = np.arange(len(y))

    fig = figure(title=f'{title}', x_axis_label=x_axis_label,
                 y_axis_label=y_axis_label, plot_width=plot_width,
                 plot_height=plot_height)

    fig.line(x=x, y=y, color=color, legend_label=str(legend))

    if points:
        fig.circle(x=x, y=y, fill_color=color)

    if show_graph:
        show(fig)

    return fig


def plot_lines(
        y: tuple,
        x: np.ndarray = None,
        points: bool = True,
        x_axis_label: str = 'Index',
        y_axis_label: str = 'Value',
        plot_width: int = 1000,
        plot_height: int = 500,
        color: tuple = None,
        legend: tuple = None,
        title: str = 'Graph lines',
        show_graph: bool = True
) -> figure:
    """
    Plot lines from y tuple. Number of lines equal len(y)
    with plot params
    """
    if x is None:
        x = np.arange(len(y[0]))

    if legend is None:
        legend = [f'Line {i}' for i in range(len(y))]

    if color is None:
        color = COLORS

    fig = figure(title=title, x_axis_label=x_axis_label,
                        y_axis_label=y_axis_label, plot_width=plot_width,
                        plot_height=plot_height)

    for i in range(len(y)):
        fig.line(
            y=y[i],
            x=x,
            color=color[i],
            legend=legend[i]
        )

        if points is not None:
            fig.circle(
                y=y[i],
                x=x,
                fill_color=color[i]
            )

    if show_graph:
        show(fig)

    return fig


def acf_plot(
        y: np.ndarray,
        lags: int = 40,
        title: str = 'y',
        alpha: float = 0.05,
        plot_width: int = 1000,
        plot_height: int = 500,
        show_graph: bool = True
) -> figure:
    """
    PLot ACF of y
    """
    if any(i is None for i in y):
        y = y[y != None].astype(np.float64)

    if len(y) <= lags:
        raise ValueError('Length of series must be greater then num lags pacf')

    a_corr, conf = acf(y, nlags=lags, alpha=alpha, fft=True)
    x = np.arange(lags)
    conf_up = conf[1:, 0] - a_corr[1:]
    conf_low = conf[1:, 1] - a_corr[1:]

    fig = figure(title=f'ACF of {title}', x_axis_label='Lag',
                        y_axis_label='Correlation', plot_width=plot_width,
                        plot_height=plot_height)

    fig.vbar(x=x, top=a_corr[1:], width=None, color='blue')
    fig.circle(x=x, y=a_corr[1:], fill_color='black', size=8)
    fig.line(x=x, y=conf_up, line_dash='4 4', color='red', line_width=1)
    fig.line(x=x, y=conf_low, line_dash='4 4', color='red')

    if show_graph:
        show(fig)

    return fig


def pacf_plot(
        y: np.ndarray,
        lags: int = 40,
        title: str = 'y',
        alpha: float = 0.05,
        plot_width: int = 1000,
        plot_height: int = 500,
        show_graph: bool = True
) -> figure:
    """
    Plot PACF of y
    """
    if any(i is None for i in y):
        y = y[y != None].astype(np.float64)

    if len(y) <= lags:
        raise ValueError('Length of series must be greater then num lags pacf')

    p_corr, conf = pacf(y, nlags=lags, method='yw', alpha=alpha)
    x = np.arange(lags)
    conf_up = conf[1:, 0] - p_corr[1:]
    conf_low = conf[1:, 1] - p_corr[1:]

    fig = figure(title=f'PACF of {title}', x_axis_label='Lag',
                        y_axis_label='Correlation', plot_width=plot_width,
                        plot_height=plot_height)

    fig.vbar(x=x, top=p_corr[1:], width=None, color='blue')
    fig.circle(x=x, y=p_corr[1:], fill_color='black', size=8)
    fig.line(x=x, y=conf_up, line_dash='4 4', color='red', line_width=1)
    fig.line(x=x, y=conf_low, line_dash='4 4', color='red')

    if show_graph:
        show(fig)

    return fig


def plot_hist(
        y: np.ndarray,
        bins: int = 40,
        title: str = 'y',
        width: float = None,
        plot_width: int = 1000,
        plot_height: int = 500,
        x_axis_label: str = 'Bins',
        y_axis_label: str = 'P',
        show_graph: bool = True
) -> figure:
    """
    PLot histogram of y
    """
    if any(i is None for i in y):
        y = y[y != None].astype(np.float64)

    hist, edges = np.histogram(y, density=True, bins=bins)

    if width is None:
        width = np.diff(edges).mean() / 1.5

    fig = figure(title=f'Histogram of {title}', x_axis_label=x_axis_label,
                        y_axis_label=y_axis_label, plot_width=plot_width,
                        plot_height=plot_height)

    fig.vbar(x=edges[:-1], top=hist, color='blue', width=width)

    if show_graph:
        show(fig)

    return fig


def plot_qq(
        y: np.ndarray,
        title: str = 'y',
        dist: str = 'norm',
        plot_width: int = 1000,
        plot_height: int = 500,
        show_graph: bool = True
) -> figure:
    """
    Plot quantile-quantile graph of y
    """
    if any(i is None for i in y):
        y = y[y != None].astype(np.float64)

    probabilities = sts.probplot(y.astype(np.float64), dist=dist)
    reg_line = probabilities[1][1] + probabilities[1][0] * probabilities[0][0]

    fig = figure(title=f'QQ of {title}',
                        x_axis_label=f'Theoretical quantiles of {dist.upper()} distribution',
                        y_axis_label='Sample quantiles',
                        plot_width=plot_width,
                        plot_height=plot_height)

    fig.circle(x=probabilities[0][0], y=probabilities[0][1], fill_color='blue', color='blue')
    fig.line(x=probabilities[0][0], y=reg_line, color='red')

    if show_graph:
        show(fig)

    return fig


def min_max_scale_rolling(y: np.ndarray, window_size: int = 20) -> np.ndarray:
    scale = []

    for i in range(window_size, len(y)):
        scale.append((y[i] - np.min(y[:i + 1])) / (np.max(y[:i + 1]) - np.min(y[:i + 1])))

    return np.array(scale)


def _statsmodels_univariate_kde(
        data: np.ndarray,
        kernel: str = 'gau',
        bw: str = 'scott',
        cumulative: bool = False
) -> tuple:
    """Compute a univariate kernel density estimate using statsmodels."""
    fft = kernel == "gau"
    kde = smnp.KDEUnivariate([data])
    kde.fit(kernel=kernel, bw=bw, fft=fft)

    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density

    return grid, y


def kde_plot(
        y: np.ndarray,
        fit: str = None,
        title: str = 'y',
        plot_width: int = 1000,
        plot_height: int = 500,
        show_graph: bool = True,
        kernel: str = 'gau',
        bw: str = 'scott',
        cumulative: bool = False
) -> figure:
    """
    Plot kernel density function of y.
    """
    if any(i is None for i in y):
        y = y[y != None].astype(np.float64)

    y = y.astype(np.float64)

    support, density = _statsmodels_univariate_kde(y, kernel, bw, cumulative)

    if fit is not None:  # TODO
        dens_func = getattr(sts, fit)

        def pdf(x):
            return dens_func.pdf(x, *params)

        params = dens_func.fit(y)
        density = pdf(y)

    fig = plot_line(y=density, x=support, points=False, title=f'KDE plot of {title}',
                    x_axis_label='Bins', y_axis_label='density',
                    plot_width=plot_width, plot_height=plot_height,
                    color='blue', show_graph=show_graph, legend='KDE')

    if show_graph:
        show(fig)

    return fig


def box_plot():
    pass


def heatmap(
        corr: pd.DataFrame,
        title: str = 'Heatmap',
        width: int = 800,
        height: int = 800,
        show_graph: bool = False
) -> figure:
    """
    Create and plot heatmap of y
    """
    reshape_corr = pd.DataFrame(corr.stack(), columns=["correlation"]).reset_index()
    source = ColumnDataSource(reshape_corr)
    mapper = LinearColorMapper(palette=palettes.Viridis256, low=-1.0, high=1.0)

    tooltips = [("correlation", "@correlation{0.3f}")]
    corr_columns = corr.columns.tolist()

    fig = figure(title=title, tooltips=tooltips, toolbar_location="right",
                 x_range=corr_columns, y_range=corr_columns,
                 plot_width=width, plot_height=height)

    fig.rect(x="level_0", y="level_1", width=1, height=1, source=source,
            line_color="white", line_alpha=0.8, fill_color=transform("correlation", mapper))

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=10))

    fig.add_layout(color_bar, 'right')
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = "10pt"
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = 1

    if show_graph:
        show(fig)

    return fig


def hbar(
        y: np.ndarray,
        columns: list = None,
        title: str = 'y',
        plot_width: int = 700,
        plot_height: int = 800,
        show_graph: bool = True,
        height: float = 1.0,
        left: float = 0.0,
        **_
):

    if columns is None:
        columns = list(np.arange(len(y)))

    data = pd.DataFrame({'features': columns, 'values': y.ravel()})
    source = ColumnDataSource(data)

    fig = figure(y_range=columns, plot_height=plot_height, plot_width=plot_width,
                 title=title, toolbar_location='right', tooltips=[('Value', '@values{0.3f}')])
    fig.hbar(y='features', left=left, height=height, source=source,
             right='values', line_color='white', fill_color='blue')

    if show_graph:
        show(fig)

    return fig


def univariate_summary_plot(
        y: np.ndarray,
        title: str = 'y'
):
    _acf = acf_plot(y, show_graph=False, title=title)
    _pacf = pacf_plot(y, show_graph=False, title=title)
    _histo = plot_hist(y, show_graph=False, title=title)
    _line = plot_line(y, show_graph=False, title=title)
    _qq = plot_qq(y, show_graph=False, title=title)
    _kde = kde_plot(y, show_graph=False, title=title)

    r1 = row(_line, _histo, _kde)
    r2 = row(_acf, _pacf, _qq)
    c = column(r1, r2)

    show(c)


if __name__ == '__main__':
    pass
