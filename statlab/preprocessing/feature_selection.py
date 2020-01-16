import warnings
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats as sts

import statsmodels.api as sm
from minepy import MINE
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from errors import ShapeError, IncorrectIndex
from bokeh.layouts import gridplot
from graphs.plot import heatmap, hbar, show
from stats_utils import array_drop_nan, array_fill_nan


def cosine(
        y: np.ndarray,
        x: np.ndarray = None,
        only_endog: bool = True,
        endog_index: int = 0,
        prepare_data: bool = True,
        **_
) -> np.ndarray:

    """Calculates cosine similarity
    Returns cosine similarity between y and x if x is defined,
    else if x is not defined y shape has to be greater than 1 (y~matrix)

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation of y and x by columns if x exist. Else creates numpy array
    from y. y has to be a matrix of pandas DataFrame

    endog_index int: what column will be endog variable
    only_endog bool: if True return similarity by endog variable. In this case
    endog_variable defined by parameter endog_index


    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 3)
        >>> print(cosine(y=y)) # matrix (1, 3)
        >>> print(cosine(y=y, endog_index=1)) # matrix (1, 3)
        >>> print(cosine(y=y[:, 0], x=y[:, 1:3] , endog_index=1)) # matrix (1, 3)
        >>> print(cosine(y, only_endog=False)) # matrix (3, 3)

    Return
    _____________
    Cosine similarity numpy array
    """
    if not isinstance(endog_index, int):
        raise TypeError("Endog index has to be int type in cosine")

    y = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > y.shape[-1] or endog_index == y.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in cosine")

    quadratic = np.transpose(y) @ y
    sqrt_sum = np.sum(y * y, axis=0, keepdims=True) ** .5
    similarities = quadratic / sqrt_sum / np.transpose(sqrt_sum)
    distance = np.round(similarities, 3)

    return distance[endog_index, :].reshape(1, -1) if only_endog else distance


def spearman(
        y: np.ndarray,
        x: np.ndarray = None,
        only_endog: bool = True,
        endog_index: int = 0,
        stat_significant: bool = False,
        alpha: float = 0.1,
        prepare_data: bool = True,
        **_
) -> np.ndarray:
    """Calculates Spearman rank correlation
    Return Spearman rank correlation between y and x if x is defined,
    else if x is not defined y shape has to be greater than 1 (y~matrix)

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation of y and x by columns if x exist. Else create numpy array
    from y. y has to be a matrix of pandas DataFrame or other matrix dtype

    stat_significant bool: if True return H1 hypothesis by given alpha
    significance level p-values.
    alpha bool: probability of rejection H0 hypothesis.
    endog_index int: what column will be endog variable
    only_endog bool: if True return similarity by endog variable. In this case
    endog_variable is defined by parameter endog_index


    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 3)
        >>> print(spearman(y=y)) # matrix (1, 3)
        >>> print(spearman(y=y, endog_index=1)) # matrix (1, 3)
        >>> print(spearman(y=y[:, 0], x=y[:, 1:3] , endog_index=1)) # matrix (1, 3)
        >>> print(spearman(y, only_endog=False)) # matrix (3, 3)

    Return
    _____________
    Spearman rank correlation numpy array
    """
    if not isinstance(endog_index, int):
        raise TypeError("Endog index has to be int type in spearman")

    assert 0 < alpha < 1, 'Alpha must be in interval (0,1) in spearman'

    y = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > y.shape[-1] or endog_index == y.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in spearman")

    correlation = np.empty(shape=(y.shape[-1], y.shape[-1]))
    num_index = 0 if not stat_significant else 1

    for i, j in product(np.arange(correlation.shape[-1]), np.arange(correlation.shape[-1])):
        correlation[i, j] = sts.spearmanr(y[:, i], y[:, j])[num_index]

    if stat_significant:
        correlation = correlation < alpha

    return correlation[endog_index, :].reshape(1, -1) if only_endog else correlation


def pearson(
        y: np.ndarray,
        x: np.ndarray = None,
        only_endog: bool = True,
        endog_index: int = 0,
        stat_significant: bool = False,
        alpha: float = 0.1,
        prepare_data: bool = True, **_
) -> np.ndarray:
    """Calculate Pearson correlation
    Return Pearson correlation between y and x if x is defined,
    else if x is not defined y shape has to be greater than 1 (y~matrix)

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtype float32
    from concatenation of y and x by columns if x exist. Else create numpy array
    from y. y has to be pandas DataFrame or other matrix dtype.

    stat_significant bool: if True return H1 hypothesis. by given by given alpha
    significance level p-values
    alpha bool: probability of rejection of H0 hypothesis.
    endog_index int: what column will be endog hypothesis
    only_endog bool: if True return similarity by endog variable. In this case
    endog_variable is defined by parameter endog_index


    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 3)
        >>> print(pearson(y=y)) # matrix (1, 3)
        >>> print(pearson(y=y, endog_index=1)) # matrix (1, 3)
        >>> print(pearson(y=y[:, 0], x=y[:, 1:3] , endog_index=1)) # matrix (1, 3)
        >>> print(pearson(y, only_endog=False)) # matrix (3, 3)

    Return
    _____________
    Pearson correlation numpy array
    """
    if not isinstance(endog_index, int):
        raise TypeError("Endog index has to be int type in pearson")

    assert 0 < alpha < 1, 'Alpha must be in interval (0,1) in pearson'

    y = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > y.shape[-1] or endog_index == y.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in spearman")

    correlation = np.empty(shape=(y.shape[-1], y.shape[-1]))
    num_index = 0 if not stat_significant else 1

    for i, j in product(np.arange(correlation.shape[-1]), np.arange(correlation.shape[-1])):
        correlation[i, j] = sts.pearsonr(y[:, i], y[:, j])[num_index]

    if stat_significant:
        correlation = correlation < alpha

    return correlation[endog_index, :].reshape(1, -1) if only_endog else correlation


def kendall(
        y: np.ndarray,
        x: np.ndarray = None,
        only_endog: bool = True,
        endog_index: int = 0,
        stat_significant: bool = False,
        alpha: float = 0.1,
        prepare_data: bool = True,
        **_
) -> np.ndarray:
    """Calculate Kendall rank correlation
    Return Kendall rank correlation between y and x if x is defined,
    else if x is not defined y shape has to be greater than 1 (y~matrix)

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation of y and x by columns if x exist. Else create numpy array
    from y. y has to be pandas DataFrame or other matrix-like dtype

    stat_significant bool: if True return H1 hypothesis by given alpha
    significance level p-values
    alpha bool: probability of rejection of H0 hypothesis.
    endog_index int: what column will be endog
    only_endog bool: if True return similarity by endog variable. In this case
    endog_variable is defined by parameter endog_index


    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 3)
        >>> print(kendall(y=y)) # matrix (1, 3)
        >>> print(kendall(y=y, endog_index=1)) # matrix (1, 3)
        >>> print(kendall(y=y[:, 0], x=y[:, 1:3] , endog_index=1)) # matrix (1, 3)
        >>> print(kendall(y, only_endog=False)) # matrix (3, 3)

    Return
    _____________
    Kendall rank correlation numpy array
    """
    if not isinstance(endog_index, int):
        raise TypeError("Endog index has to be int type in kendall")

    assert 0 < alpha < 1, 'Alpha must be in interval (0,1) in kendall'

    y = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > y.shape[-1] or endog_index == y.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in kendall")

    correlation = np.empty(shape=(y.shape[-1], y.shape[-1]))
    num_index = 0 if not stat_significant else 1

    for i, j in product(np.arange(correlation.shape[-1]), np.arange(correlation.shape[-1])):
        correlation[i, j] = sts.kendalltau(y[:, i], y[:, j])[num_index]

    if stat_significant:
        correlation = correlation < alpha

    return correlation[endog_index, :].reshape(1, -1) if only_endog else correlation


def mine(
        y: np.ndarray,
        x=None,
        only_endog: bool = True,
        endog_index: int = 0,
        prepare_data: bool = True,
        options=None,
        **_
) -> np.ndarray:
    """Calculate Maximal Information Coefficient
    Returns the Maximal Information Coefficient between y and x if x is defined,
    else if x is not defined y shape has to be greater than 1 (y~matrix)

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation of y and x by columns if x exist. Else create numpy array
    from y. y has to be like a matrix of pandas DataFrame

    endog_index int: what column will be endog
    only_endog bool: if True return similarity by endog variable. In this case
    endog_variable defined by parameter endog_index
    options dict: settings for MINE

    Note
    ____________
    For more information of MINE method see
    https://minepy.readthedocs.io/en/latest/

    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 3)
        >>> print(cosine(y=y)) # matrix (1, 3)
        >>> print(cosine(y=y, endog_index=1)) # matrix (1, 3)
        >>> print(cosine(y=y[:, 0], x=y[:, 1:3] , endog_index=1)) # matrix (1, 3)
        >>> print(cosine(y, only_endog=False)) # matrix (3, 3)

    Return
    _____________
    Maximal Information Coefficient numpy array
    """
    if options is None:
        options = {'alpha': 0.6, 'c': 15, 'est': 'mic_approx'}

    if not isinstance(endog_index, int):
        raise TypeError("Endog index has to be int type in kendall")

    y = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > y.shape[-1] or endog_index == y.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in kendall")

    correlation = np.empty(shape=(y.shape[-1], y.shape[-1]))
    mine = MINE(**options)
    for i, j in product(np.arange(correlation.shape[-1]), np.arange(correlation.shape[-1])):
        mine.compute_score(y[:, i], y[:, j])
        correlation[i, j] = mine.mic()

    return correlation[endog_index, :].reshape(1, -1) if only_endog else correlation


def importance_forest(
        y: np.ndarray,
        x: np.ndarray = None,
        type_model: str = 'regression',
        endog_index: int = 0,
        options: dict = None,
        prepare_data: bool = True,
        **_
) -> np.ndarray:
    """Calculate feature importance by ExtraTrees model
    Return feature importance by defined endog variables of matrix (y, x)
    if x is defined. Else if x is not defined y shape has to be greater than 1 (y~matrix)
    and y contains all data.

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation of y and x by columns if x exist. Else create numpy array
    from y. y has to be like a matrix of pandas DataFrame or other matrix-like dtype

    endog_index int: what column will be endog
    type_model str: classifier or regression
    options dict (default None): parameters of ExtraTrees model

    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 2)
        >>> print(importance_forest(y=y)) # matrix (1, 2)
        >>> print(importance_forest(y=y, endog_index=1), type_model='classifier') # matrix (1, 2)

    Return
    _____________
    ExtraTrees feature importance numpy array
    """
    if options is None:
        options = {'n_estimators': 10}

    data = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > data.shape[-1] or endog_index == data.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in importance_forest")

    index = np.arange(data.shape[-1])
    y, x = data[:, endog_index], data[:, index[index != endog_index]]

    engine = {
        'regression': ExtraTreesRegressor,
        'classifier': ExtraTreesClassifier
    }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = engine[type_model](**options)
        model.fit(X=x, y=y)

    return model.feature_importances_.reshape(1, -1)


def importance_catboost(
        y: np.ndarray,
        x: np.ndarray = None,
        type_model: str = 'regression',
        endog_index: int = 0,
        options: dict = None,
        prepare_data: bool = True,
        **_
) -> np.ndarray:
    """Calculate feature importance using Catboost model
    Return feature importance by defined endog variables of matrix (y, x)
    if x is defined. Else if x is not defined y shape has to be greater than 1 (y~matrix)
    and y contains all data.

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation y and x by columns if x exist. Else create numpy array
    from y. y has to be like a matrix of pandas DataFrame

    endog_index int: what column will be endog variables
    type_model str: classifier or regression
    options dict (default None): parameters of ExtraTrees model

    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 2)
        >>> print(importance_catboost(y=y)) # matrix (1, 2)
        >>> print(importance_catboost(y=y, endog_index=1), type_model='classifier') # matrix (1, 2)

    Return
    _____________
    Catboost feature importance numpy array
    """
    if options is None:
        options = {'verbose': 0, 'iterations': 1000}

    data = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > data.shape[-1] or endog_index == data.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in importance_catboost")

    index = np.arange(data.shape[-1])
    y, x = data[:, endog_index], data[:, index[index != endog_index]]

    engine = {
        'regression': CatBoostRegressor,
        'classifier': CatBoostClassifier
    }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = engine[type_model](**options)
        model.fit(X=x, y=y)

    return model.feature_importances_.reshape(1, -1)


def importance_ols(
        y: np.ndarray,
        x: np.ndarray = None,
        endog_index: int = 0,
        prepare_data: bool = True,
        **_
) -> np.ndarray:
    """Calculate p-values of coefs of LinearRegression (OLS method) model
    Return values by defined endog variables of matrix (y, x)
    if x is defined. Else if x is not defined y shape has to be greater than 1 (y~matrix)
    and y proposeed all data.

    Parameters
    _____________
    prepare_data bool: create numpy array (matrix) with dtypes float32
    from concatenation of y and x by columns if x exist. Else create numpy array
    from y. y has to be like a matrix of pandas DataFrame

    endog_index int: what column will be endog

    Example:
        >>> y = sts.norm(0, 1).rvs((100, 3)) # matrix (100, 3)
        >>> print(importance_ols(y=y)) # matrix (1, 2)
        >>> print(importance_ols(y=y[:,0], x=y[:,1:], endog_index=2) # matrix (1, 2)

    Return
    _____________
    OLS p-values of feature numpy array
    """
    data = DataForFeatureSelection(y=y, x=x).prepare_data() if prepare_data else y

    if endog_index > data.shape[-1] or endog_index == data.shape[-1]:
        raise ShapeError("Endog index has to be lower num columns in importance_ols")

    index = np.arange(data.shape[-1])
    y, x = data[:, endog_index], data[:, index[index != endog_index]]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()

    return model.pvalues[1:].reshape(1, -1)


def chi_test():
    pass


class DataForFeatureSelection:

    """
    This class is used inside FeatureSelection class in order
    to stack y and x

    Parameters
    ----------
    y : array_like or pandas DataFrame/Series without nan-values.
    x: array_like or pandas DataFrame/Series without nan-values.

    """
    def __init__(self, *, y: np.ndarray or pd.DataFrame, x: np.ndarray or pd.DataFrame):

        if x is not None:
            assert y.shape[0] == x.shape[0], "Length of variables has to be equal in FeatureSelector"

        self.y = np.float32(y) if isinstance(y, np.ndarray) else np.array(y, dtype=np.float32)
        self.x = np.float32(x) if isinstance(x, np.ndarray) else np.array(x, dtype=np.float32)

    def prepare_data(self):

        if not np.isnan(self.x).all():
            self.y = np.column_stack((self.y, self.x))

        return self.y


class FeatureSelector:

    """
    This class allows to measure the influence of
    variables on other variables given in parameters.

    Parameters
    ----------
    y : array_like or pandas DataFrame/Series with or without nan-values.
    x: array_like or pandas DataFrame/Series with or without nan-values.
    columns: list containing names of variables if x and y are np.ndarray

    Note
    ----------
    # TODO about DEFAULT_PARAMS
    Examples
    --------
    >>> train = sts.norm(loc=0, scale=1).rvs((100, 3))
    >>> feature_selected = FeatureSelector(train)
    >>> feature_selected.summary(fill_na=np.median, endog_index=1)
    """

    DEFAULT_PARAMS = {
        'fill_na': np.mean,
        'dropna': None,
        'endog_index': 0,
        'to_dataframe': True,
        'show_graph': False,
        '_get_graph': False
    }

    def __init__(
            self,
            y: np.ndarray or pd.DataFrame,
            x: np.ndarray or pd.DataFrame = None,
            columns: list = None
    ):
        self.data = DataForFeatureSelection(y=y, x=x).prepare_data()
        self.columns = self._create_columns(columns, y, x)

    def _update_default_params(self, **kwargs):

        params = self.DEFAULT_PARAMS.copy()

        if kwargs:
            params.update(**kwargs)

        return params

    def cosine(
            self,
            sort_correlation: bool = True,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)
        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        similar = cosine(y=y, **params)
        y_label = self._get_endog_label(similar, endog_index)
        similar = self.make_sort_correlation(
            data=pd.DataFrame(similar, index=[y_label], columns=columns),
            sort_correlation=sort_correlation
        )
        label_to_plot = 'Data' if isinstance(y_label, list) else y_label

        fig = self.make_plot(similar, title=f'Cosine importance of {label_to_plot}', **params)
        similar = self._to_datafeame(similar, y_label, params["to_dataframe"])

        if params['_get_graph']:
            return fig

        return similar

    def mine(
            self,
            sort_correlation: bool = True,
            **kwargs
    ):
        params = self._update_default_params(**kwargs)
        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        similar = mine(y=y, **params)
        y_label = self._get_endog_label(similar, endog_index)
        similar = self.make_sort_correlation(
            data=pd.DataFrame(similar, index=[y_label], columns=columns),
            sort_correlation=sort_correlation
        )
        label_to_plot = 'Data' if isinstance(y_label, list) else y_label

        fig = self.make_plot(similar, title=f'Mic importance of {label_to_plot}', **params)
        similar = self._to_datafeame(similar, y_label, params["to_dataframe"])

        if params['_get_graph']:
            return fig

        return similar

    def spearman(
            self,
            sort_correlation: bool = True,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)

        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        similar = spearman(y=y, **params)
        y_label = self._get_endog_label(similar, endog_index)
        label_to_plot = 'Data' if isinstance(y_label, list) else y_label
        similar = self.make_sort_correlation(
            data=pd.DataFrame(similar, index=[y_label], columns=columns),
            sort_correlation=sort_correlation
        )

        fig = self.make_plot(similar, title=f'Spearman importance of {label_to_plot}', **params)
        similar = self._to_datafeame(similar, y_label, params["to_dataframe"])

        if params['_get_graph']:
            return fig

        return similar

    def kendall(
            self,
            sort_correlation: bool = True,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)

        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        similar = kendall(y=y, **params)
        y_label = self._get_endog_label(similar, endog_index)
        label_to_plot = 'Data' if isinstance(y_label, list) else y_label
        similar = self.make_sort_correlation(
            data=pd.DataFrame(similar, index=[y_label], columns=columns),
            sort_correlation=sort_correlation
        )

        fig = self.make_plot(similar, title=f'Kendall importance of {label_to_plot}', **params)
        similar = self._to_datafeame(similar, y_label, params['to_dataframe'])

        if params['_get_graph']:
            return fig

        return similar

    def pearson(
            self,
            sort_correlation: bool = True,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)

        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        similar = pearson(y=y, **params)
        y_label = self._get_endog_label(similar, endog_index)

        label_to_plot = 'Data' if isinstance(y_label, list) else y_label
        similar = self.make_sort_correlation(
            data=pd.DataFrame(similar, index=[y_label], columns=columns),
            sort_correlation=sort_correlation
        )

        fig = self.make_plot(similar, title=f'Pearson importance of {label_to_plot}', **params)
        similar = self._to_datafeame(similar, y_label, params["to_dataframe"])

        if params['_get_graph']:
            return fig

        return similar

    def importance_forest(
            self,
            type_model: str = 'regression',
            options_model: dict = None,
            sort_correlation: bool = True,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)

        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        feature_importance = importance_forest(
            y=y,
            type_model=type_model,
            options=options_model,
            **params
        )

        if params['to_dataframe']:
            feature_importance = self.make_sort_correlation(
                data=pd.DataFrame(
                    feature_importance,
                    index=[y_label],
                    columns=columns[:endog_index] + columns[endog_index+1:]
                ),
                sort_correlation=sort_correlation,
                drop=False
            )
        else:
            feature_importance = (
                feature_importance,
                y_label,
                columns[:endog_index] + columns[endog_index+1:]
            )

        fig = self.make_plot(feature_importance, title=f'ExtraTrees importance of {y_label}', **params)
        feature_importance = self._to_datafeame(feature_importance, y_label, params["to_dataframe"])

        if params['_get_graph']:
            return fig

        return feature_importance

    def importance_catboost(
            self,
            type_model: str = 'regression',
            options_model: dict = None,
            sort_correlation: bool = True,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)

        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        feature_importance = importance_catboost(
            y=y,
            type_model=type_model,
            options=options_model,
            **params
        )

        feature_importance /= 100

        if params['to_dataframe']:
            feature_importance = self.make_sort_correlation(
                data=pd.DataFrame(
                    feature_importance,
                    index=[y_label],
                    columns=columns[:endog_index] + columns[endog_index+1:]
                ),
                sort_correlation=sort_correlation,
                drop=False
            )
        else:
            feature_importance = (
                feature_importance,
                y_label,
                columns[:endog_index] + columns[endog_index+1:]
            )

        fig = self.make_plot(feature_importance, title=f'Catboost importance of {y_label}', **params)
        feature_importance = self._to_datafeame(feature_importance, y_label, params['to_dataframe'])

        if params['_get_graph']:
            return fig

        return feature_importance

    def importance_ols(
            self,
            type_model: str = 'regression',
            options_model: dict = None,
            **kwargs
    ):

        params = self._update_default_params(**kwargs)

        y, endog_index, y_label, columns = self._get_data_label_columns(
            fill_na=params["fill_na"],
            dropna=params["dropna"],
            endog_index=params['endog_index']
        )
        params.update(endog_index=endog_index)

        feature_importance = importance_ols(
            y=y,
            type_model=type_model,
            options=options_model,
            **params
        )

        if params['to_dataframe']:
            feature_importance = pd.DataFrame(
                feature_importance,
                index=[y_label],
                columns=columns[:endog_index] + columns[endog_index+1:]
            )
        else:
            feature_importance = (
                feature_importance,
                y_label,
                columns[:endog_index] + columns[endog_index+1:]
            )

        fig = self.make_plot(feature_importance, title=f'OLS importance of {y_label}', **params)
        feature_importance = self._to_datafeame(feature_importance, y_label, params['to_dataframe'])

        if params['_get_graph']:
            return fig

        return feature_importance

    def chi_test(self):
        pass

    @staticmethod
    def make_sort_correlation(
            data: pd.DataFrame,
            sort_correlation: bool = True,
            drop: bool = True
    ):

        if data.shape[0] != 1:
            return data

        if sort_correlation:
            data = data.drop(data.index, axis=1).sort_values(by=data.index[0], axis=1) if drop\
                else data.sort_values(by=data.index[0], axis=1)

        return data

    def summary(self, metrics: list = None, get_metrics: bool = False, **kwargs):

        _get_graph = get_metrics is False
        figs = []

        if metrics is None:
            metrics = self._get_metrics
        metrics = dict.fromkeys(metrics)

        for i in metrics.copy():
            metrics[i] = getattr(self, i)(_get_graph=_get_graph, only_endog=True, **kwargs)
            if i in ('cosine', 'mine', 'spearman', 'pearson', 'kendall'):
                metrics[i+'_full'] = getattr(self, i)(_get_graph=_get_graph, only_endog=False, **kwargs)
                figs.append(metrics[i+'_full'])
            figs.append(metrics[i])

        if _get_graph:
            grid = gridplot(figs, ncols=2)
            self._show_graph(grid)

        return metrics if not _get_graph else None

    @staticmethod
    def _show_graph(graph):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            show(graph)

    def _create_columns(self, columns, y, x):

        if columns is not None:
            columns = columns
        elif isinstance(x, pd.DataFrame):
            columns = x.columns.to_list()
        elif isinstance(y, pd.DataFrame):
            columns = y.columns.to_list()
        else:
            columns = list(np.arange(self.data.shape[-1]).astype(str))

        return columns

    def _get_endog_label(self, data, endog_index: int):

        if data.shape[0] > 1:
            y_label = self.columns
        else:
            y_label = self.columns[endog_index]

        return y_label

    def _to_datafeame(self, data, y_label: str or list, to_dataframe: bool = True):

        if not to_dataframe:
            data = (data, y_label, self.columns[1:])

        return data

    def _get_data_label_columns(
            self,
            fill_na=None,
            dropna: int = None,
            endog_index: str or int = 0
    ):
        y = self.data
        columns = self.columns

        if isinstance(endog_index, str):
            try:
                endog_index = self.columns.index(endog_index)
            except ValueError:
                raise IncorrectIndex("FeatureSelector - Endog index has to be lower num columns if int"
                                     "or contain in columns data")

        y_label = self.columns[endog_index]



        if dropna is not None:
            y = array_drop_nan(y, axis=dropna)

        if fill_na is not None:
            y = array_fill_nan(y, fill_na)

        return y, endog_index, y_label, columns

    @property
    def _get_metrics(self):
        metrics = ['cosine', 'spearman', 'kendall',
                   'pearson', 'mine', 'importance_forest',
                   'importance_catboost', 'importance_ols']

        return metrics

    def make_plot(
            self,
            data: pd.DataFrame,
            title: str,
            show_graph: bool = False,
            plot_width: int = 800,
            **_
    ):

        if data.shape[0] > 1:
            fig = heatmap(
                corr=data,
                title=title,
                show_graph=show_graph,
                width=plot_width,
                height=plot_width
            )
        elif data.shape[0] == 1:
            fig = hbar(
                y=np.array(data),
                columns=data.columns.to_list(),
                title=title,
                show_graph=show_graph,
                plot_width=np.int(plot_width * 0.8),
                plot_height=plot_width
            )
        else:
            raise IndexError('FeatureSelection - Incorrect shape for plotting')

        return fig


if __name__ == '__main__':
    url = 'http://bit.ly/kaggletrain'
    df = pd.read_csv(url)

    train = df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    train = train.merge(pd.get_dummies(df['Sex']), how='inner', on=df['PassengerId']).drop('key_0', axis=1)
    train = train.merge(pd.get_dummies(df['Embarked']), how='inner', on=df['PassengerId']).drop('key_0', axis=1)

    feature_selected = FeatureSelector(train)
    feature_selected.summary(endog_index='Survived')
