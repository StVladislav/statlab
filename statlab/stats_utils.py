import datetime

import numpy as np
import pandas as pd
import scipy.stats as sts
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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


def sort_dictionary_by_value(dictionary: dict):
    return dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True))


def sample_data(x, sample_size=30):
    uniq = set(x)
    indeses = []

    for i in uniq:
        indeses.extend(np.random.choice(list(np.where(x == i)[0]), int(sample_size / len(uniq))))

    np.random.shuffle(indeses)

    return indeses


def dataframe_col_values_by_func(df, to_replace, function: callable) -> pd.DataFrame:
    data = np.array(df)
    cols = df.columns

    for col in range(data.shape[1]):
        data_col = data[:, col]
        value = function(data_col[data_col != to_replace].astype(float))
        data_col[data_col == to_replace] = value

    return pd.DataFrame(data, columns=cols)


def dataframe_label_encoder(dataframe: pd.DataFrame, sparse: bool = False) -> tuple:
    cols = dataframe.columns
    encoder_function = LabelEncoder()
    encoded_data = np.array(dataframe)

    for i in range(encoded_data.shape[1]):
        encoder_function.fit(encoded_data[:, i])
        encoded_data[:, i] = encoder_function.transform(encoded_data[:, i])

    return pd.DataFrame(encoded_data, columns=cols), encoder_function


def dataframe_onehot_encoder(dataframe: pd.DataFrame, sparse: bool = False) -> tuple:
    cols = dataframe.columns
    encoder_function = OneHotEncoder(sparse=sparse)
    encoder = encoder_function.fit(dataframe)
    encoder_cols = []
    encoder_data = encoder.transform(dataframe)

    for i in range(len(encoder.categories_)):
        for j in encoder.categories_[i]:
            encoder_cols.append(cols[i] + '.' + j)

    return pd.DataFrame(encoder_data, columns=encoder_cols), encoder


def dataframe_drop_by_row_value(dataframe, check, return_index: bool = False):
    cols = dataframe.columns
    data = np.ravel(dataframe.values)
    index = []

    for i in range(len(data)):
        if check != data[i]:
            index.append(i)

    to_ret = pd.DataFrame(data[index], columns=cols)

    if return_index:
        return to_ret, index

    return to_ret


def dataframe_columns_by_type(dataframe, cols_type, value_to_change='empty'):
    index = []

    for col in cols_type.keys():
        train = np.array(dataframe[col].values)

        for i in range(len(train)):
            if not isinstance(train[i], cols_type[col]):
                index.append(i)

        if index:
            dataframe[col].iloc[index] = value_to_change

    return dataframe


def dataframe_column_not_str(dataframe: pd.DataFrame) -> tuple:
    """
    Separate dataframe to dataframes with columns wich are contained string values and
    are not contained string values

    Parameters
    ----------
    dataframe - dataframe like pandas

    Returns
    -------
    tuple:
    0 index - dataframe without columns it contained string values
    1 index - dataframe with columns it contained string values
    """
    data = np.array(dataframe)
    cols = dataframe.columns
    cols_not_str = []
    cols_str = []

    for i in range(data.shape[1]):
        for j in data[:, i]:
            if not isinstance(j, str):
                cols_not_str.append(cols[i])
                break

    for i in cols:
        if i not in cols_not_str:
            cols_str.append(i)

    data_not_str = dataframe[cols_not_str]
    data_str = dataframe[cols_str]

    return data_not_str, data_str


def dataframe_row_index_by_type(dataframe, _type=str):
    rows = []
    data = np.array(dataframe)

    for i in range(data.shape[0]):
        for value in data[i, :]:
            if isinstance(value, _type):
                rows.append(i)
                break

    return rows


def dataframe_replacer(dataframe, value, to) -> pd.DataFrame:
    """
    Replace values in dataframte to required values

    Parameters
    ----------
    dataframe - dataframe like pandas
    value - what will be replaced
    to - value to be replaced

    Returns
    -------
    pandas.DataFrame with replaced values
    """
    cols = dataframe.columns
    data = np.array(dataframe)

    if not isinstance(value, list):
        value = [value]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] in value:
                data[i, j] = to

    return pd.DataFrame(data, columns=cols)


def dataframe_nan_replacer(dataframe, to) -> pd.DataFrame:
    """
    Replace np.nan values in dataframe
    Parameters
    ----------
    dataframe - dataframe like pandas
    to - value to be replaced nan

    Returns
    -------
    pandas.DataFrame with replaced np.nan values
    """
    cols = dataframe.columns
    data = np.array(dataframe)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if isinstance(data[i, j], float) or isinstance(data[i, j], int):
                if np.isnan(data[i, j]):
                    data[i, j] = to

    return pd.DataFrame(data, columns=cols)


def dataframe_replacer_by_type(dataframe, _type, to) -> pd.DataFrame:
    cols = dataframe.columns
    data = np.array(dataframe)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            if isinstance(data[i, j], _type):
                data[i, j] = to

    return pd.DataFrame(data, columns=cols)


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


class NumpyDataFrame:
    """
    This class simplify interaction with pandas.DataFrame
    and increases speed as it used numpy.
    Instance of this class may sliced by rows/columns. As well
    instance may set new data by indexing. Both ordinal numbers
    and their names (from pandas) can be used as column indexes.
    Also it may replaced nan values by value or function.

    EXAMPLE:
        >>> df = pd.DataFrame(np.random.normal(size=(10,4)), columns=['a', 'b'. 'c', 'd'])
        >>> new = NumpyDataFrame(df)
        >>> new[1:3, 0:3]
        >>> new[[1,2,3,4,5], 'a']
        >>> new[[1,2,3], ['a', 'b', 'c']]
        >>> new[1:5, ['a', 'b', 'c']]
        >>> new[1:8, 'a']
        >>> new[1:3, 'd'] = 0.5

    """
    def __init__(self, df: pd.DataFrame):

        if not isinstance(df, pd.DataFrame):
            raise TypeError

        self.columns = df.columns
        self.data = np.array(df)
        self._columns = {i: j for i, j in zip(self.columns, np.arange(df.shape[1]))}

    @property
    def get_nans_value(self) -> dict:
        """
        Get all indeses it containes Nans value.
        Nans value are defined in mask below.
        :return: dict with key - ordinal number of column; value - list of indeses rows
        """
        nan_mask = ('None', 'NaT', 'nan', 'Nan', 'NaN', 'NULL')
        nans_value_index = dict()

        for col in range(self.data.shape[1]):
            nans_in_row = []
            for row in range(self.data.shape[0]):
                if str(self.data[row, col]) in nan_mask:
                    nans_in_row.append(row)
            if nans_in_row:
                nans_value_index[col] = nans_in_row

        return nans_value_index

    def replace_nan(self, replaced_by):
        """
        Replace nan value
        replaced_by: value of callable object which replaced Nans values
        """
        nans_value_index = self.get_nans_value

        if not nans_value_index:
            return

        for col, rows in nans_value_index.items():
            if callable(replaced_by):
                self.data[rows, col] = replaced_by(
                    [self.data[i, col] for i in range(self.data.shape[0]) if i not in rows])
            else:
                self.data[rows, col] = replaced_by

    @property
    def get_data_frame(self) -> pd.DataFrame:
        """
        Return padnas.DataFrame
        """
        return pd.DataFrame(self.data, columns=self.columns)

    def get_by_key(self, key) -> tuple:
        """
        This method used __getitem__ and __setitem__
        """
        if isinstance(key, int) or isinstance(key, str):
            new = self.data[:, key if isinstance(key, int) else self._columns[key]]
        else:
            key = list(key)
            if isinstance(key[1], str):
                key[1] = self._columns[key[1]]
            if isinstance(key[1], list):
                key[1] = [self._columns[i] if isinstance(i, str) else i for i in key[1]]
            if isinstance(key[0], list) and isinstance(key[1], list):
                key[0], key[1] = np.ix_(key[0], key[1])
            new = self.data[key[0], key[1]]

        return new, key

    def __getitem__(self, key):
        return self.get_by_key(key)[0]

    def __setitem__(self, key, data):
        _, key = self.get_by_key(key)
        self.data[tuple(key)] = data

    def __repr__(self):
        return str(pd.DataFrame(self.data, columns=self.columns))


if __name__ == '__main__':
    pass
