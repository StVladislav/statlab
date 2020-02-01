from itertools import product

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tabulate import tabulate

from preprocessing.timeseries import lagged_matrix


class ArimaModel:
    """This is an auxiliary class which contains
    methods that are used to calculate basic values
    and components for ARIMA-type models

    Parameters
    ----------
    ma: int - number of lags for residuals.
    ar: int - number of lags for autoregressive model.

    Attributes
    ----------
    self.k: int - number of estimated parameters of ARIMA model including intercept
    self.y: array_like - timeseries array for model
    self.fitted_values: array_like - fitted values by models after estimation
    self.et: array_like - residual series
    self.ma_matrix: array_like - matrix which contains lags of reiduals
    self.ar_matrix: array_like - matrix which contains lags of endog variable
    self.rss: float - sum of square residuals
    self.ess: float - sum of square deviations fitted values from mean endog variable
    self.tss: float - sum of square deviations endog variable from it's mean
    self.r_square: float - coefficient of determination like ess / tss or (1 - rss / tss)
    self.loglike: float - value of negative loglikelihood function
    self.sic: float - Schwarz (Bayesian) information criterion
    self.aic: float - Akaike information criterion

    Notes
    ----------
    __slots__ is used to reduce memory usage

    """
    __slots__ = ['y', 'ma', 'ar', 'k', 'fitted_values',
                 'et', 'ma_matrix', 'ar_matrix', 'rss',
                 'ess', 'tss', 'std', 'r_square', 'loglike', 'sic', 'aic']

    def __init__(self, ar: int, ma: int):

        if ar <= 0:
            raise ValueError('ar in ArimaModel must be greater than 0')

        if ma < 0:
            raise ValueError('ma in ArimaModel must be greater than or equal to 0')

        self.ma = ma
        self.ar = ar
        self.k = ar + ma + 1

        self.y = None
        self.fitted_values = None
        self.et = None
        self.ma_matrix = None
        self.ar_matrix = None

        self.rss = None
        self.ess = None
        self.tss = None
        self.std = None

        self.r_square = None
        self.loglike = None
        self.sic = None
        self.aic = None

    def create_ar_matrix(self, y) -> np.ndarray:
        return lagged_matrix(y, fill=0, endog=False, lagg=self.ar)

    def create_ma_matrix(self, et) -> np.ndarray:
        return lagged_matrix(et, fill=0, endog=False, lagg=self.ma)

    @staticmethod
    def calculate_et(y, y_hat):
        return y - y_hat

    @property
    def calculate_rss(self) -> np.float:
        return tf.reduce_sum(tf.square(self.y - self.fitted_values)).numpy()

    @property
    def calculate_tss(self) -> np.float:
        return tf.reduce_sum(tf.square(self.y - tf.reduce_mean(self.y))).numpy()

    @property
    def calculate_ess(self) -> np.float:
        return tf.reduce_sum(tf.square(self.fitted_values - tf.reduce_mean(self.y))).numpy()

    @property
    def calculate_r_score(self) -> np.float:
        return self.ess / self.tss

    @property
    def calculate_aic(self) -> np.float:
        try:
            aic = 2 * self.k + len(self.y) * np.log(self.rss)
        except RuntimeWarning:
            print('Loglike is negative or equal to 0. '
                  'Schwarz information criteria has not been calculated for ArimaModel')
            aic = None

        return aic

    @property
    def calculate_sic(self) -> np.float:
        try:
            sic = 2 * np.log(self.y.shape[0]) * self.k + 2 * np.log(self.loglike)
        except RuntimeWarning:
            print('Loglike is negative or equal to 0. '
                  'Schwarz information criteria has not been calculated for ArimaModel')
            sic = None

        return sic

    @property
    def calculate_std(self):
        return np.sqrt(self.rss)


class SimpleArima(ArimaModel):
    """Realization of ARIMA(p,0, q) which allows to predict time series
    with integration of order zero (stationary at levels).
    This class allows to optimize coefficients of a model with the use of
    method of maximum likelihood on the basis of normal distribution
    function and Student distribution. Parameters p,q reflect maximum
    number of lags included in the model. Parameter p - stands for
    number of lags of endogenous variable, q - number of lags in residuals.

    Parameters
    ----------
    ma: int - number of lags for residuals.
    ar: int - number of lags for autoregressive model.
    loss: str - type of loss (at present mse, loglike normal, student are accessible)
    learning_rate: int - learning rate for optimizer
    n_iter: int - maximum number of iterations for optimization. Must be in (1, inf)

    Attributes
    ----------
    self.bias : tf.Variable - bias parameter of arima model
    self.ar_coef: tf.Variable - coefficients for autoregressive part of arima model
    self.ma_coef: tf.Variable - coefficients for ma part of arima model
    self.dist_param: tf.Variable - it's shape param for Student distribution
    self.optimizer: tf.optimizer - tensorflow 2.0 optimizer

    Notes
    ----------
    __slots__ is used to reduce memory usage

    """
    __slots__ = ['n_iter', 'loss', 'bias', 'ar_coef', 'ma_coef', 'dist_param', 'optimizer', '__weakref__']
    avaliable_losses = ('loglike_norm', 'loglike_student', 'loss_mse')

    def __init__(
            self,
            ar: int = 1,
            ma: int = 1,
            loss: str = 'loglike_norm',
            learning_rate: int = 0.01,
            n_iter: int = 1000
    ):
        if loss not in self.avaliable_losses:
            raise ValueError('Incorrect loss function for SimpleArima. See avaliable_losses')

        if n_iter <= 0:
            raise ValueError('Incorrect n_ter in SimpleArima. It has to be greather than 1')

        super(SimpleArima, self).__init__(ar=ar, ma=ma)

        self.n_iter = n_iter
        self.loss = loss

        self.bias = tf.Variable(initial_value=tf.random.normal(shape=(1, 1)), trainable=True)
        self.ar_coef = tf.Variable(initial_value=tf.random.normal(shape=(self.ar, 1)), trainable=True)
        self.ma_coef = tf.Variable(initial_value=tf.random.normal(shape=(self.ma, 1)), trainable=True)
        self.dist_param = tf.Variable(initial_value=1.0, trainable=True)
        self.optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

    def get_loss(self, name: str) -> list:

        losses_coef = {
            'loss_mse': [self.bias, self.ar_coef, self.ma_coef],
            'loglike_norm': [self.bias, self.ar_coef, self.ma_coef],
            'loglike_student': [self.bias, self.ar_coef, self.ma_coef, self.dist_param]
        }

        return losses_coef[name]

    @property
    def calculate_ar(self) -> tf.Tensor:
        """Calculate autoregressive part of ARIMA model:

        y_hat = bo + b1 * yt-1 + b2 * yt-2 + ... + bi * yt-p

        Returns
        -------
        y_hat with shape = (n, 1); where n = len(y)
        """
        return self.bias + self.ar_matrix @ self.ar_coef

    def calculate_fitted_values(self) -> tf.Tensor:
        """Calculate fitted values of ARIMA model:

        y_hat =  bo + b1 * yt-1 + b2 * yt-2 + ... + bi * yt-p + g1 * et-1 + g2 * et-2 + ... + gj * et-q

        Returns
        -------
        y_hat with shape = (n, 1); where n = len(y)
        """
        ar_hat = self.calculate_ar
        et = self.calculate_et(self.y, ar_hat)
        self.change_ma_matrix(et)
        fitted_values = ar_hat + self.ma_matrix @ self.ma_coef
        self.et.assign(self.calculate_et(self.y, fitted_values))

        return fitted_values

    def _arima(self, y) -> tuple:
        """Calculate aotoregressive, et like matrix and fitted values
        of ARIMA model with self.ar_coef and self.ma_coef

        Parameters
        ----------
        y: numpy array or tensor with shape = (n, 1); where n is length of y

        Notes
        ----------
        It's a subsidiary method used for prediction in self.predict

        Returns
        -------
        autoregressive fitted values, matrix of residuals and fitted values
        """
        ar_mat = self.create_ar_matrix(y)
        ar_hat = self.bias + ar_mat @ self.ar_coef
        et = self.calculate_et(y, ar_hat)
        ma_matrix = self.create_ma_matrix(et)
        fit = ar_hat + ma_matrix @ self.ma_coef

        return ar_hat, ma_matrix, fit

    def predict(self, periods: int = 1) -> np.ndarray:
        """Predicts for n number of points

        Parameters
        ----------
        periods: int - length of prediction

        Returns
        -------
        Series of predicted values
        """
        pred = []
        y = np.array(self.y)
        y = np.append(y, 0).reshape(-1, 1)

        for _ in range(periods):
            _, _, p = self._arima(y)
            pred.append(p[-1])
            y[-1] = pred[-1]
            y = np.append(y, pred[-1]).reshape(-1, 1)

        return np.ravel(pred)

    def predict_sample(self, periods):
        """Creates stochastic time series with periods length,
        mean of which is explained by estimated ARIMA model.

        Parameters
        ----------
        periods: int - sample length

        Returns
        -------
        Stochastic time series, mean of which conforms with
        estimated ARIMA model and it's estimations.
        """
        sample_y = []
        pred = []

        y = np.array(self.y)
        y = np.append(y, 0).reshape(-1, 1)

        for _ in range(periods):
            _, _, p = self._arima(y)
            pred.append(p[-1])
            y[-1] = pred[-1] + np.random.normal(0, self.std)
            y = np.append(y, pred[-1] + np.random.normal(0, self.std)).reshape(-1, 1)
            sample_y.append(y[-1])

        return sample_y, pred

    def fit(self, y):
        """Estimates ARIMA coefficients and statistics of the model.
        Input values y - time series for which model is estimated

        Parameters
        ----------
        y - time series for which it is required to estimate ARIMA model

        Returns
        -------
        Value loss function
        """
        self.set_endog(y)
        self.optimize()

        return self

    @tf.function
    def loss_mse(self):
        """Calculates mean squared error of ARIMA model

        Returns
        -------
        Value of loss
        """
        fitted_values = self.calculate_fitted_values()
        loss = tf.reduce_mean(tf.square(self.y - fitted_values))

        return loss

    @tf.function
    def loglike_norm(self):
        """Calculates negative loglike function of ARIMA model.
        Loglike of normal distribution with mean = fitted values
        and std = 1/n sum(et**2)

        Returns
        -------
        Value of loglike
        """
        fitted_values = self.calculate_fitted_values()
        sigma = tf.sqrt(tf.reduce_mean(tf.square(self.y - fitted_values)))
        loglike = tfp.distributions.Normal(loc=fitted_values, scale=sigma)

        return tf.negative(tf.reduce_sum(loglike.log_prob(self.y)))

    @tf.function
    def loglike_student(self):
        """Calculates negative loglike function of ARIMA model.
        Loglike of Student distribution with mean = fitted values
        and std = 1/n sum(et**2) and optimized shape param

        Returns
        -------
        Value of loglike
        """
        fitted_values = self.calculate_fitted_values()
        sigma = tf.sqrt(tf.reduce_mean(tf.square(self.y - fitted_values)))
        loglike = tfp.distributions.StudentT(loc=fitted_values, scale=sigma, df=self.dist_param)

        return tf.negative(tf.reduce_sum(loglike.log_prob(self.y)))

    def change_ma_matrix(self, y):

        for i in range(self.ma_matrix.shape[1]):
            self.ma_matrix[i + 1:, i].assign(y[:-(i + 1), 0])

    def optimize(self):
        """Estimates ARIMA model coefficiants
        on the basis of given loss function.

        """
        func = getattr(self, self.loss)
        var_list = self.get_loss(self.loss)

        for _ in range(self.n_iter):
            self.optimizer.minimize(func, var_list=var_list)

        self.fitted_values = self.calculate_fitted_values()
        self.ess = self.calculate_ess
        self.tss = self.calculate_tss
        self.rss = self.calculate_rss
        self.r_square = self.calculate_r_score
        self.std = self.calculate_std

        if self.loss == 'loss_mse':
            self.loglike = self.loglike_norm()
        else:
            self.loglike = getattr(self, self.loss)()

        self.aic = self.calculate_aic
        self.sic = self.calculate_sic

    def summary(self):

        bias = [('bo', self.bias.numpy().ravel()[0])]
        ar_coeffs = [(f'ar_{i}', j[0]) for i,j in enumerate(self.ar_coef.numpy())]
        ma_coeffs = [(f'ma_{i}', j[0]) for i,j in enumerate(self.ma_coef.numpy())]
        header_coeffs = ['Coefficient', 'Value']
        header_stats = ['Statistics', 'Value']

        statistics = [
            ('rss', self.rss),
            ('tss', self.tss),
            ('ess', self.ess),
            ('r_square', self.r_square),
            ('mse loss', self.loss_mse().numpy()),
            ('loglike normal', self.loglike_norm().numpy()),
            ('loglike Student', self.loglike_student().numpy()),
            ('aic', self.aic),
            ('sic', self.sic)
        ]

        print(tabulate(bias + ar_coeffs + ma_coeffs, headers=header_coeffs))
        print(tabulate(statistics, headers=header_stats))

    def auto_arima(self, y, by: str = 'aic', max_lagg: int = 5):
        """Automatic selection of the best specification for the ARIMA model.
        The specification is currently being searched for based on
        Schwartz or Akayke information criteria.

        Parameters
        ----------
        y - time series for which it is required to estimate ARIMA model
        max_lagg: int - maximum value for number of lags p,q
        by: str - selects by which criterion to choose specification

        Returns
        -------
        SimpleArima object
        """
        if by not in ('aic', 'sic'):
            raise ValueError('Incorrect parameter for optimization specification of ARIMA model in SimpleArima')

        specification_laggs = product(np.arange(1, max_lagg + 1), repeat=2)
        current_quality = np.inf
        current_arima = None

        for i, j in specification_laggs:
            model = SimpleArima(ar=i, ma=j, loss=self.loss).fit(y)
            criteria = getattr(model, by)

            if criteria < current_quality:
                current_quality = criteria
                current_arima = model

        return current_arima

    def set_endog(self, y):
        """Defines an endogenous variable based on which the
        ARIMA model is optimized. It can be used to indicate a new variable.

        For example:
        we have estimated the model based on the y time series and made a prediction.
        Then a new value of y appeared and we can submit “new”
        series and without changing the coefficients make a forecast for t + m periods.
        """
        self.y = tf.constant(y, dtype=np.float32, shape=(len(y), 1))
        self.ar_matrix = tf.constant(self.create_ar_matrix(y), dtype=tf.float32)

        self.et = tf.Variable(tf.zeros((len(y), 1)))
        self.ma_matrix = tf.Variable(tf.zeros(shape=(len(y), self.ma)), dtype=np.float32)

    def __call__(self):
        return self.summary()


if __name__ == '__main__':
    y = tfp.distributions.Normal(loc=0, scale=1).sample(100).numpy()
    model = SimpleArima(ar=1, ma=1, loss='loglike_student').auto_arima(y, by='aic', max_lagg=3)
    model()
