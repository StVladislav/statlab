import numpy as np
import tabulate

from distributions import ContinousDistributionEstimator, DiscreteDistributionEstimator


class LossDistributionApproach:

    __slots__ = ['losses', 'events', 'continuous_distribution', 'discrete_distribution', 'alpha', 'samples',
                 'lda_risk', 'extreme_duration', 'asset_name', 'negative']

    _assets = []

    def __init__(
            self,
            losses: np.ndarray,
            events: np.ndarray,
            dist_continuous: str or tuple = ('gumbel_r', 'genhalflogistic', 'genextreme', 'gamma'),
            dist_discrete: str or tuple = ('poisson', 'planck', 'dlaplace'),
            alpha: float = 0.05,
            negative: bool = False,
            samples: int = 10000,
            asset_name: str = 'Name'
    ):

        assert 0 <= alpha <= 1, 'Incorrect alpha in LossDistributionApproach. It must be in (0, 1)'
        assert losses.dtype != np.object, 'dtype values of losses in LossDistributionApproach is np.object'
        assert events.dtype != np.object, 'dtype values of events in LossDistributionApproach is np.object'

        self.losses = losses if not negative else losses * -1
        self.events = events

        self.continuous_distribution = ContinousDistributionEstimator(dist_continuous)
        self.discrete_distribution = DiscreteDistributionEstimator(dist_discrete)

        self.alpha = alpha if not negative else (1 - alpha)
        self.samples = samples
        self.lda_risk = None
        self.extreme_duration = None

        self.asset_name = asset_name
        self.negative = negative
        self._assets.append(self)

    @property
    def _dist_fit(self) -> tuple:
        discrete = self.discrete_distribution.fit(self.events)
        continuous = self.continuous_distribution.fit(self.losses)

        return discrete, continuous

    @property
    def _generate_events_losses(self) -> tuple:

        if self.discrete_distribution.function is None:
            self.fit()

        events = self.discrete_distribution.function.rvs(self.samples)
        losses = np.array([
            sum(self.continuous_distribution.function.rvs(i)) for i in events
        ])

        return events, losses

    def _calculate_market_risk_lda(self, losses: np.ndarray):
        return np.percentile(losses, self.alpha * 100)

    @property
    def _calculate_duration(self):
        alpha = self.alpha if not self.negative else 1 - self.alpha
        return self.discrete_distribution.function.ppf(1 - alpha)

    def fit(self):
        _ = self._dist_fit
        events, losses = self._generate_events_losses
        self.lda_risk = self._calculate_market_risk_lda(losses)

        if self.negative:
            self.lda_risk = np.negative(self.lda_risk)

        self.extreme_duration = self._calculate_duration

        return self

    @property
    def mean_risk_extreme_events(self) -> float:

        if self.discrete_distribution.function is None:
            self.fit()

        return self.discrete_distribution.function.ppf(1 - self.alpha) * self.continuous_distribution.function.mean()

    @property
    def extreme_risk_mean_events(self) -> float:

        if self.discrete_distribution.function is None:
            self.fit()

        return self.discrete_distribution.function.mean() * self.continuous_distribution.function.ppf(self.alpha)

    @property
    def _summary(self) -> list:

        if self.discrete_distribution.function is None:
            self.fit()

        return [('Asset is', f'{self.asset_name.upper()}'),
                ('Lda risk is', f'{self.lda_risk}'),
                ('Extreme duration', f'{self.extreme_duration}'),
                ('Mean risk extreme events is', f'{self.mean_risk_extreme_events}'),
                ('Extreme risk mean events is', f'{self.extreme_risk_mean_events}'),
                ('Distribution of events is', f'{self.discrete_distribution.dist}'),
                ('Distribution of losses is', f'{self.continuous_distribution.dist}')]

    @classmethod
    def get_lda_assets(cls):
        return cls._assets

    @classmethod
    def print_lda_assets(cls):
        for asset in cls._assets:
            asset()

    def __call__(self, *args, **kwargs):
        print(tabulate.tabulate(self._summary))

    def __str__(self):
        return f'LDA of {self.asset_name} is {self.lda_risk}\nDiscrete dist is {self.discrete_distribution.dist}'


if __name__ == '__main__':
    losses = np.random.normal(loc=0, scale=1, size=500)
    losses = losses[losses < 0]
    events = np.random.poisson(5, size=100)
    lda_1 = LossDistributionApproach(losses=losses, events=events, asset_name='Asset 1', negative=True).fit()
    print(lda_1)
    losses = np.random.normal(loc=0, scale=5, size=500)
    losses = losses[losses < 0]
    events = np.random.poisson(5, size=100)

    lda_2 = LossDistributionApproach(losses=losses, events=events, asset_name='Asset 2', negative=True).fit()
    LossDistributionApproach.print_lda_assets()
