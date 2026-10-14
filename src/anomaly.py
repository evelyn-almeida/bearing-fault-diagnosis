import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

SCALE_FEATURES = ["rms", "peak", "peak_to_peak", "std"]


class HealthIndex:
    """Mahalanobis distance to the healthy state, learned from normal data only.

    The alarm threshold is calibrated on operating conditions held out from the fit,
    so it anticipates the shift that a new motor load produces on healthy signals.
    """

    def __init__(self, quantile=0.99):
        self.quantile = quantile

    @staticmethod
    def _prepare(features):
        f = features.copy()
        f[SCALE_FEATURES] = np.log(f[SCALE_FEATURES])
        return f

    def _fit_distance(self, features):
        f = self._prepare(features)
        scaler = StandardScaler().fit(f)
        cov = LedoitWolf().fit(scaler.transform(f))
        return lambda z: np.log(cov.mahalanobis(scaler.transform(self._prepare(z))))

    def fit(self, normal_features, conditions):
        conditions = np.asarray(conditions)
        held_out = [self._fit_distance(normal_features[conditions != c])(normal_features[conditions == c])
                    for c in np.unique(conditions)]
        self.threshold = np.quantile(np.concatenate(held_out), self.quantile)
        self.distance = self._fit_distance(normal_features)
        return self

    def score(self, features):
        return self.distance(features)
