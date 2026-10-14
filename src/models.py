import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import SEED
from src.features import FEATURE_GROUPS, all_features, envelope_spectrum, order_spectrum


def _forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


class FeatureModel:
    """Random forest on a group of hand-crafted features."""

    def __init__(self, name, groups):
        self.name, self.groups = name, groups

    def _select(self, X, rpm):
        feats = all_features(X, rpm)
        cols = [c for g in self.groups for c in FEATURE_GROUPS[g](feats.columns)]
        return feats[cols]

    def fit(self, X, rpm, y):
        self.model = _forest().fit(self._select(X, rpm), y)
        return self

    def predict(self, X, rpm):
        return self.model.predict(self._select(X, rpm))


class OrderSpectrumModel:
    """Data-driven: logistic regression on the full envelope order spectrum."""
    name = "Envelope order spectrum + logistic regression"

    @staticmethod
    def _spectrum(X, rpm):
        return np.log1p(1e3 * order_spectrum(*envelope_spectrum(X), rpm)[1])

    def fit(self, X, rpm, y):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000,
                                                                        class_weight="balanced"))
        self.model.fit(self._spectrum(X, rpm), y)
        return self

    def predict(self, X, rpm):
        return self.model.predict(self._spectrum(X, rpm))


def model_zoo():
    return [
        FeatureModel("Time-domain statistics", ["time"]),
        FeatureModel("Envelope fault frequencies (physics)", ["envelope"]),
        FeatureModel("All hand-crafted features", ["time", "spectral", "envelope"]),
        OrderSpectrumModel(),
    ]
