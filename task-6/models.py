import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ZeroClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        num_samples = X.shape[0]
        return np.hstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))

    def score(self, X, y):
         return (y == 0).mean()