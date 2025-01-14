import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = X[col].apply(lambda x: np.nan if x <= 0 else x)
            X[col] = np.log(X[col])
            X[col] = X[col].fillna(0)
        return X