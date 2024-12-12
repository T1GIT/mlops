import json

import numpy as np
from sklearn.linear_model import LogisticRegression

with open("task-4/weights.json", "r") as f:
    loaded_data = json.load(f)

lr = LogisticRegression()
lr.coef_ = np.array(loaded_data["coef"])
lr.intercept_ = np.array(loaded_data["intercept"])
lr.classes_ = np.array(loaded_data["classes"])


def predict(attrs):
    return lr.predict(np.array(attrs).reshape(1, -1)).ravel()[0]
