import pickle

import numpy as np

with open("task-5/dump.pkl", "rb") as f:
    rf = pickle.load(f)

def predict(attrs):
    return rf.predict(np.array(attrs).reshape(1, -1)).ravel()[0]
