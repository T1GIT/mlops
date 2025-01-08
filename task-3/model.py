import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./task-3/data/income.csv")
raw_df = df.copy(deep=True)

df.drop(
    ['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country', 'relationship'],
    axis=1,
    inplace=True
)
df.fillna(df.median(), inplace=True)

X = df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
Y = df['income >50K']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=50, max_depth=10)
rf.fit(X_train, Y_train)

def setup_logger(name, level=logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs("task-3/logs", exist_ok=True)
    file_handler = logging.FileHandler(f'task-3/logs/{name}.log')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name.upper())
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


predict_logger = setup_logger('predict')

def predict(attrs):
    prediction = rf.predict(np.array(attrs).reshape(1, -1)).ravel()[0]

    predict_logger.info(f'Input: {attrs}. Prediction: {prediction}')

    return prediction
