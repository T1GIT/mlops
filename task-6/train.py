import json
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transformers import LogTransformer
from models import ZeroClassifier

df = pd.read_csv(r"task-6/data/income.csv")

binary_features = ['workclass', 'marital-status', 'occupation']
onehot_features = ['education', 'race', 'sex', 'native-country', 'relationship']
log_features = ['capital-gain', 'capital-loss']
numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


df.fillna(df[numerical_features].median(), inplace=True)
df.fillna(df[onehot_features + binary_features].mode().iloc[0], inplace=True)


categorical_feature_info = {}
for col in binary_features + onehot_features:
    categorical_feature_info[col] = list(df[col].unique())

numerical_feature_info = {}
for col in numerical_features:
    numerical_feature_info[col] = {
        "min": int(df[col].min()),
        "max": int(df[col].max()),
        "avg": int(df[col].mean())
    }

with open("task-6/meta.json", "w") as f:
    json.dump({"categorical": categorical_feature_info, "numerical": numerical_feature_info}, f)


X = df.drop('income >50K', axis=1)
Y = df['income >50K']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



pipe = Pipeline([
    ('transformer', ColumnTransformer(transformers=[
        ('binary', BinaryEncoder(), binary_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_features),
        ('log', LogTransformer(), log_features),
        ('scaler', StandardScaler(), numerical_features)
    ], remainder='passthrough')),
    # ('model', RandomForestClassifier(n_estimators=50, max_depth=10))
    # ('model', AdaBoostClassifier(n_estimators=50))
    ('model', ZeroClassifier())
])

pipe.fit(X_train, Y_train)


result = subprocess.run(['poetry', 'version'], capture_output=True, text=True, check=True)
version = result.stdout.strip().split(' ')[1]
tags = [f'v{version}', 'latest']

os.makedirs("task-6/models", exist_ok=True)
for tag in tags:
    with open(f"task-6/models/{tag}.pkl", "wb") as f:
        pickle.dump(pipe, f)
