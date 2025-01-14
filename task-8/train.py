import json
import os
import pickle
import subprocess

import mlflow
import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder, OneHotEncoder
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transformers import LogTransformer

df = pd.read_csv(r"data/income.csv")

binary_features = ['workclass', 'marital-status', 'occupation']
onehot_features = ['education', 'race', 'sex', 'native-country', 'relationship']
log_features = ['capital-gain', 'capital-loss']
numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

df.fillna(df[numerical_features].median(), inplace=True)
df.fillna(df[onehot_features + binary_features].mode().iloc[0], inplace=True)

params_list = [
    {'n_estimators': 50, 'max_depth': 5, },
    {'n_estimators': 50, 'max_depth': 7, },
    {'n_estimators': 50, 'max_depth': 10, },
    {'n_estimators': 100, 'max_depth': 12, },
    {'n_estimators': 100, 'max_depth': 15, },
]

for params in params_list:
    with mlflow.start_run() as run:
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
            ('model', RandomForestClassifier(**params))
        ])

        pipe.fit(X_train, Y_train)

        y_pred = pipe.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics({
            'Accuracy': accuracy_score(Y_test, y_pred),
            'Precision': precision_score(Y_test, y_pred),
            'Recall': recall_score(Y_test, y_pred),
            'F1': f1_score(Y_test, y_pred),
        })

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="income-model",
            signature=signature,
            registered_model_name="sk-learn-random-forest-income-model",
        )
