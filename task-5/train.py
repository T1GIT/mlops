import json
import pickle

import pandas as pd
from category_encoders import BinaryEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transformers import LogTransformer

df = pd.read_csv(r"task-5/data/income.csv")

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

with open("task-5/categorical_feature_info.json", "w") as f:
    json.dump(categorical_feature_info, f)

with open("task-5/numerical_feature_info.json", "w") as f:
    json.dump(numerical_feature_info, f)


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
    ('model', RandomForestClassifier(n_estimators=50, max_depth=10))
])

pipe.fit(X_train, Y_train)

with open("task-5/pipe.pkl", "wb") as f:
    pickle.dump(pipe, f)
