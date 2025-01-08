import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r"./task-2/data/income.csv")
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

models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1.0),
    'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10),
    'LogisticRegression': LogisticRegression(),
    'XGBoost': xgb.XGBClassifier()
}
for name, model in models.items():
    model.fit(X_train, Y_train)

results = []
for name, model in models.items():
    Y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(Y_test, Y_pred),
        'Precision': precision_score(Y_test, Y_pred),
        'Recall': recall_score(Y_test, Y_pred),
        'F1': f1_score(Y_test, Y_pred),
    })

compare_df = pd.DataFrame(results)
