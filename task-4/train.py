import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./task-4/data/income.csv")
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

lr = LogisticRegression(max_iter=100_000)
lr.fit(X, Y)

model_data = {
    "model_type": "LogisticRegression",
    "coef": lr.coef_.tolist(),
    "intercept": lr.intercept_.tolist(),
    "classes": lr.classes_.tolist(),
}

with open("task-4/weights.json", "w") as f:
    json.dump(model_data, f, indent=4)
