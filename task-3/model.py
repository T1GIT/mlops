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

def predict(attrs):
    return rf.predict(np.array(attrs).reshape(1, -1)).ravel()[0]
