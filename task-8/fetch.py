import mlflow

model_name = "sk-learn-random-forest-income-model"
model_version = "3"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

data = [
    {
        "workclass": "State-gov",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "education": "Bachelors",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "relationship": "Not-in-family",
        "age": 38,
        "education-num": 10,
        "capital-gain": 1077,
        "capital-loss": 87,
        "hours-per-week": 40
    },
    {
        "workclass": "State-gov",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "education": "Bachelors",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "relationship": "Not-in-family",
        "age": 60,
        "education-num": 30,
        "capital-gain": 107700,
        "capital-loss": 0,
        "hours-per-week": 60
    },
]

print(list(map(lambda x: model.predict(x)[0], data)))
