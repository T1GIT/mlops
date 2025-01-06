import json
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model

from transformers import LogTransformer


def load_feature_info(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


categorical_info = load_feature_info("task-6/categorical_feature_info.json")
numerical_info = load_feature_info("task-6/numerical_feature_info.json")

with open("task-6/pipe.pkl", "rb") as f:
    pipe = pickle.load(f)


def predict(data):
    df = pd.DataFrame([data.model_dump()])
    prediction = pipe.predict(df)
    return prediction.tolist()[0]


FeaturesSchema = create_model("FeaturesSchema", **{
    **{key: (str, None) for key in categorical_info.keys()},
    **{key: (int, None) for key in numerical_info.keys()}
})

app = FastAPI()


@app.get("/features")
def req_features():
    return {
        "categorical_info": categorical_info,
        "numerical_info": numerical_info
    }


@app.post("/predict")
def req_predict(payload: FeaturesSchema):
    prediction = predict(payload)
    return {
        "prediction": prediction
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
