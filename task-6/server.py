import json
import os
import pickle
from typing import Union, Dict

import uvicorn
import uuid
import logging

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, create_model, RootModel

from transformers import LogTransformer


def setup_logger(name, level=logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs("task-6/logs", exist_ok=True)
    file_handler = logging.FileHandler(f'task-6/logs/{name}.log')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name.upper())
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


predict_logger = setup_logger('predict')
feedback_logger = setup_logger('feedback')


class FeaturesSchema(RootModel[Dict[str, Union[str, int]]]):
    pass


class FeedbackSchema(BaseModel):
    predicted: int
    corrected: int
    input: FeaturesSchema
    id: str



def get_meta():
    with open("task-6/meta.json", "r") as f:
        return json.load(f)


def get_versions():
    names = os.listdir("task-6/models")
    versions = list(map(lambda name: name[:-4], names))
    return sorted(versions)


def predict(features: FeaturesSchema, version: str):
    df = pd.DataFrame([features.model_dump()])

    path = f"task-6/models/{version}.pkl"
    with open(path, "rb") as f:
        pipe = pickle.load(f)

    prediction = pipe.predict(df).tolist()[0]

    id = str(uuid.uuid4())
    predict_logger.info(f"Predicted value: {prediction} for features: {features}. Prediction id: {id}. Model version: {version}")

    return {"prediction": prediction, "id": id}


def feedback(payload: FeedbackSchema, version: str):
    feedback_logger.info(
        f"Predicted value: {payload.predicted} for features: {payload.input}. Corrected value: {payload.corrected}. Prediction id: {id}. Model version: {version}")


app = FastAPI()


@app.get("/meta")
def req_meta():
    return get_meta()


@app.get("/versions")
def req_versions():
    return get_versions()

@app.post("/predict")
def req_predict(payload: FeaturesSchema, req: Request):
    version = req.query_params.get("version", "latest")
    return predict(payload, version)


@app.post("/feedback")
def req_feedback(payload: FeedbackSchema, req: Request):
    version = req.query_params.get("version", "latest")
    return feedback(payload, version)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
