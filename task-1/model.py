import logging
import os
import random
import time


def setup_logger(name, level=logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs("task-1/logs", exist_ok=True)
    file_handler = logging.FileHandler(f'task-1/logs/{name}.log')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name.upper())
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


predict_logger = setup_logger('predict')


def predict(gender, age):
    time.sleep(2)
    prediction = random.uniform(0, 1)

    data = {"gender": gender, "age": age}
    predict_logger.info(f'Input: {data}. Prediction: {prediction}')

    return prediction
