import json
import pickle

import pandas as pd
import streamlit as st
from transformers import LogTransformer

def load_feature_info(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


categorical_info = load_feature_info("task-5/categorical_feature_info.json")
numerical_info = load_feature_info("task-5/numerical_feature_info.json")

with open("task-5/pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

st.header("Лабораторная работа 5")

st.subheader("Категориальные признаки")
categorical_inputs = {}
for feature, values in categorical_info.items():
    categorical_inputs[feature] = st.sidebar.selectbox(
        label=feature,
        options=values
    )

st.subheader("Числовые признаки")
numerical_inputs = {}
for feature, info in numerical_info.items():
    numerical_inputs[feature] = st.sidebar.slider(
        label=feature,
        min_value=info["min"],
        max_value=info["max"],
        value=info["avg"],
        step=1
    )

if st.button("Сделать предсказание"):
    data = {**categorical_inputs, **numerical_inputs}
    prediction = pipe.predict(pd.DataFrame([data]))[0]

    st.header("Предсказание")
    sign = ">" if prediction else "<"
    st.write(f"Доход {sign} 50 тысяч долларов")
