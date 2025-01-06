import requests
import os
import streamlit as st

API_URL = f'http://{os.environ["API_URL"]}'

@st.cache_data
def get_feature_info():
    response = requests.get(f"{API_URL}/features")
    response.raise_for_status()
    return response.json()

feature_info = get_feature_info()
categorical_info = feature_info["categorical_info"]
numerical_info = feature_info["numerical_info"]

st.header("Лабораторная работа 6")

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

    response = requests.post(f"{API_URL}/predict", json=data)
    response.raise_for_status()
    prediction = response.json()["prediction"]

    st.header("Предсказание")
    sign = ">" if prediction else "<"
    st.write(f"Доход {sign} 50 тысяч долларов")
