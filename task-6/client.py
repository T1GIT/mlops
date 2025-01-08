import requests
import os
import streamlit as st

API_URL = f'http://{os.environ["API_URL"]}'

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = False

@st.cache_data
def get_meta():
    response = requests.get(f"{API_URL}/meta")
    response.raise_for_status()
    return response.json()

@st.cache_data
def get_versions():
    response = requests.get(f"{API_URL}/versions")
    response.raise_for_status()
    return response.json()

meta = get_meta()
versions = get_versions()

st.header("Лабораторная работа 6")

st.subheader("Категориальные признаки")
categorical_inputs = {}
for feature, values in meta["categorical"].items():
    categorical_inputs[feature] = st.sidebar.selectbox(
        label=feature,
        options=values
    )

st.subheader("Числовые признаки")
numerical_inputs = {}
for feature, info in meta["numerical"].items():
    numerical_inputs[feature] = st.sidebar.slider(
        label=feature,
        min_value=info["min"],
        max_value=info["max"],
        value=info["avg"],
        step=1
    )

st.session_state.version = st.selectbox(label="Версия", options=versions)

if st.button("Сделать предсказание", type="primary"):
    st.session_state.input = {**categorical_inputs, **numerical_inputs}

    try:
        response = requests.post(f"{API_URL}/predict", json=st.session_state.input, params={"version": st.session_state.version})
        response.raise_for_status()
        st.session_state.prediction = response.json()["prediction"]
        st.session_state.id = response.json()["id"]
        st.success("Ответ получен")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при запросе к API: {e}")
        st.session_state.prediction = None
    st.session_state.feedback_sent = False

if st.session_state.prediction is not None:
    sign = ">" if st.session_state.prediction else "<"
    st.metric(label="Ответ", value=f"Доход {sign} 50 тысяч долларов")
    st.divider()

    with st.container(border=True):
        st.subheader("Предложить правильный ответ")
        corrected = st.selectbox(label="Правильный ответ", options=[0, 1], format_func=lambda i: {0: "< 50", 1: "> 50"}[i])
        feedback = {
            "predicted": st.session_state.prediction,
            "corrected": corrected,
            "input": st.session_state.input,
            "id": st.session_state.id
        }

        def disable_send():
            st.session_state.feedback_sent = True

        if st.button("Отправить", disabled=st.session_state.feedback_sent, on_click=disable_send):
            try:
                requests.post(f"{API_URL}/feedback", json=feedback, params={"version": st.session_state.version})
                st.session_state.feedback_sent = True
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при отправке фидбека: {e}")

if st.session_state.feedback_sent:
    st.success("Ответ отправлен")
