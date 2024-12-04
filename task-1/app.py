import streamlit as st
import model

st.header("Лабораторная работа 1")

gender = st.sidebar.selectbox('Пол', ['Мужской', 'Женский'])
age = st.sidebar.number_input('Возраст', 10, 99)

st.title("Предсказание")
happy = model.predict(gender, age)
st.write(f'Коффициент счастья: {round(happy * 100)}%')