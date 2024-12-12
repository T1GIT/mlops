import streamlit as st

import model

st.header("Лабораторная работа 5")


age = st.sidebar.number_input("Возраст", 10, 50, 20)
edu_num = st.sidebar.number_input("Лет образования", 0, 30, 4)
cap_gain = st.sidebar.number_input("Прирост капитала", 0, 100_000, 100)
cap_loss = st.sidebar.number_input("Убыток капитала", 0, 100_000, 100)
work_hours = st.sidebar.number_input("Часов работы в неделю", 0, 80, 40)

st.header("Предсказание")
result = model.predict([age, edu_num, cap_gain, cap_loss, work_hours])
sign = ">" if result else "<"
st.write(f"Доход {sign} 50 тысяч долларов")
