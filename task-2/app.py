import streamlit as st
import model

st.header("Лабораторная работа 2")

st.title("Датасет")
st.dataframe(
    model.raw_df,
    hide_index=True
)

st.title("Cравнение")
st.dataframe(
    model.compare_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1']),
    hide_index=True,
)
