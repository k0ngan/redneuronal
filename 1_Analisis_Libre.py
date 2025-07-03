import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis Exploratorio", layout="wide")
st.title("🔍 Análisis exploratorio de datos")

csv = st.file_uploader("Sube un CSV compatible con StudentsPerformance", type="csv")
if csv:
    df = pd.read_csv(csv)
    st.subheader("Vista previa")
    st.dataframe(df.head())

    st.write("Registros:", len(df))
    st.write("Columnas:", list(df.columns))

    if 'math score' in df.columns:
        st.subheader("Distribución de puntajes de matemáticas")
        fig, ax = plt.subplots()
        ax.hist(df['math score'], bins=20)
        st.pyplot(fig)
