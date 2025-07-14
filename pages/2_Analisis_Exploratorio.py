import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Análisis Exploratorio", layout="wide")
st.title("🔍 Análisis exploratorio")

if "data_uploaded" not in st.session_state or not st.session_state["data_uploaded"]:
    st.warning("Por favor sube un archivo en la página principal primero.")
    st.stop()

df = st.session_state["df"]

st.subheader("Vista previa")
st.dataframe(df.head())

st.write(f"Registros: {len(df)}")
st.write(f"Columnas: {list(df.columns)}")

if 'math score' in df.columns:
    st.subheader("Distribución de puntajes de matemáticas")
    fig = px.histogram(df, x='math score', nbins=20, title="Distribución de puntajes de matemáticas")
    st.plotly_chart(fig, use_container_width=True)
