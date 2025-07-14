import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime

# ------------------- Parámetros -------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "students_db"
COL_PRED = "predictions"

# ------------------- Conexión MongoDB -------------------
def get_mongo_data():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        db = client[DB_NAME]
        data = list(db[COL_PRED].find())
        return pd.DataFrame(data)
    except Exception as e:
        st.warning("No se pudo conectar a MongoDB. Usa un archivo local.")
        return None

# ------------------- Título -------------------
st.title("📊 Dashboard de Aprobación Estudiantil")

# ------------------- Fuente de datos -------------------
df = get_mongo_data()
if df is None:
    uploaded_file = st.file_uploader("Sube tu archivo CSV de predicciones", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if df is None or df.empty:
    st.info("No hay datos disponibles.")
    st.stop()

# ------------------- Limpieza mínima -------------------
df = df.dropna(subset=["approved"])
df["approved"] = df["approved"].astype(int)

# ------------------- Visualizaciones -------------------

# Aprobados vs No aprobados
st.subheader("Distribución de resultados")
fig1 = px.histogram(df, x="approved", color="approved", barmode="group",
                    category_orders={"approved": [0,1]},
                    color_discrete_sequence=["red", "green"],
                    labels={"approved": "Resultado"},
                    title="Aprobado (1) vs No aprobado (0)")
fig1.update_xaxes(tickvals=[0,1], ticktext=["No aprobado", "Aprobado"])
st.plotly_chart(fig1)

# Probabilidad de aprobación
st.subheader("Histograma de probabilidades")
if "probability" in df.columns:
    fig2 = px.histogram(df, x="probability", nbins=20, title="Distribución de probabilidades")
    st.plotly_chart(fig2)

# Porcentaje total
st.subheader("Resumen")
aprobados = df["approved"].sum()
total = len(df)
st.metric("Porcentaje de aprobación", f"{100 * aprobados / total:.1f}%")

# Muestra de datos
with st.expander("📋 Ver tabla de predicciones"):
    st.dataframe(df)
