import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime

st.set_page_config(page_title="📊 Dashboard MongoDB", layout="wide")
st.title("📊 Dashboard conectado a MongoDB")

# Conexión MongoDB
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "students_db"
COL_DATA = "datasets"
COL_PRED = "predictions"

@st.cache_resource
def get_db():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client[DB_NAME]
    except:
        return None

db = get_db()

if db is None:
    st.error("❌ No se pudo conectar a MongoDB. Asegúrate de que el servidor esté activo.")
    st.stop()

# Leer datos de predicción
docs = list(db[COL_PRED].find({}, {"_id": 0}))
if not docs:
    st.warning("⚠️ No hay predicciones registradas en MongoDB.")
    st.stop()

df = pd.DataFrame(docs)

# Conversión de fecha
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sección de métricas
col1, col2 = st.columns(2)
with col1:
    st.metric("Total de predicciones", len(df))
    aprob = df["approved"].mean() * 100
    st.metric("Tasa de aprobación (%)", f"{aprob:.2f}")

with col2:
    if "timestamp" in df.columns:
        recientes = df[df["timestamp"] > datetime.utcnow() - pd.Timedelta(days=7)]
        st.metric("Predicciones esta semana", len(recientes))

# Gráfico de aprobaciones
st.subheader("📌 Aprobaciones vs No aprobaciones")
if "approved" in df.columns:
    val_counts = df["approved"].value_counts().sort_index()
    labels = ["No aprobado" if i == 0 else "Aprobado" for i in val_counts.index]
    fig1 = px.bar(x=labels, y=val_counts.values, labels={"x": "Aprobación", "y": "Cantidad"},
                  color=labels, color_discrete_sequence=["#e74c3c", "#2ecc71"])
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

# Promedios de puntaje por grupo (si están disponibles)
if all(col in df.columns for col in ["math score", "reading score", "writing score", "race/ethnicity"]):
    st.subheader("📌 Promedio de puntajes por grupo étnico")
    mean_scores = df.groupby("race/ethnicity")[["math score", "reading score", "writing score"]].mean().reset_index()
    fig2 = px.bar(mean_scores, x="race/ethnicity", y=["math score", "reading score", "writing score"],
                  barmode="group", labels={"value": "Promedio", "race/ethnicity": "Grupo étnico", "variable": "Puntaje"})
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("ℹ️ No se encontraron columnas de puntaje o grupo étnico para análisis por grupo.")

st.set_page_config(page_title="📊 Dashboard", layout="wide")
st.title("📊 Dashboard general")

if "df_etl" not in st.session_state:
    st.warning("Primero procesa los datos en la etapa ETL.")
    st.stop()

df = st.session_state["df_etl"]

col1, col2 = st.columns(2)
with col1:
    st.metric("Total registros", len(df))
    st.metric("Tasa de aprobación (%)", f"{df['passed'].mean()*100:.2f}")
with col2:
    st.metric("Promedio matemática", f"{df['math score'].mean():.1f}")
    st.metric("Promedio lectura", f"{df['reading score'].mean():.1f}")
    st.metric("Promedio escritura", f"{df['writing score'].mean():.1f}")

st.subheader("Aprobaciones por grupo étnico")
if "race/ethnicity" in df.columns:
    aprob_group = df.groupby("race/ethnicity")["passed"].mean().reset_index()
    fig3 = px.bar(aprob_group, x="race/ethnicity", y="passed", labels={"passed": "Tasa de aprobación", "race/ethnicity": "Grupo étnico"})
    fig3.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)
