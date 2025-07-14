import os
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import streamlit as st

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "students_db"
COL_DATA = "datasets"

def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info()
    return client[DB_NAME]

def run_etl(csv_path: str):
    df = pd.read_csv(csv_path)
    scores = ['math score', 'reading score', 'writing score']
    df['average_score'] = df[scores].mean(axis=1)
    df['passed'] = (df['average_score'] >= 70).astype(int)
    df_out = df.to_dict("records")
    db = get_db()
    db[COL_DATA].insert_one({
        "filename": os.path.basename(csv_path),
        "timestamp": datetime.utcnow(),
        "rows": df_out
    })
    return len(df)

st.set_page_config(page_title="ETL", layout="wide")
st.title("⚙️ Simulación de flujo ETL")

if "data_uploaded" not in st.session_state or not st.session_state["data_uploaded"]:
    st.warning("Primero sube un archivo en la página principal.")
    st.stop()

df = st.session_state["df"]
scores = ['math score', 'reading score', 'writing score']

df['average_score'] = df[scores].mean(axis=1)
df['passed'] = (df['average_score'] >= 70).astype(int)

st.session_state["df_etl"] = df
st.success("✔️ ETL completado y datos transformados.")

st.dataframe(df.head())
