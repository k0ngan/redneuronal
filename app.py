import streamlit as st
import pandas as pd
import numpy as np

from nn import NeuralNetwork
import utils

st.set_page_config(page_title="Student Performance", layout="centered")

st.title("Predicción de aprobados en exámenes")
st.markdown(
    """Esta app carga el dataset *Students Performance in Exams*,
    entrena una red neuronal **implementada manualmente** (solo NumPy) y
    muestra la precisión alcanzada.""")

# Carga de datos
uploaded = st.file_uploader("Sube tu propio CSV (formato StudentsPerformance)", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Archivo cargado correctamente.")
else:
    csv_path = utils.download_dataset()
    df = utils.load_dataframe(csv_path)
    st.info("Usando dataset por defecto de KaggleHub.")

st.subheader("Vista preliminar de los datos")
st.dataframe(df.head())

# Preprocesamiento
X, y = utils.preprocess(df)

# División
X_train, X_test, y_train, y_test = utils.train_test_split(X, y)

# Hiperparámetros
st.sidebar.header("Hiperparámetros")
hidden = st.sidebar.slider("Neuronas en capa oculta", 8, 32, 12, step=2)
lr = st.sidebar.number_input("Learning rate", 0.001, 1.0, 0.01, step=0.001, format="%.3f")
epochs = st.sidebar.slider("Epochs", 500, 5000, 2000, step=100)

if st.button("Entrenar red neuronal"):
    with st.spinner("Entrenando…"):
        nn = NeuralNetwork([X_train.shape[1], hidden, 1], lr=lr)
        losses = nn.fit(X_train, y_train, epochs)
        y_pred = nn.predict(X_test)
        accuracy = (y_pred == y_test).mean()
    st.success(f"Precisión en test: **{accuracy:.3f}**")
    utils.plot_loss(losses)

