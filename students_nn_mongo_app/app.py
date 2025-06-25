import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

from db import COL_PRED, COL_RAW, fetch_dataframe, insert_dataframe
from nn import NeuralNetwork
import utils

st.set_page_config(page_title="Students NN Mongo App", layout="wide")

st.title("Predicci\u00f3n de aprobado en ex\u00e1menes")

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Sube un CSV con el formato de StudentsPerformance o descarga el original",
    type="csv",
)

if uploaded is not None:
    csv_path = None
    df_raw = pd.read_csv(uploaded)
else:
    csv_path = utils.download_dataset()
    df_raw = utils.load_dataframe(csv_path)
    st.info("Usando dataset por defecto")

st.subheader("Vista previa de datos")
st.dataframe(df_raw.head())

if st.button("Cargar CSV \u279c MongoDB"):
    insert_dataframe(df_raw, COL_RAW)
    st.success("Datos almacenados en MongoDB")

# ---------------------------------------------------------------------------
# Preprocesamiento y split
# ---------------------------------------------------------------------------

df = utils.preprocess(df_raw)
X_cols = [c for c in df.columns if c != "passed"]
train_df, test_df, X_train, X_test, y_train, y_test = utils.train_test_split_df(
    df, X_cols, "passed"
)

# ---------------------------------------------------------------------------
# Controles de entrenamiento
# ---------------------------------------------------------------------------

st.sidebar.header("Hiperpar\u00e1metros")
hidden = st.sidebar.slider("Neuronas ocultas", 4, 32, 12, step=2)
lr = st.sidebar.number_input("Learning rate", 0.001, 1.0, 0.01, step=0.001, format="%.3f")
epochs = st.sidebar.slider("Epochs", 100, 5000, 1000, step=100)

if st.button("Entrenar modelo"):
    nn = NeuralNetwork([X_train.shape[1], hidden, 1], lr=lr)
    with st.spinner("Entrenando red neuronal..."):
        losses = nn.fit(X_train, y_train, epochs)
    y_pred = nn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Precisi\u00f3n: {acc:.3f}")
    # Guardar predicciones en Mongo
    pred_df = test_df.copy()
    pred_df["prediction"] = y_pred
    insert_dataframe(pred_df, COL_PRED)
    # Gr\u00e1ficas
    utils.plot_loss(losses)
    utils.plot_model_error(y_test, y_pred)
    utils.show_confusion_matrix(y_test, y_pred)
    utils.show_classification_report(y_test, y_pred)

# ---------------------------------------------------------------------------
# Consulta de colecciones
# ---------------------------------------------------------------------------

with st.expander("Consultar registros almacenados"):
    collection = st.selectbox(
        "Selecciona colecci\u00f3n", options=[COL_RAW, COL_PRED], key="collection"
    )
    if st.button("Cargar registros", key="load"):
        df_mongo = fetch_dataframe(collection)
        st.dataframe(df_mongo)
