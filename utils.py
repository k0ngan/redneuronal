import os
import pandas as pd
import numpy as np
import kagglehub
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple

_CACHE_DIR = os.path.expanduser("~/.cache/kagglehub/datasets/spscientist/students-performance-in-exams/latest")

def download_dataset() -> str:
    """Descarga (si es necesario) y retorna la ruta al CSV de StudentsPerformance."""
    if not os.path.exists(_CACHE_DIR):
        st.info("Descargando dataset desde KaggleHub…")
        path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
    else:
        path = _CACHE_DIR
    return os.path.join(path, "StudentsPerformance.csv")

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def preprocess(df: pd.DataFrame):
    """One-hot encode de variables categóricas y crea el target binario 'passed'."""
    # target: aprobar si promedio de tres notas >= 70
    df['average_score'] = df[['math score','reading score','writing score']].mean(axis=1)
    df['passed'] = (df['average_score'] >= 70).astype(int)

    X = df.drop(columns=['math score','reading score','writing score','average_score','passed'])
    X = pd.get_dummies(X)
    y = df['passed'].values.reshape(-1,1)
    return X.values.astype(float), y

def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.3, seed=42) -> Tuple[np.ndarray, ...]:
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    test_size = int(len(X)*test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def plot_loss(losses):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Curva de pérdida durante el entrenamiento")
    st.pyplot(fig)