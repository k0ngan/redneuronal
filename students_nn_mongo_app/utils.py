"""Funciones auxiliares para manejo de datos y gr\u00e1ficos."""
from __future__ import annotations

import os
from typing import Tuple

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
import streamlit as st

_DATASET = "spscientist/students-performance-in-exams"
_CACHE_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/spscientist/students-performance-in-exams/latest"
)


# ---------------------------------------------------------------------------
# Descarga y carga
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    """Descarga el dataset de KaggleHub y devuelve la ruta local al CSV."""
    if not os.path.exists(_CACHE_DIR):
        st.info("Descargando dataset desde KaggleHub...")
        path = kagglehub.dataset_download(_DATASET)
    else:
        path = _CACHE_DIR
    return os.path.join(path, "StudentsPerformance.csv")


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el DataFrame generando variables dummies y objetivo."""
    df = df.copy()
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["passed"] = (df["average_score"] >= 70).astype(int)
    X = pd.get_dummies(
        df.drop(columns=["math score", "reading score", "writing score", "average_score", "passed"])
    )
    y = df["passed"]
    result = pd.concat([X, y], axis=1)
    return result


def train_test_split_df(
    df: pd.DataFrame, X_cols: list[str], y_col: str, test_ratio: float = 0.3, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    test_size = int(len(df) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train = train_df[X_cols].values.astype(float)
    X_test = test_df[X_cols].values.astype(float)
    y_train = train_df[y_col].values.reshape(-1, 1)
    y_test = test_df[y_col].values.reshape(-1, 1)
    return train_df, test_df, X_train, X_test, y_train, y_test


def plot_loss(losses: list[float]) -> None:
    fig, ax = plt.subplots()
    ax.plot(losses, label="MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de p\u00e9rdida")
    ax.legend()
    st.pyplot(fig)


def plot_model_error(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from scipy import stats

    errs = y_true.flatten() - y_pred.flatten()
    classes = ["reprob\u00f3", "aprob\u00f3"]
    data = []
    for c in (0, 1):
        c_err = errs[y_true.flatten() == c]
        mean = c_err.mean()
        ci = stats.t.interval(0.95, len(c_err)-1, loc=mean, scale=stats.sem(c_err)) if len(c_err) > 1 else (mean, mean)
        data.append((classes[c], mean, ci))
    fig, ax = plt.subplots()
    for label, mean, ci in data:
        ax.bar(label, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=5)
    ax.set_ylabel("Error")
    ax.set_title("Error de predicci\u00f3n por clase (95% IC)")
    st.pyplot(fig)


def show_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicciones")
    ax.set_ylabel("Actual")
    ax.set_title("Matriz de confusi\u00f3n")
    st.pyplot(fig)


def show_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, output_dict=False)
    st.text("Reporte de clasificaci\u00f3n:\n" + report)
