"""Funciones de acceso a MongoDB."""
import os
from typing import Optional

import pandas as pd
from pymongo import MongoClient

# Nombre de base de datos
DB_NAME = "students_nn"
COL_RAW = "raw_scores"
COL_PRED = "predictions"

# Leer URI desde la variable de entorno
URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_client: Optional[MongoClient] = None

def _get_client() -> MongoClient:
    """Obtiene la instancia de MongoClient (singleton)."""
    global _client
    if _client is None:
        _client = MongoClient(URI)
    return _client

def insert_dataframe(df: pd.DataFrame, collection: str) -> None:
    """Inserta todas las filas de un DataFrame en una colecci\u00f3n."""
    if df.empty:
        return
    client = _get_client()
    db = client[DB_NAME]
    db[collection].insert_many(df.to_dict("records"))

def fetch_dataframe(collection: str) -> pd.DataFrame:
    """Obtiene todos los documentos de una colecci\u00f3n como DataFrame."""
    client = _get_client()
    db = client[DB_NAME]
    data = list(db[collection].find())
    return pd.DataFrame(data)
