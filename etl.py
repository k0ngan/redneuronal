"""ETL para DataHub Personal – Etapa 3"""

import os
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "students_db"
COL_DATA = "datasets"

def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info()
    return client[DB_NAME]

def run_etl(csv_path: str):
    """Carga, transforma y guarda datos en MongoDB."""
    df = pd.read_csv(csv_path)

    # --- transformaciones básicas ---
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python etl.py <archivo.csv>")
        sys.exit(1)
    n = run_etl(sys.argv[1])
    print(f"✔️ ETL completado, {n} filas almacenadas.")
