# Students NN Mongo App

Aplicaci\u00f3n Streamlit que entrena una red neuronal implementada con NumPy sobre el dataset **Students Performance in Exams**. Los datos y las predicciones se almacenan en MongoDB.

## Instalaci\u00f3n
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Configura la variable de entorno `MONGO_URI` si se necesita un URI distinto a `mongodb://localhost:27017`.

## Ejecuci\u00f3n
```bash
streamlit run app.py
```

Las predicciones y los datos crudos se guardan en las colecciones `raw_scores` y `predictions` respectivamente. Opcionalmente se pueden incluir capturas de la interfaz.
