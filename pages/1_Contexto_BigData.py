import streamlit as st

st.set_page_config(page_title="Contexto Big Data", layout="wide")

st.title("📚 Contexto del Big Data")

st.markdown(
    """### ¿Qué es Big Data?

Big Data se caracteriza por las **5 V**:

| V | Descripción |
|---|-------------|
| **Volumen** | Cantidades masivas de datos que exceden el almacenamiento tradicional. |
| **Velocidad** | Generación y procesamiento en tiempo (casi) real. |
| **Variedad** | Datos estructurados, semi‑estructurados y no estructurados. |
| **Veracidad** | Calidad e incertidumbre de la información. |
| **Valor** | Utilidad que se extrae mediante análisis avanzados. |

### Rol de DataHub Personal

El proyecto integra:

* **Carga y limpieza de datasets estudiantiles**  
* **Transformaciones ETL** para generar variables de interés  
* **Almacenamiento** en MongoDB para trazabilidad  
* **Visualizaciones** interactivas en Streamlit

Cada etapa se documenta y se entrega conforme al cronograma oficial.
"""
)
