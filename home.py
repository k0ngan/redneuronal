import streamlit as st
import pandas as pd

st.set_page_config(page_title="DataHub Personal", layout="wide")

st.title("🏠 DataHub Personal - Inicio")

st.markdown("""
# Bienvenido al proyecto DataHub Personal

Este prototipo integra módulos progresivos para la asignatura **INFB6052 - Herramientas para Ciencia de Datos**.

### Etapas completadas:

1. 📚 Contexto del Big Data  
2. 🔍 Análisis exploratorio de datos  
3. ⚙️ Flujo ETL  
4. 🤖 Entrenamiento y simulación de red neuronal desde cero  
5. 📊 Dashboard final

---

Usa la **barra lateral de Streamlit** para navegar entre las secciones.  
Puedes explorar, entrenar modelos y visualizar resultados finales de forma interactiva.
""")

st.markdown("Sube el archivo CSV una vez para comenzar el flujo completo del sistema.")

csv = st.file_uploader("🔽 Sube tu archivo CSV", type="csv")

if csv:
    df = pd.read_csv(csv)
    st.session_state["df"] = df
    st.session_state["data_uploaded"] = True
    st.success(f"Archivo cargado: {csv.name} con {len(df)} registros")
else:
    st.session_state["data_uploaded"] = False
