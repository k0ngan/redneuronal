
# 🧠 DataHub Personal – Proyecto INFB6052

**Francisco Pinto**  
Ingeniería Civil en Ciencia de Datos – UTEM  
Curso: INFB6052 – Herramientas para Ciencia de Datos  
Docente: Dr. Michael Miranda Sandoval  
Semestre: Primer semestre 2025

---

## 📌 Descripción general

**DataHub Personal** es una aplicación web desarrollada en **Streamlit** que integra un flujo completo de análisis educativo:
1. Carga y análisis exploratorio de datos.
2. Transformación mediante flujo ETL.
3. Entrenamiento de una red neuronal **desde cero**, sin librerías de ML.
4. Predicción interactiva y almacenamiento en MongoDB.
5. Visualización de resultados a través de un dashboard final.

---

## 🚀 Tecnologías utilizadas

- Python 3.11
- Streamlit
- Pandas, NumPy, Matplotlib
- MongoDB (como base de datos documental)
- Diseño modular con múltiples archivos `.py`

---

## 🗂️ Estructura del proyecto

```bash
├── home.py                  # Inicio de la app y carga del CSV
├── 1_Contexto_BigData.py    # Introducción y explicación del proyecto
├── 2_Analisis_Exploratorio.py
├── 3_ETL.py                 # Flujo ETL: limpieza y exportación a MongoDB
├── 4_Red_neuronal.py        # Red neuronal multicapa desde cero
├── 5_Dashboard.py           # Visualización de métricas y análisis final
├── documentos/              # Presentaciones, diagramas y documentación técnica
│   ├── documentacion_etl.docx
│   ├── flujo_etl_diagrama.pdf
│   └── Presentacion_Final_DataHub_Personal.pptx
```

---

## 🧪 Instrucciones para ejecutar localmente

1. Clona este repositorio:
```bash
git clone https://github.com/tuusuario/datahub-personal.git
cd datahub-personal
```

2. Instala dependencias:
```bash
pip install -r requirements.txt
```

3. Asegúrate de que **MongoDB esté activo** en `localhost:27017`.

4. Ejecuta la app:
```bash
streamlit run home.py
```

5. Usa la barra lateral para navegar entre las etapas.

---

## 🧠 Red Neuronal desde Cero

- Codificada manualmente usando NumPy.
- Permite ajustar hiperparámetros:
  - Neuronas ocultas
  - Learning rate
  - Épocas
  - Tamaño del batch
- Entrenamiento y validación con early-stopping.
- Visualización de:
  - Curva de pérdida
  - Histograma de pesos
  - Predicción interactiva con gauge

---

## 📊 Dashboard final

- Métricas globales de aprobación
- Comparaciones por grupo étnico
- Historial de predicciones almacenadas en MongoDB

---

## 📄 Documentación

La carpeta `documentos/` incluye:
- 📘 Diagrama del flujo ETL
- 📊 Presentación final en PowerPoint
- 📑 Documentación breve en Word
- 📄 Versión PDF generada en LaTeX (opcional)

---

## ✅ Estado del proyecto

| Etapa         | Estado       |
|---------------|--------------|
| Contexto      | ✔️ Completo  |
| Exploratorio  | ✔️ Completo  |
| ETL           | ✔️ Completo  |
| Red neuronal  | ✔️ Completo  |
| Dashboard     | ✔️ Completo  |

---

## ✍️ Autor

Francisco Pinto  
[UTEM – Ingeniería en Ciencia de Datos]

---

## 📜 Licencia

Este proyecto es de uso académico. No se permite su reproducción sin autorización del autor o del docente a cargo.
