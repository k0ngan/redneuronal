from __future__ import annotations
import os, pickle, io
import math
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st
from pymongo import MongoClient
import matplotlib.pyplot as plt  # ← nuevo

# ---------- Conexión Mongo ----------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "students_db"
COL_DATA = "datasets"     # csv crudo
COL_MODEL = "models"      # pesos y metadatos
COL_PRED = "predictions"  # resultados de inferencia

def get_db():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client[DB_NAME]
    except Exception:
        st.error("❌ No se pudo conectar a MongoDB")
        return None

db = get_db()

# ---------- Red neuronal manual ----------
SEED = 42; np.random.seed(SEED)
def sigmoid(z): return 1/(1+np.exp(-z))

def init_params(n_in,n_hid,n_out):
    W1=np.random.randn(n_in,n_hid)*np.sqrt(2/n_in)
    b1=np.zeros((1,n_hid))
    W2=np.random.randn(n_hid,n_out)*np.sqrt(2/n_hid)
    b2=np.zeros((1,n_out))
    return W1,b1,W2,b2

def forward(X,W1,b1,W2,b2):
    Z1=X@W1+b1; A1=sigmoid(Z1); Z2=A1@W2+b2; A2=sigmoid(Z2)
    return Z1,A1,Z2,A2

def loss(y,yp):
    m=len(y)
    return float(-np.sum(y*np.log(yp+1e-15)+(1-y)*np.log(1-yp+1e-15))/m)

def backward(X,y,Z1,A1,A2,W2):
    m=len(X)
    dZ2=A2-y; dW2=A1.T@dZ2/m; db2=dZ2.sum(0,keepdims=True)/m
    dA1=dZ2@W2.T; dZ1=dA1*A1*(1-A1); dW1=X.T@dZ1/m; db1=dZ1.sum(0,keepdims=True)/m
    return dW1,db1,dW2,db2

def update(params,grads,lr):
    W1,b1,W2,b2=params; dW1,db1,dW2,db2=grads
    W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2
    return W1,b1,W2,b2

def to_bson(obj):
    """Convierte np.ndarray → list y pd.Series → dict para que MongoDB los acepte."""
    import numpy as np, pandas as pd
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj

# ---------- Streamlit ----------
st.title("Predicción aprobación estudiantil")

st.markdown("""
### 📝 Guía de uso detallada

**¿Qué hace esta app?**  
Entrena (o carga) una red neuronal manual para clasificar a un estudiante como **Aprobado** o **No aprobado**.  
Se basa en sus puntajes de *matemáticas, lectura y escritura* más cinco variables socio-académicas.

---

#### 1. Barra lateral – Hiperparámetros

| Control | Significado | Efecto práctico |
|---------|-------------|-----------------|
| **Neuronas ocultas** | Tamaño de la capa oculta (complejidad del modelo). | Más neuronas ⇒ mayor capacidad para aprender patrones, pero también riesgo de sobreajuste y entrenamiento más lento. |
| **Learning rate** | Paso del descenso de gradiente. | Valores altos convergen rápido pero pueden oscilar; valores bajos son estables pero lentos. |
| **Épocas** | Ciclos máximos de entrenamiento. | Más épocas permiten converger mejor *si* el modelo no se ha estancado. El sistema detiene antes con *early-stopping* si no mejora. |
| **Batch size** | Cantidad de ejemplos por actualización. | Batches pequeños ⇒ gradientes ruidosos pero mejor generalización. Batches grandes ⇒ entrenamiento más estable y uso eficiente de CPU/GPU. |

> Cambiar cualquiera de estos parámetros **fuerza un reentrenamiento** inmediato.

---

#### 2. Exploración de datos

* Expande la pestaña **Exploración de datos** para ver:  
  * Primeras filas del CSV (contexto del dataset).  
  * Número total de registros y variables predictoras.  
  * **Tasa global de aprobación** (porcentaje).  
  * Gráfico de barras *Aprobado / No*.

Esto te permite verificar la calidad y el balance de la información antes de entrenar o probar el modelo.

---

#### 3. Entrada de características

1. Ajusta los **sliders** de **Math**, **Reading** y **Writing** (0-100).  
2. Selecciona opciones en los *selectboxes*:  
   * `Gender`  
   * `Race/Ethnicity`  
   * `Parental level of education`  
   * `Lunch` (subsidized o estándar)  
   * `Test preparation course` (none / completed)

Cada combinación se convierte en variables *one-hot* y se normaliza con los **µ / σ** calculados durante el entrenamiento.

---

#### 4. Botón **Predecir**

* Ejecuta la inferencia:  
  * Muestra la **probabilidad** en porcentaje.  
  * Muestra un cuadro verde (**Aprobado**) o rojo (**No aprobado**).
* El dato se guarda en MongoDB (`students_db.predictions`) si el servidor está disponible.

Campos almacenados: puntajes, variables categóricas, `probability`, `approved`, `timestamp`.

---

#### 5. Curva de pérdida

Si el modelo se reentrenó:  
* Gráfico **train vs validation loss** por época.  
* Te permite evaluar convergencia y detectar sobreajuste.

---

##### Resumen del flujo

1. Configura hiperparámetros → *entrena/carga*.  
2. Revisa los datos.  
3. Ingresa características → **Predecir**.  
4. Observa resultado y curva.  
5. Repite con otros estudiantes o parámetros.

¡Listo! Ahora sabes exactamente qué controla cada parte de la app y cómo influye en el rendimiento y la salida del modelo.
""")
# ---------- 1. Subir CSV ----------
csv_file = st.file_uploader("1️⃣ Sube tu archivo CSV (mismas columnas que StudentsPerformance)", type="csv")
if csv_file is None:
    st.info("Sube un archivo para comenzar.")
    st.stop()

# Lee CSV
df_raw = pd.read_csv(csv_file)
st.success(f"Cargado {len(df_raw)} registros.")
if db is not None:  # guarda dataset en Mongo
    db[COL_DATA].insert_one({
        "filename": csv_file.name,
        "timestamp": datetime.utcnow(),
        "rows": df_raw.to_dict("records")   # cuidado con tamaños grandes
    })

# ---------- 2. Preprocesamiento ----------
scores = ['math score','reading score','writing score']
cat_cols = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

df_raw['average_score'] = df_raw[scores].mean(1)
df_raw['passed'] = (df_raw['average_score'] >= 70).astype(int)

X = pd.concat([df_raw[scores],
               pd.get_dummies(df_raw[cat_cols], dtype=float)],
              axis=1).astype(float)
y = df_raw['passed'].values.reshape(-1,1)

# Normalización
mean = X.mean(); std = X.std().replace(0,1)
Xn = (X-mean)/std

# ---------- 3. Hiperparámetros ----------
st.sidebar.header("Hiperparámetros")
hid    = st.sidebar.slider("Neuronas ocultas",5,50,20,5)
lr     = st.sidebar.number_input("Learning rate",0.0001,1.0,0.01,format="%.4f")
epochs = st.sidebar.number_input("Épocas",100,5000,2000,100)
batch  = st.sidebar.selectbox("Batch size",[16,32,64,128],1)

# ---------- 4. Entrenamiento ----------
if "model" not in st.session_state:
    st.session_state["model"] = None

def train(X, y, hid, lr, epochs, batch, pat=10):
    idx = np.random.permutation(len(X))
    val = int(.2*len(X))
    Xv, yv = X[idx[:val]], y[idx[:val]]
    Xt, yt = X[idx[val:]], y[idx[val:]]

    W1, b1, W2, b2 = init_params(X.shape[1], hid, 1)
    best_val, wait, best = 1e9, 0, None
    loss_tr_hist, loss_val_hist = [], []

    for ep in range(epochs):
        # mini-batch
        for s in range(0, len(Xt), batch):
            xb, yb = Xt[s:s+batch], yt[s:s+batch]
            Z1, A1, _, A2 = forward(xb, W1, b1, W2, b2)
            grads = backward(xb, yb, Z1, A1, A2, W2)
            W1, b1, W2, b2 = update((W1, b1, W2, b2), grads, lr)

        # métricas por época
        _, _, _, A2_tr = forward(Xt, W1, b1, W2, b2)
        _, _, _, A2_val = forward(Xv, W1, b1, W2, b2)
        l_tr, l_val = loss(yt, A2_tr), loss(yv, A2_val)
        loss_tr_hist.append(l_tr)
        loss_val_hist.append(l_val)

        # early stopping
        if l_val < best_val:
            best_val, best, wait = l_val, (W1, b1, W2, b2), 0
        else:
            wait += 1
            if wait >= pat:
                break

    return best, loss_tr_hist, loss_val_hist

if st.button("🚀 Entrenar modelo"):
    idx=np.random.permutation(len(Xn))
    val=int(.2*len(Xn)); Xv,yv=Xn.iloc[idx[:val]],y[idx[:val]]
    Xt,yt=Xn.iloc[idx[val:]],y[idx[val:]]
    # Entrenamiento y obtención de historia de pérdidas
    (W1,b1,W2,b2), loss_tr_hist, loss_val_hist = train(Xn.values, y, hid, lr, epochs, batch)
    total_steps=epochs*math.ceil(len(Xt)/batch)
    st.success("Entrenamiento terminado ✔️")
    st.session_state["model"] = dict(W1=W1,b1=b1,W2=W2,b2=b2,
                                     mean=mean,std=std,cols=list(X.columns))
    if db is not None:
        model_doc = {
            "timestamp": datetime.utcnow(),
            "hid": hid,
            "lr": lr,
            "epochs": epochs,
            "batch": batch,
            **{k: to_bson(v) for k, v in st.session_state["model"].items()}
        }
        db[COL_MODEL].insert_one(model_doc)
        st.toast("Modelo guardado en MongoDB")
    # ----- GRÁFICO 1: Curva de pérdida train/val -----
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_tr_hist, label="train")
    ax1.plot(loss_val_hist, label="val")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida"); ax1.legend()
    st.pyplot(fig1)
    # ----- GRÁFICO 2: Histograma de pesos de la capa oculta -----
    fig2, ax2 = plt.subplots()
    ax2.hist(W1.flatten(), bins=30)
    ax2.set_title("Distribución de pesos W1")
    st.pyplot(fig2)
    # ----- GRÁFICO 3: Gauge de probabilidad de aprobación (caso individual) -----
    # Mostrar gauge solo si hay predicción reciente
    if "last_prob" in st.session_state:
        prob = st.session_state["last_prob"]
        fig_gauge, ax_gauge = plt.subplots(subplot_kw={'aspect': 'equal'})
        vals = [prob, 1-prob]
        colors = ["#4caf50", "#e0e0e0"]
        wedges, _ = ax_gauge.pie(vals, startangle=180, radius=1.2, colors=colors, counterclock=False)
        ax_gauge.add_artist(plt.Circle((0,0), 0.7, color='white'))
        ax_gauge.text(0, -0.1, f"{prob*100:.1f}%", ha='center', va='center', fontsize=18, fontweight='bold')
        ax_gauge.set_title("Probabilidad de aprobar (Gauge)")
        ax_gauge.set(aspect="equal")
        plt.tight_layout()
        st.pyplot(fig_gauge)

# ---------- 5. Predicción ----------
st.divider()
st.header("🔮 Hacer predicciones")

if st.session_state["model"] is None:
    st.info("Entrena primero para habilitar la predicción.")
    st.stop()

math  = st.slider("Math score",0,100,70)
read  = st.slider("Reading score",0,100,70)
write = st.slider("Writing score",0,100,70)
col1,col2 = st.columns(2)
gen   = col1.selectbox("Gender", sorted(df_raw['gender'].unique()))
race  = col2.selectbox("Race/Ethnicity", sorted(df_raw['race/ethnicity'].unique()))
parent= st.selectbox("Parental education", sorted(df_raw['parental level of education'].unique()))
lunch = st.selectbox("Lunch", sorted(df_raw['lunch'].unique()))
prep  = st.selectbox("Test prep course", sorted(df_raw['test preparation course'].unique()))

input_dict = {'math score':math,'reading score':read,'writing score':write,
              'gender':gen,'race/ethnicity':race,'parental level of education':parent,
              'lunch':lunch,'test preparation course':prep}



if st.button("Predecir"):
    user = pd.get_dummies(pd.DataFrame([input_dict])).reindex(
        columns=st.session_state["model"]["cols"], fill_value=0)
    user = (user - st.session_state["model"]["mean"]) / st.session_state["model"]["std"]
    W1=st.session_state["model"]["W1"]; b1=st.session_state["model"]["b1"]
    W2=st.session_state["model"]["W2"]; b2=st.session_state["model"]["b2"]
    _,_,_,prob = forward(user.values.astype(float), W1,b1,W2,b2)
    prob = prob[0,0]; pred = int(prob>=0.5)
    st.session_state["last_prob"] = prob  # <-- para el gauge
    st.metric("Probabilidad de aprobar (%)", f"{prob*100:.1f}")
    st.success("Aprobado") if pred else st.error("No aprobado")
    if db is not None:
        db[COL_PRED].insert_one({**input_dict,
                                 "probability":prob,
                                 "approved":bool(pred),
                                 "timestamp":datetime.utcnow()})
        st.toast("Predicción guardada en MongoDB")
