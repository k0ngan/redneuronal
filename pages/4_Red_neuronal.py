from __future__ import annotations
import os, pickle, io
import math
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st
from pymongo import MongoClient
import plotly.graph_objects as go
import plotly.express as px

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


# ---------- 1. Cargar dataset desde session ----------
if "df" not in st.session_state:
    st.warning("Primero sube el archivo en la pestaña de inicio.")
    st.stop()

df_raw = st.session_state["df"].copy()
st.success(f"Cargado {len(df_raw)} registros.")
if db is not None:  # guarda dataset en Mongo
    db[COL_DATA].insert_one({
        "filename": "session_state_df",
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
    # ----- GRÁFICO 1: Curva de pérdida train/val (Plotly) -----
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=loss_tr_hist, mode='lines', name='train'))
    fig1.add_trace(go.Scatter(y=loss_val_hist, mode='lines', name='val'))
    fig1.update_layout(title="Curva de pérdida", xaxis_title="Época", yaxis_title="Pérdida")
    st.plotly_chart(fig1, use_container_width=True)
    # ----- GRÁFICO 2: Histograma de pesos de la capa oculta (Plotly) -----
    fig2 = px.histogram(W1.flatten(), nbins=30, title="Distribución de pesos W1")
    st.plotly_chart(fig2, use_container_width=True)
    # ----- GRÁFICO 3: Gauge de probabilidad de aprobación (Plotly) -----
    if "last_prob" in st.session_state:
        prob = st.session_state["last_prob"]
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob*100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de aprobar (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4caf50"},
                'steps': [
                    {'range': [0, 50], 'color': "#e0e0e0"},
                    {'range': [50, 100], 'color': "#4caf50"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

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
