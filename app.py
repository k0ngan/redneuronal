from __future__ import annotations
import os, pickle
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st
from pymongo import MongoClient

SEED = 42
np.random.seed(SEED)

CSV_PATH = r"C:\Users\francisco\red neuronal\data\StudentsPerformance_synthetic_200.csv"
MODEL_PATH = "student_nn_model.pkl"
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME, COL_PRED = "students_db", "predictions"

def sigmoid(z): return 1/(1+np.exp(-z))

def init_params(n_in,n_hid,n_out):
    W1 = np.random.randn(n_in,n_hid)*np.sqrt(2/n_in)
    b1 = np.zeros((1,n_hid))
    W2 = np.random.randn(n_hid,n_out)*np.sqrt(2/n_hid)
    b2 = np.zeros((1,n_out))
    return W1,b1,W2,b2

def forward(X,W1,b1,W2,b2):
    Z1=X@W1+b1; A1=sigmoid(Z1); Z2=A1@W2+b2; A2=sigmoid(Z2)
    return Z1,A1,Z2,A2

def loss(y,yp): m=len(y); return float(-np.sum(y*np.log(yp+1e-15)+(1-y)*np.log(1-yp+1e-15))/m)

def backward(X,y,Z1,A1,A2,W2):
    m=len(X); dZ2=A2-y; dW2=A1.T@dZ2/m; db2=dZ2.sum(0,keepdims=True)/m
    dA1=dZ2@W2.T; dZ1=dA1*A1*(1-A1); dW1=X.T@dZ1/m; db1=dZ1.sum(0,keepdims=True)/m
    return dW1,db1,dW2,db2

def update(params,grads,lr):
    W1,b1,W2,b2=params; dW1,db1,dW2,db2=grads
    W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2; return W1,b1,W2,b2

def load_prepare(path):
    df=pd.read_csv(path)
    scores=['math score','reading score','writing score']
    df['average_score']=df[scores].mean(1); df['passed']=(df['average_score']>=70).astype(int)
    X=pd.concat([df[scores],pd.get_dummies(df[['gender','race/ethnicity','parental level of education','lunch','test preparation course']])],1).astype(float)
    y=df['passed']; return df,X,y,scores

def norm(df,mean=None,std=None):
    if mean is None:
        mean=df.mean(); std=df.std().replace(0,1)
    return (df-mean)/std,mean,std

def train(X,y,hid,lr,epochs,batch,pat=10):
    idx=np.random.permutation(len(X)); val=int(.2*len(X))
    val_idx, tr_idx=idx[:val], idx[val:]
    Xv,yv=X[val_idx],y[val_idx]; Xt,yt=X[tr_idx],y[tr_idx]
    W1,b1,W2,b2=init_params(X.shape[1],hid,1)
    best, best_val, wait=None,1e9,0
    for ep in range(epochs):
        for s in range(0,len(Xt),batch):
            e=s+batch; xb, yb= Xt[s:e], yt[s:e]
            Z1,A1,_,A2=forward(xb,W1,b1,W2,b2)
            grads=backward(xb,yb,Z1,A1,A2,W2)
            W1,b1,W2,b2=update((W1,b1,W2,b2),grads,lr)
        _,_,_,A2v=forward(Xv,W1,b1,W2,b2); lv=loss(yv,A2v)
        if lv<best_val: best_val=lv; best=(W1,b1,W2,b2); wait=0
        else: wait+=1; 
        if wait>=pat: break
    return best

def mongo():
    try: c=MongoClient(MONGO_URI,serverSelectionTimeoutMS=2000); c.server_info(); return c[DB_NAME]
    except: st.warning("MongoDB off"); return None

def log_prediction(db, record: dict) -> None:
    """Guarda la predicción en MongoDB si la conexión es válida."""
    if db is not None:
        db[COL_PRED].insert_one(record)

@st.cache_resource(show_spinner="Cargando modelo…")
def get_model(hid,lr,epochs,batch):
    df,X,y,scores=load_prepare(CSV_PATH)
    Xn,mean,std=norm(X)
    if os.path.exists(MODEL_PATH):
        try:
            W1,b1,W2,b2,mean,std,cols=pickle.load(open(MODEL_PATH,'rb'))
            if list(Xn.columns)!=cols: raise ValueError
            return (W1,b1,W2,b2),mean,std,X,df,scores
        except: os.remove(MODEL_PATH)
    st.info("Entrenando...")
    weights=train(Xn.values,y.values.reshape(-1,1),hid,lr,epochs,batch)
    W1,b1,W2,b2=weights
    pickle.dump((W1,b1,W2,b2,mean,std,list(Xn.columns)),open(MODEL_PATH,'wb'))
    return (W1,b1,W2,b2),mean,std,X,df,scores

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


hid=st.sidebar.slider("Hidden",5,50,20,5); lr=st.sidebar.number_input("lr",0.0001,1.0,0.01,format="%.4f")
epochs=st.sidebar.number_input("epochs",100,5000,2000,100); batch=st.sidebar.selectbox("batch",[16,32,64,128],1)
(model,mean,std,X_df,df_raw,scores)=get_model(hid,lr,epochs,batch)
W1,b1,W2,b2=model

math=st.slider("Math",0,100,70); read=st.slider("Read",0,100,70); write=st.slider("Write",0,100,70)
col1,col2=st.columns(2)
gen=col1.selectbox("Gender",sorted(df_raw['gender'].unique()))
race=col2.selectbox("Race",sorted(df_raw['race/ethnicity'].unique()))
parent=st.selectbox("Parental edu",sorted(df_raw['parental level of education'].unique()))
lunch=st.selectbox("Lunch",sorted(df_raw['lunch'].unique()))
prep=st.selectbox("Prep",sorted(df_raw['test preparation course'].unique()))
inp={**dict(zip(scores,[math,read,write])),
     'gender':gen,'race/ethnicity':race,'parental level of education':parent,
     'lunch':lunch,'test preparation course':prep}
user=pd.get_dummies(pd.DataFrame([inp])).reindex(columns=X_df.columns,fill_value=0)
user=(user-mean)/std
_,_,_,prob=forward(user.values.astype(float),W1,b1,W2,b2)
pred=int(prob[0,0]>=0.5)
if st.button("Predecir"):
    st.metric("Prob%",f"{prob[0,0]*100:.1f}")
    st.success("Aprobado") if pred else st.error("No aprobado")
    db=mongo(); log_prediction(db,{**inp,'prob':float(prob[0,0]),'ok':bool(pred),'ts':datetime.utcnow()})

# ---------- Carga dinámica de dataset ----------
st.sidebar.subheader("Fuente de datos")
use_upload = st.sidebar.checkbox("🔄 Usar mi propio CSV", value=False)

if use_upload:
    uploaded_file = st.sidebar.file_uploader(
        "Selecciona un archivo .csv (mismas columnas que StudentsPerformance)",
        type="csv",
    )
    if uploaded_file is not None:
        CSV_PATH = uploaded_file  # Streamlit devuelve un buffer tipo BytesIO
        st.success("Archivo cargado exitosamente. Se usará para el entrenamiento.")
    else:
        st.warning("Sube un archivo para continuar. Se mantendrá el dataset por defecto.")
