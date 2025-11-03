# model_evaluation.py
import os
import json
import requests
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================
# 🎨 UI CONFIG
# ============================
st.set_page_config(
    page_title="Evaluación Modelo FastAPI",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
.reportview-container { background: #0E1117; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Evaluación del Modelo Desplegado (FastAPI)")

# ============================
# 1️⃣ Config
# ============================
API_URL = "http://localhost:8000/predict_batch"
TARGET = "membresia_premium"

def load_project():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["proyecto"]

def load_from_bigquery():
    project_id = load_project()
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT *
        FROM `{project_id}.proyecto_final_cdp.restaurantes_features`
    """
    return client.query(query).to_dataframe()


# ============================
# 2️⃣ Cargar datos
# ============================
st.write("📥 Extrayendo dataset desde BigQuery...")
df = load_from_bigquery()

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.success(f"✅ Datos cargados: {df.shape} — X_test:{X_test.shape}, y_test:{y_test.shape}")


# ============================
# 3️⃣ API Request
# ============================
st.write("🔗 Solicitando predicciones al endpoint `/predict_batch` ...")

payload = {"batch": X_test.values.tolist()}
response = requests.post(API_URL, json=payload)

if response.status_code != 200:
    st.error(f"❌ Error API: {response.text}")
    st.stop()

result = response.json()
preds = result["predictions"]
probs = result.get("probabilities")


# ============================
# 🧷 UI Tabs
# ============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Métricas",
    "🧊 Matriz de Confusión",
    "📈 Curva ROC",
    "📉 Precision-Recall & Probabilidades"
])


# ============================
# TAB 1️⃣ Métricas
# ============================
with tab1:
    st.subheader("📌 Métricas de Clasificación")
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    st.json(report)


# ============================
# TAB 2️⃣ Matriz de Confusión
# ============================
with tab2:
    st.subheader("🧊 Matriz de Confusión")
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Premium", "Premium"],
                yticklabels=["No Premium", "Premium"], ax=ax)
    ax.set_xlabel("Predicciones")
    ax.set_ylabel("Reales")
    st.pyplot(fig)


# ============================
# TAB 3️⃣ ROC Curve
# ============================
with tab3:
    st.subheader("📈 Curva ROC")
    if probs:
        probs1 = [p[1] for p in probs]
        fpr, tpr, _ = roc_curve(y_test, probs1)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⚠️ La API no regresó probabilidades.")


# ============================
# TAB 4️⃣ Precision-Recall + Prob Hist
# ============================
with tab4:
    if probs:
        probs1 = [p[1] for p in probs]

        # PR Curve
        st.subheader("📉 Curva Precision-Recall")
        precision, recall, _ = precision_recall_curve(y_test, probs1)

        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        st.pyplot(fig)

        # Prob Histogram
        st.subheader("📦 Distribución Probabilidades (Clase Premium)")
        fig, ax = plt.subplots()
        ax.hist(probs1, bins=20)
        ax.set_xlabel("Probabilidad")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No se recibieron probabilidades para análisis adicional.")
