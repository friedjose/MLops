# model_monitoring.py
import os
import time
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split
from google.cloud import bigquery
import json

# ================================
# 1. Configuración
# ================================
API_URL = "http://localhost:8000/predict_batch"
MONITOR_LOG = "monitoring_log.csv"
TARGET = "membresia_premium"

# =============== Leer config proyecto ========
def load_project():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["proyecto"]

# ================================
# 2. Cargar dataset desde BigQuery y dividir
# ================================
@st.cache_data
def load_data():
    project_id = load_project()
    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT *
        FROM `{project_id}.proyecto_final_cdp.restaurantes_features`
    """

    df = client.query(query).to_dataframe()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()

# ================================
# 3. API para predicciones
# ================================
def get_predictions(X_batch: pd.DataFrame):
    payload = {"batch": X_batch.values.tolist()}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        preds = response.json()["predictions"]
        return preds
    except Exception as e:
        st.error(f"❌ Error conectando con la API: {e}")
        return None

# ================================
# 4. Guardar logs con timestamp
# ================================
def log_predictions(X_batch, preds):
    log_df = X_batch.copy()
    log_df["prediction"] = preds
    log_df["timestamp"] = pd.Timestamp.now()

    if os.path.exists(MONITOR_LOG):
        log_df.to_csv(MONITOR_LOG, mode="a", header=False, index=False)
    else:
        log_df.to_csv(MONITOR_LOG, index=False)

# ================================
# 5. Reporte Evidently
# ================================
def generate_drift_report(ref_data, new_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)
    return report

# ================================
# 6. Streamlit UI
# ================================
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
st.title("📊 Monitoreo del Modelo en Producción")

# Métricas principales
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Predicciones", len(logged_data))
    with col2: st.metric("Promedio Predicción", f"{logged_data['prediction'].mean():.3f}")
    with col3: st.metric("Desv. Estándar", f"{logged_data['prediction'].std():.3f}")
    with col4:
        positive_rate = (logged_data['prediction'] > 0.5).mean() * 100
        st.metric("Tasa Positiva (%)", f"{positive_rate:.1f}%")

st.sidebar.header("Opciones")
sample_size = st.sidebar.slider("Tamaño de muestra:", 50, 500, 200)

if st.button("🔄 Generar nuevas predicciones y actualizar log"):
    sample = X_new.sample(n=sample_size, random_state=int(time.time()))
    preds = get_predictions(sample)
    if preds:
        log_predictions(sample, preds)
        st.success("✅ Nuevas predicciones agregadas.")
        st.rerun()

# Mostrar datos y gráficos
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)

    tab1, tab2, tab3 = st.tabs(["📈 Gráficas", "📊 Data Drift", "📂 Logs"])

    # --- Tab 1: Gráficas ---
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                logged_data, x='prediction', nbins=20,
                title="Distribución de Predicciones"
            )
            st.plotly_chart(fig_hist, width="stretch")

        with col2:
            if 'timestamp' in logged_data.columns:
                logged_data['timestamp'] = pd.to_datetime(logged_data['timestamp'])
                temporal_data = logged_data.groupby(
                    logged_data['timestamp'].dt.floor('T')
                )['prediction'].mean().reset_index()

                fig_time = px.line(
                    temporal_data, x='timestamp', y='prediction',
                    title="Evolución Temporal de Predicciones"
                )
                st.plotly_chart(fig_time, width="stretch")

        st.subheader("🔍 Comparación con Datos de Referencia")
        numeric_cols = [col for col in logged_data.select_dtypes(['float64','int64']).columns if col != 'prediction'][:4]

        if numeric_cols:
            comp_df = pd.DataFrame([
                {
                    'Feature': col,
                    'Referencia': X_ref[col].mean(),
                    'Actual': logged_data[col].mean(),
                }
                for col in numeric_cols if col in X_ref.columns
            ])

            fig_comp = go.Figure([
                go.Bar(name='Referencia', x=comp_df['Feature'], y=comp_df['Referencia']),
                go.Bar(name='Actual', x=comp_df['Feature'], y=comp_df['Actual'])
            ])
            fig_comp.update_layout(title="Comparación de Medias", barmode='group')
            st.plotly_chart(fig_comp, width="stretch")

    # --- Tab 2: Evidently Drift ---
    with tab2:
        drift_report = generate_drift_report(
            X_ref, logged_data.drop(columns=["prediction","timestamp"], errors="ignore")
        )

        try:
            st.components.v1.html(drift_report._repr_html_(), height=1000, scrolling=True)
        except:
            st.write("✅ Reporte Evidently generado")

    # --- Tab 3: Logs ---
    with tab3:
        show_rows = st.selectbox("Mostrar últimas filas:", [10,25,50,100])
        st.dataframe(logged_data.tail(show_rows))
        st.download_button("📥 Descargar CSV", logged_data.to_csv(index=False), "monitoring_log.csv")

else:
    st.warning("⚠️ Aún no hay datos. Genera predicciones para iniciar.")
