# 💎 Predicción de Membresía Premium – Proyecto MLOps

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20Backend-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![BigQuery](https://img.shields.io/badge/BigQuery-Datos-blue?logo=googlebigquery)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-red?logo=jenkins)
![EvidentlyAI](https://img.shields.io/badge/EvidentlyAI-Data%20Drift-orange)
![MLOps](https://img.shields.io/badge/MLOps-End%20to%20End-success)

---

## 📘 Descripción del Proyecto

Este proyecto implementa un flujo **MLOps completo** para predecir qué **clientes de restaurantes** tienen mayor probabilidad de adquirir una **membresía premium**.  

El enfoque se centra en construir un pipeline reproducible y automatizado con **Jenkins**, garantizando:
- Control de versiones con **GitHub**  
- Validación de estructura y secretos con **PyOps**  
- Integración y despliegue continuo  
- Visualización y monitoreo del modelo con **Streamlit**
- Detección de data drift con **EvidentlyAI**

---

## ⚙️ Flujo General del Proyecto

### 1️⃣ Ingesta y Procesamiento de Datos
- Extracción directa desde **BigQuery**
- Limpieza, codificación y generación de características
- División del conjunto de datos en entrenamiento y prueba
- Variable objetivo: `membresia_premium`

### 2️⃣ Entrenamiento y Evaluación de Modelos
- Modelos probados:
  - **Random Forest**
  - **Logistic Regression**
  - **XGBoost**
  - **Modelo Heurístico (baseline)**
- Métricas principales:
  - Accuracy  
  - Precision / Recall  
  - F1-score  
  - ROC-AUC  

### 3️⃣ Despliegue y Monitoreo
- **API** de predicción desarrollada con **FastAPI**
- **Dashboard interactivo** con **Streamlit** para evaluar el rendimiento
- Monitoreo de logs en `monitoring_logs.csv` *(excluido del repositorio con .gitignore)*
- **Detección de data drift** con **EvidentlyAI**
- Validaciones automáticas de código con **PyOps**
- Automatización con **Jenkins (CI/CD)**

---

## 📂 Estructura del Repositorio
```
Restaurant-Premium-Prediction/
│
├── Mlops/
│   ├── src/
│   │   ├── cargar_datos.py
│   │   ├── comprension_eda.ipynb
│   │   ├── ft_engineering.py
│   │   ├── heuristic_model.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   ├── model_deploy.py
│   │   └── model_monitoring.py
│   └── requirements.txt
│
├── pyops/
│   ├── check_structure.py      # Verifica archivos obligatorios
│   └── check_secrets.py        # Revisa exposición de claves o tokens
│
├── Dockerfile                  # Imagen Docker de la API
├── Jenkinsfile                 # Pipeline de CI/CD automatizado
└── README.md
```

---

## 🤖 Jenkins CI/CD Pipeline

El pipeline de Jenkins automatiza la integración y validación del proyecto.

### 🔧 Etapas del Pipeline

1. **Checkout:** Clona el repositorio desde GitHub  
2. **Entorno:** Crea un entorno virtual de Python  
3. **Instalación:** Instala dependencias desde `requirements.txt`  
4. **Validaciones PyOps:**  
   - `check_structure.py` → Verifica la estructura esperada  
   - `check_secrets.py` → Revisa posibles claves o contraseñas  
5. **Ejecución de scripts base**  
6. **Notificación por Discord webhook** con el resultado (éxito o fallo)

### 📬 Notificaciones

El pipeline envía una notificación a **Discord** mediante webhook con el resumen del proceso:

- **Webhook de Discord:**  
  - ✅ Éxito → Mensaje con embed verde indicando "Pipeline MLOps completado con éxito"  
  - ❌ Error → Mensaje con embed rojo indicando "Error en Pipeline MLOps"

- **Contenido de la notificación:**  
  - Repositorio  
  - Rama  
  - Número de build  
  - Estado final  
  - Enlace a los logs del build  

---

## 🧩 PyOps – Validaciones Automáticas

### 🗂️ `check_structure.py`
Comprueba que existan los archivos esenciales del flujo MLOps.
```bash
python pyops/check_structure.py
```

### 🔒 `check_secrets.py`
Verifica que no existan tokens, contraseñas o claves expuestas en el repositorio.
```bash
python pyops/check_secrets.py
```

Estas validaciones se ejecutan automáticamente en cada build de Jenkins.

---

## 🚀 Despliegue Local con Docker

La API se ejecuta con **FastAPI** dentro de un contenedor **Docker**:
```bash
docker build -t restaurant-premium-api .
docker run -p 8000:8000 restaurant-premium-api
```

### 🌐 Endpoints Principales

| Método | Endpoint         | Descripción                                         |
| ------ | ---------------- | --------------------------------------------------- |
| `POST` | `/predict_one`   | Predice si un cliente comprará la membresía premium |
| `POST` | `/predict_batch` | Realiza predicciones para múltiples clientes        |

**Ejemplo de petición:**
```json
{
  "edad": 35,
  "ingresos_anuales": 25000,
  "frecuencia_visitas": 4,
  "calificacion_satisfaccion": 4.5,
  "tipo_cliente": "fidelizado"
}
```

---

## 📊 Dashboard de Monitoreo (Streamlit)

El dashboard permite visualizar métricas clave de desempeño:

* Accuracy, F1-score, ROC-AUC
* Distribución de errores
* Comparación entre modelos
* **Detección de drift** (cambio de distribución en los datos) con **EvidentlyAI**

**EvidentlyAI** genera reportes automáticos que detectan:
- Drift en características numéricas y categóricas
- Cambios en la distribución de la variable objetivo
- Alertas tempranas de degradación del modelo

Ejecutar:
```bash
streamlit run Mlops/src/model_monitoring.py
```

---

## 📈 Resultados del Modelo

| Modelo                | Accuracy | F1   | ROC-AUC |
| --------------------- | -------- | ---- | ------- |
| Random Forest         | 0.88     | 0.85 | 0.91    |
| Logistic Regression   | 0.83     | 0.80 | 0.87    |
| XGBoost               | 0.89     | 0.86 | 0.92    |
| Heurístico (Baseline) | 0.70     | 0.67 | 0.74    |

---

## 👨‍💻 Autor

**Proyecto Final CDP 2025 – MLOps**  
Desarrollado por **Jose Fernando Villegas Lora**

**Tecnologías utilizadas:**
* Python 🐍
* FastAPI ⚡
* Streamlit 📊
* BigQuery ☁️
* Docker 🐳
* Jenkins 🔧
* Discord Webhooks 💬
* EvidentlyAI 📉
---
⭐ *Si este proyecto te resultó útil, apóyalo con una estrella en GitHub.*
