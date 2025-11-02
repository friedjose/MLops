import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import json
import os


# =====================================================
# Helper function
# =====================================================
def load_project():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["proyecto"]


# =====================================================
# 1. Heuristic Model
# =====================================================
class HeuristicModel(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 gasto_threshold=80,
                 frecuencia_threshold=4,
                 edad_min=18):
        self.gasto_threshold = gasto_threshold
        self.frecuencia_threshold = frecuencia_threshold
        self.edad_min = edad_min

    def fit(self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        preds = []

        premium_cities = ["Miami", "San Diego", "Seattle"]

        for _, row in X.iterrows():

            # =====================
            # Reglas determinísticas fuertes
            # =====================
            if row.get("cat__estrato_socioeconomico_Muy Alto", 0) == 1:
                preds.append(1)
                continue
            if row.get("cat__estrato_socioeconomico_Bajo", 0) == 1:
                preds.append(0)
                continue

            score = 0

            # Estrato socioeconómico
            if row.get("cat__estrato_socioeconomico_Alto", 0) == 1:
                score += 2
            if row.get("cat__estrato_socioeconomico_Medio", 0) == 1:
                score -= 1

            # Ciudades asociadas a premium
            if any(row.get(f"cat__ciudad_residencia_{c}", 0) == 1 for c in premium_cities):
                score += 1

            # Gasto promedio comida
            if row["num__promedio_gasto_comida"] >= self.gasto_threshold:
                score += 1
            elif row["num__promedio_gasto_comida"] < self.gasto_threshold * 0.5:
                score -= 1

            # Frecuencia de visita
            if row["num__frecuencia_visita"] >= self.frecuencia_threshold:
                score += 1
            elif row["num__frecuencia_visita"] < 1:
                score -= 1

            # Edad inválida / mínima
            if row["num__edad"] < self.edad_min:
                score -= 2

            # Método de pago
            if row.get("cat__tipo_de_pago_mas_usado_Efectivo", 0) == 1:
                score += 1
            if row.get("cat__tipo_de_pago_mas_usado_Tarjeta", 0) == 1:
                score += 0.5

            preds.append(1 if score >= 1 else 0)

        return np.array(preds)


# =====================================================
# 2. Load processed dataset from BigQuery
# =====================================================
from google.cloud import bigquery

project_id = load_project()
table_id = f"{project_id}.proyecto_final_cdp.restaurantes_features"

print(f"📥 Loading data from BigQuery: {table_id}")

client = bigquery.Client(project=project_id)
query = f"""
SELECT *
FROM `{table_id}`
"""
df = client.query(query).to_dataframe()

print(f"✅ Dataset cargado: {df.shape}")
print(f"📊 Columnas disponibles: {len(df.columns)}")

# =====================================================
# CONVERTIR TARGET A NUMÉRICO
# =====================================================
print(f"\n🔄 Valores originales de membresia_premium: {df['membresia_premium'].unique()}")

# Mapear 'Sí' -> 1, 'No' -> 0
df['membresia_premium'] = df['membresia_premium'].map({'Sí': 1, 'No': 0})

print(f"✅ Valores convertidos: {df['membresia_premium'].unique()}")
print(f"🎯 Distribución de membresia_premium:")
print(df['membresia_premium'].value_counts())

X = df.drop(columns=["membresia_premium"])
y = df["membresia_premium"]

# Split estratificado
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📦 Train set: {x_train.shape}")
print(f"📦 Test set: {x_test.shape}")
print(f"🎯 Distribución target en train:")
print(y_train.value_counts(normalize=True))


# =====================================================
# 3. Validación cruzada
# =====================================================
print("\n" + "="*60)
print("🚀 ENTRENAMIENTO Y VALIDACIÓN CRUZADA")
print("="*60)

model = HeuristicModel()
pipe = Pipeline([("model", model)])

metrics = ["accuracy", "f1", "precision", "recall"]
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Cross-validation scores
print("\n⏳ Calculando scores de validación cruzada...")
cv_results = {m: cross_val_score(pipe, x_train, y_train, cv=kfold, scoring=m) for m in metrics}

# Entrenar el modelo
pipe.fit(x_train, y_train)

# Calcular train scores correctamente
y_train_pred = pipe.predict(x_train)
train_scores = {
    "accuracy": accuracy_score(y_train, y_train_pred),
    "f1": f1_score(y_train, y_train_pred),
    "precision": precision_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred)
}

cv_df = pd.DataFrame(cv_results)

print("\n===== Cross-Validation Scores (Mean ± Std) =====")
for metric in metrics:
    print(f"{metric.capitalize()}: {cv_df[metric].mean():.4f} ± {cv_df[metric].std():.4f}")

print("\n===== Train Scores =====")
for metric, score in train_scores.items():
    print(f"{metric.capitalize()}: {score:.4f}")

print("\n===== Test Scores =====")
y_test_pred = pipe.predict(x_test)
test_scores = {
    "accuracy": accuracy_score(y_test, y_test_pred),
    "f1": f1_score(y_test, y_test_pred),
    "precision": precision_score(y_test, y_test_pred),
    "recall": recall_score(y_test, y_test_pred)
}
for metric, score in test_scores.items():
    print(f"{metric.capitalize()}: {score:.4f}")


# =====================================================
# 4. Boxplot CV
# =====================================================
plt.figure(figsize=(10, 6))
cv_df.plot.box(title="Cross Validation Scores", ylabel="Score")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# 5. Train vs CV scores
# =====================================================
means = cv_df.mean()
stds = cv_df.std()
x_pos = np.arange(len(metrics))

plt.figure(figsize=(10, 6))
width = 0.35
plt.bar(x_pos - width/2, [train_scores[m] for m in metrics], width, label="Train", alpha=0.8)
plt.bar(x_pos + width/2, means, width, yerr=stds, capsize=5, label="CV (Mean ± Std)", alpha=0.8)
plt.xticks(x_pos, [m.capitalize() for m in metrics])
plt.title("Train vs Cross-Validation Scores")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# 6. Confusion Matrix
# =====================================================
y_pred = pipe.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Premium", "Premium"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Heuristic Model (Test Set)")
plt.tight_layout()
plt.show()

print("\n===== Confusion Matrix Analysis =====")
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")


# =====================================================
# 7. Learning Curve
# =====================================================
print("\n⏳ Generando Learning Curve...")
train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    pipe,
    x_train,
    y_train,
    cv=kfold,
    scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores_lc.mean(axis=1)
train_std = train_scores_lc.std(axis=1)
test_mean = test_scores_lc.mean(axis=1)
test_std = test_scores_lc.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-", label="Train F1", linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, test_mean, "o-", label="CV F1", linewidth=2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.xlabel("Training Examples")
plt.ylabel("F1 Score")
plt.title("Learning Curve - Heuristic Model")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*60)