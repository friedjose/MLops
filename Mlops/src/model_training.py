import pandas as pd
import joblib
import matplotlib.pyplot as plt
from google.cloud import bigquery
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# ============================
# 1. Load data from BigQuery
# ============================

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
# 2. Train & evaluate model
# ============================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    # ✅ Train predictions
    y_pred_train = model.predict(X_train)

    # ✅ Test predictions
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train),
        "train_recall": recall_score(y_train, y_pred_train),
        "train_precision": precision_score(y_train, y_pred_train),

        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test)
    }

    return metrics, model

# ============================
# 3. Main 
# ============================

if __name__ == "__main__":
    
    print("📥 Cargando datos desde BigQuery...")
    df = load_from_bigquery()

    target = "membresia_premium"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Bagging": BaggingClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            learning_rate=0.05,
            n_estimators=400,
            max_depth=5
        )
    }

    results = {}
    trained = {}

    print("\n🚀 Entrenando modelos...")
    for name, model in models.items():
        print(f"➡ {name}")
        metrics, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        trained[name] = trained_model

    df_results = pd.DataFrame(results).T
    print("\n📊 Resultados:")
    print(df_results)

    # Plot test metrics only for visualization
    df_results[[c for c in df_results.columns if "test" in c]].plot(kind="bar", figsize=(10,6))
    plt.title("Comparación Modelos Ensamble (Test Metrics)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    best_model_name = df_results.sort_values(by="test_f1", ascending=False).index[0]
    best_model = trained[best_model_name]

    print(f"\n🏆 Mejor modelo: {best_model_name}")

    joblib.dump(best_model, "best_model.pkl")
    print("✅ Modelo guardado como best_model.pkl")

