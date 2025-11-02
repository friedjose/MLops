import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from google.cloud import bigquery


# =====================================
# Load GCP Project from config.json
# =====================================
def load_project():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["proyecto"]


# =====================================
# Save DataFrame to BigQuery
# =====================================
def save_to_bigquery(df, table_id):
    project_id = load_project()  # read from config
    client = bigquery.Client(project=project_id)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    print(f"Processed data uploaded to BigQuery: {table_id}")


# =====================================
# 1. Cleaning Function 
# =====================================
def clean_data(df):
    df = df.copy()

    # ---- Fix Edad ----
    edad_median = df['edad'].median()
    df.loc[(df['edad'] < 15) | (df['edad'] > 90), 'edad'] = edad_median

    # ---- Fix Frecuencia de visita ----
    df['frecuencia_visita'] = df['frecuencia_visita'].apply(lambda x: np.nan if x < 0 else x)
    freq_median = df['frecuencia_visita'].median()
    df['frecuencia_visita'] = df['frecuencia_visita'].fillna(freq_median)

    # Fill remaining numeric nulls
    num_cols = ["edad", "frecuencia_visita", "promedio_gasto_comida", "ingresos_mensuales"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical nulls with 'Desconocido'
    df["preferencias_alimenticias"] = df["preferencias_alimenticias"].fillna("Desconocido")

    return df


# =====================================
# 2. Build Pipeline
# =====================================
def build_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
            ("num", MinMaxScaler(), numeric_features)
        ]
    )
    return Pipeline([("preprocessor", preprocessor)])


# =====================================
# 3. Train/Test Split
# =====================================
def split_data(df, target="membresia_premium"):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# =====================================
# 4. Main
# =====================================
def main(df):
    df = clean_data(df)

    target = "membresia_premium"
    X = df.drop(columns=[target])
    y = df[target]

    categorical = [
        "genero", "ciudad_residencia", "estrato_socioeconomico",
        "ocio", "consume_licor", "preferencias_alimenticias", "tipo_de_pago_mas_usado"
    ]
    numeric = ["edad", "frecuencia_visita", "promedio_gasto_comida", "ingresos_mensuales"]

    pipeline = build_pipeline(categorical, numeric)

    X_trans = pipeline.fit_transform(X)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    X_df = pd.DataFrame(
        X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans,
        columns=feature_names
    )
    df_processed = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    project_id = load_project()
    table_id = f"{project_id}.proyecto_final_cdp.restaurantes_features"
    save_to_bigquery(df_processed, table_id)

    X_train, X_test, y_train, y_test = split_data(df_processed)

    print("Feature Engineering Completed")
    print("Train:", X_train.shape, " | Test:", X_test.shape)


if __name__ == "__main__":
    from cargar_datos import load_data
    df = load_data()
    main(df)

