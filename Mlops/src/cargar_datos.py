import json
from google.cloud import bigquery
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

project_id = config["proyecto"]  # Make sure you fill it in config.json

client = bigquery.Client(project=project_id)

query = f"SELECT * FROM `{project_id}.proyecto_final_cdp.restaurantes_prediccion`"
df = client.query(query).to_dataframe()

print(df.head())
