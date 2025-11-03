import os
import sys

required_files = [
    "Mlops/src/cargar_datos.py",
    "Mlops/src/comprension_eda.ipynb",
    "Mlops/src/ft_engineering.py",
    "Mlops/src/model_training.py",
    "Mlops/src/model_deploy.py",
    "Mlops/src/model_evaluation.py",
    "Mlops/src/model_monitoring.py",
    "Mlops/src/heuristic_model.py",
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    print("❌ Faltan archivos requeridos:")
    for f in missing:
        print("-", f)
    sys.exit(1)

print("✅ Archivos del proyecto verificados")
