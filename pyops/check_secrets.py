import os
import re
import sys

# Patrones de posibles secretos
SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z_\-]{35}",          # GCP API Key
    r"-----BEGIN PRIVATE KEY-----",    # LLave privada
    r"(?i)(secret|password|token|apikey|access_key)\s*[:=]\s*[\'\"]?[A-Za-z0-9_\-]{8,}[\'\"]?",
]

EXCLUDED_DIRS = {"venv", ".git", "__pycache__", "pyops"}

def file_contains_secret(file_path):
    try:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()
            for pattern in SECRET_PATTERNS:
                if re.search(pattern, content):
                    return pattern
    except:
        return None
    return None

def scan_project():
    for root, dirs, files in os.walk("."):
        # excluir carpetas
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if file.endswith((".py", ".ipynb", ".json", ".yaml", ".env")):
                full_path = os.path.join(root, file)
                match = file_contains_secret(full_path)
                if match:
                    print(f"❌ Se encontró posible secreto en {full_path}")
                    print(f"   Patrón: {match}")
                    return False
    return True

if __name__ == "__main__":
    ok = scan_project()
    if ok:
        print("✅ No se encontraron secretos expuestos")
        sys.exit(0)
    else:
        print("🔴 ERROR: Se detectaron posibles secretos en el repositorio")
        sys.exit(1)
