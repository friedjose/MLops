@echo off
setlocal EnableDelayedExpansion

echo
echo === Python Virtual Environment Setup ===
echo

REM Ensure script runs from project root
cd /d %~dp0

REM Verify config exists
if not exist Mlops\src\config.json (
    echo ERROR: config.json not found
    exit /b 1
)

REM Extract project_code using PowerShell JSON parser
for /f %%A in ('powershell -NoProfile -Command "(Get-Content Mlops/src/config.json | ConvertFrom-Json).project_code"') do (
    set project_code=%%A
)

echo Project code detected: %project_code%

REM Create venv
py -m venv %project_code%-venv
call %project_code%-venv\Scripts\activate

echo === Installing requirements ===
pip install --upgrade pip
pip install --no-cache-dir -r Mlops\requirements.txt

echo === Registering kernel ===
python -m ipykernel install --user --name=%project_code%-venv --display-name="%project_code%-venv"

echo DONE ✅ Environment ready!
echo Use kernel: %project_code%-venv
pause
