@echo off
setlocal
cd /d "%~dp0.."
set "PY_CMD="
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3"
if not defined PY_CMD (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)
if not defined PY_CMD for %%V in (314 313 312) do (
    if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe""
)
if not defined PY_CMD goto :fail
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" %PY_CMD% -m venv "%CD%\.venv"
if errorlevel 1 goto :fail
cd /d "%~dp0"
"%VENV_PY%" -c "import django, pandas, numpy, sklearn, joblib, openpyxl, entsoe, congestion_nl; assert sklearn.__version__ == '1.9.1'" >nul 2>&1
if errorlevel 1 "%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
"%VENV_PY%" -m congestion_nl fetch-drive --missing
if errorlevel 1 echo Some shared files could not be downloaded; the app starts without them.
set "PIPELINE_PYTHON=%VENV_PY%"
set "PYTHONDONTWRITEBYTECODE=1"
"%VENV_PY%" manage.py check >nul 2>&1
if errorlevel 1 "%VENV_PY%" manage.py check
if errorlevel 1 goto :fail
curl -s -o nul http://localhost:9110/
if errorlevel 1 start "Congestion Intelligence" "%VENV_PY%" manage.py runserver 0.0.0.0:9110 --noreload --insecure
set "TRIES=0"
:wait
curl -s -o nul http://localhost:9110/
if not errorlevel 1 goto :open
set /a TRIES+=1
if %TRIES% GEQ 60 goto :fail
ping -n 2 127.0.0.1 >nul
goto :wait
:open
start "" http://localhost:9110/
exit /b 0
:fail
pause
exit /b 1
