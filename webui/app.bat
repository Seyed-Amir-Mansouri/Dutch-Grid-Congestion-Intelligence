@echo off
setlocal
cd /d "%~dp0"
set "PORT=9110"
set "PYTHONDONTWRITEBYTECODE=1"
set "PY=%LOCALAPPDATA%\Python\handy\Scripts\python.exe"
if not exist "%PY%" goto :fail
if not defined PIPELINE_PYTHON set "PIPELINE_PYTHON=%PY%"
"%PY%" manage.py check >nul 2>&1
if errorlevel 1 "%PY%" manage.py check
if errorlevel 1 goto :fail
curl -s -o nul http://localhost:%PORT%/
if errorlevel 1 start "Congestion Intelligence" "%PY%" manage.py runserver 127.0.0.1:%PORT% --noreload --insecure
set "TRIES=0"
:wait
curl -s -o nul http://localhost:%PORT%/
if not errorlevel 1 goto :open
set /a TRIES+=1
if %TRIES% GEQ 60 goto :fail
ping -n 2 127.0.0.1 >nul
goto :wait
:open
start "" http://localhost:%PORT%/
exit /b 0
:fail
pause
exit /b 1
