@echo off
cd /d %~dp0
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.11 or newer first.
  pause
  exit /b 1
)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\update_data.py
if errorlevel 1 echo Public data update was partial or unavailable; cached data will be used.
python scripts\build_site.py
start http://localhost:8000
python -m http.server 8000 -d public
