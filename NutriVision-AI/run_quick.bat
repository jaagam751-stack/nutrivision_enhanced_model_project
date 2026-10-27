@echo off
REM One-click setup + run (Windows). Double-click or run from the NutriVision-AI folder.
cd /d %~dp0
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-training.txt
python quick_start.py %*
pause
