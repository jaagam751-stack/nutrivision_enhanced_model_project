#!/usr/bin/env bash
# One-command setup + run (macOS / Linux):  bash run_quick.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-training.txt
python quick_start.py "$@"
