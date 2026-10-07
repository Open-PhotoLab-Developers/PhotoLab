@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv (
    echo 创建虚拟环境...
    python -m venv .venv
)
.venv\Scripts\python -c "import PySide6, numpy, cv2" 2>nul
if errorlevel 1 (
    echo 安装依赖...
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python main.py
if errorlevel 1 pause
