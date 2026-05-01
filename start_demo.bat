@echo off
:: Activate MuseTalk venv and launch the gradio demo on http://127.0.0.1:7860
:: Requires weights downloaded under .\models
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py
endlocal
