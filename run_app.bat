@echo off
title SMS Spam Detector - Web Application
echo ==========================================================
echo Starting SMS Spam Detection Dashboard...
echo ==========================================================
cd /d "%~dp0"
call ".\venv\Scripts\activate.bat"
streamlit run app.py
pause
