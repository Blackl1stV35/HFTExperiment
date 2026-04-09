@echo off
REM Wrapper to activate venv and run training with all arguments
call .venv\Scripts\activate.bat
python scripts\train_supervised.py %*
