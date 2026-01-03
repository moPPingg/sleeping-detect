@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting Drowsiness Detection System...
python Runsystem.py

pause

