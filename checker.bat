@echo off

echo.
echo [1/5] Esecuzione ISORT (Ordinamento import)...
isort .

echo.
echo [2/5] Esecuzione BLACK (Formattazione automatica)...
black .

echo.
echo [3/5] Esecuzione FLAKE8 (Controllo stile)...
flake8 src/ main.py

echo.
echo [4/5] Esecuzione MYPY (Controllo tipi)...
mypy src/ main.py

echo.
echo [5/5] Esecuzione PYLINT (Analisi statica)...
pylint src/ main.py

echo.
echo   Tutti i controlli sono stati eseguiti
pause