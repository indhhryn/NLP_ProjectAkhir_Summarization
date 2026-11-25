@echo off
setlocal
if "%NUMBA_CACHE_DIR%"=="" set NUMBA_CACHE_DIR=%TEMP%\numba_cache
cd /d "%~dp0"
if not exist .venv (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    py -3.12 -m venv .venv
    if errorlevel 1 goto python_error
  )
)
if not exist .venv\Scripts\activate.bat goto python_error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto install_error
pip install -r requirements.txt
if errorlevel 1 goto install_error
python scripts\run_pipeline.py
if errorlevel 1 goto pipeline_error
streamlit run app.py
if errorlevel 1 goto dashboard_error
goto end
:python_error
echo Python 3.11 atau 3.12 tidak ditemukan. Instal Python 3.11/3.12 dan aktifkan Python Launcher (py).
pause
goto end
:install_error
echo Instalasi dependensi gagal. Periksa koneksi internet dan pesan error di atas.
pause
goto end
:pipeline_error
echo Pipeline gagal. Baca pesan error di atas dan bagian Troubleshooting pada README.md.
pause
goto end
:dashboard_error
echo Dashboard gagal dijalankan. Coba streamlit run app.py dari virtual environment.
pause
:end
endlocal
