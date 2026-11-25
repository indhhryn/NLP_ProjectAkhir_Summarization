@echo off
setlocal
if "%NUMBA_CACHE_DIR%"=="" set NUMBA_CACHE_DIR=%TEMP%\numba_cache
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment belum tersedia. Jalankan run_windows.bat terlebih dahulu.
  pause
  goto end
)
call .venv\Scripts\activate.bat
streamlit run app.py
:end
endlocal
