@echo off
setlocal enableextensions enabledelayedexpansion

REM Predownload FastVideo HF snapshots so the dreamverse / game-streaming
REM workers don't have to fetch on first call (~25-30 GB per model).
REM Files land in the default HF cache (%USERPROFILE%\.cache\huggingface\hub)
REM where every FastVideo runtime in this tree picks them up automatically.
REM
REM Usage:
REM   download_models.bat                  ltx2 only (dreamverse default)
REM   download_models.bat --all            every registered model
REM   download_models.bat ltx2 ltx23       subset by short name
REM   download_models.bat --list           print the registry
REM
REM Env override:
REM   FASTVIDEO_DL_PY   python.exe to use (default: this repo's .venv)
REM   HF_HUB_DISABLE_SYMLINKS_WARNING=1 is set to silence the Windows
REM   symlink fallback warning that fires on every snapshot_download.

set "REPO_ROOT=%~dp0"
set "DL_SCRIPT=%REPO_ROOT%_download_models.py"

if not defined FASTVIDEO_DL_PY  set "FASTVIDEO_DL_PY=%REPO_ROOT%.venv\Scripts\python.exe"

if not exist "!FASTVIDEO_DL_PY!" (
    echo [download_models] ERROR: python not found at !FASTVIDEO_DL_PY!
    echo Set FASTVIDEO_DL_PY to a python.exe with huggingface_hub installed,
    echo or create the venv with: uv venv --python 3.11 .venv
    exit /b 1
)
if not exist "!DL_SCRIPT!" (
    echo [download_models] ERROR: helper not found at !DL_SCRIPT!
    exit /b 1
)

REM Strip inherited venv state in case the calling shell has a different
REM venv activated (per the cross-venv-spawn rule on Windows).
set VIRTUAL_ENV=
set PYTHONHOME=
set PYTHONPATH=
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo [download_models] python: !FASTVIDEO_DL_PY!
echo [download_models] args  : %*
echo.

"!FASTVIDEO_DL_PY!" "!DL_SCRIPT!" %*
set "EXIT_CODE=!ERRORLEVEL!"
if not "!EXIT_CODE!"=="0" (
    echo [download_models] failed with exit code !EXIT_CODE!
    exit /b !EXIT_CODE!
)

echo.
echo [download_models] done.
endlocal
