@echo off
:: Dreamverse install pipeline for Windows. Idempotent — re-runnable.
::
:: Steps (each is skipped if already done):
::   1. ensure venv at FastVideo\.venv\
::   2. editable fastvideo[dreamverse] install (uvicorn, cerebras-cloud-sdk, openai)
::   3. hf_xet for accelerated HuggingFace downloads
::   4. optional flashinfer (NVFP4) — only when --with-fp4 is passed, needed
::      for FastVideo/LTX2-Distilled-Diffusers (NVFP4-quantized weights)
::   5. smoke validation (scripts\smoke_backend.py phases A,B,C)
::
:: Usage:
::   setup.bat                       base install + smoke test (no FP4)
::   setup.bat --with-fp4            also install flashinfer for FP4 models
::   setup.bat --skip-fastvideo      skip step 2 (use existing fastvideo install)
::   setup.bat --skip-smoke          skip step 5
::
:: Notes:
::   - flashinfer's build backend uses os.symlink, which requires Windows
::     Developer Mode unless run as admin. Enable at:
::       Settings -^> Privacy ^& Security -^> For developers -^> Developer Mode
::   - The venv lives at the FastVideo repo root, not under apps\dreamverse,
::     because the [dreamverse] extra is a fastvideo project extra.

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

:: --- parse args ---
set INSTALL_FP4=0
set SKIP_FASTVIDEO=0
set SKIP_SMOKE=0
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--with-fp4"       ( set INSTALL_FP4=1     & shift & goto parse )
if /I "%~1"=="--skip-fastvideo" ( set SKIP_FASTVIDEO=1  & shift & goto parse )
if /I "%~1"=="--skip-smoke"     ( set SKIP_SMOKE=1      & shift & goto parse )
if /I "%~1"=="--help"           goto :help
if /I "%~1"=="-h"               goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: Strip inherited venv pointers so uv resolves cleanly.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

:: Compute repo root (apps\dreamverse\..\.. = FastVideo root).
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"
set "VENV=%REPO_ROOT%\.venv"

where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv not on PATH. Install from https://docs.astral.sh/uv/
    exit /b 2
)

echo ============================================================
echo Dreamverse setup
echo ============================================================
echo   repo      : %REPO_ROOT%
echo   venv      : %VENV%
echo   FP4       : %INSTALL_FP4%  ^(--with-fp4 to enable^)
echo   skip-fv   : %SKIP_FASTVIDEO%
echo   skip-smoke: %SKIP_SMOKE%
echo ============================================================

:: --- 1/5 venv ---
echo.
if exist "%VENV%\Scripts\python.exe" (
    echo --- 1/5  venv already exists ---
) else (
    echo --- 1/5  creating venv ---
    uv venv "%VENV%" --python 3.11
    if errorlevel 1 ( echo FAIL venv create & exit /b 1 )
)
set "PY=%VENV%\Scripts\python.exe"

:: --- 2/5 fastvideo + [dreamverse] base extras ---
echo.
if "%SKIP_FASTVIDEO%"=="1" (
    echo --- 2/5  skipped fastvideo install ^(--skip-fastvideo^) ---
) else (
    echo --- 2/5  editable fastvideo + [dreamverse] base extras ---
    uv pip install --python "%PY%" -e "%REPO_ROOT%[dreamverse]"
    if errorlevel 1 ( echo FAIL fastvideo install & exit /b 1 )
)

:: --- 3/5 hf_xet ---
echo.
echo --- 3/5  hf_xet ^(accelerated HF downloads^) ---
uv pip install --python "%PY%" hf_xet
if errorlevel 1 ( echo WARN: hf_xet install failed; downloads will be slower )

:: --- 4/5 optional FP4 ---
echo.
if "%INSTALL_FP4%"=="1" (
    echo --- 4/5  flashinfer ^(NVFP4 support^) ---
    echo NOTE: requires Windows Developer Mode for symlink permissions.
    uv pip install --python "%PY%" flashinfer-python
    if errorlevel 1 (
        echo.
        echo FAIL: flashinfer install failed.
        echo   Most common cause: Windows symlink privilege ^(WinError 1314^).
        echo   Fix: enable Developer Mode at
        echo     Settings -^> Privacy ^& Security -^> For developers -^> Developer Mode
        echo   then re-run setup.bat --with-fp4.
        echo   Alternative: run setup.bat without --with-fp4 to skip; non-FP4
        echo                models still work, FastVideo/LTX2-Distilled-Diffusers will not.
        exit /b 1
    )
) else (
    echo --- 4/5  skipped FP4 ^(--with-fp4 to enable^) ---
    echo NOTE: Without flashinfer, FastVideo/LTX2-Distilled-Diffusers will fail
    echo       to load ^(NVFP4 weights need flashinfer kernels^). Pass --with-fp4
    echo       to install ^(needs Windows Developer Mode^).
)

:: --- 5/5 smoke validation ---
echo.
if "%SKIP_SMOKE%"=="1" (
    echo --- 5/5  skipped smoke ^(--skip-smoke^) ---
) else (
    echo --- 5/5  smoke test ^(phases A,B,C^) ---
    "%PY%" "%~dp0scripts\smoke_backend.py" --only A,B,C
    if errorlevel 1 (
        echo FAIL: smoke phases A-C reported a regression.
        echo   Inspect the output above; common causes are missing Windows
        echo   portability patches in fastvideo\distributed\parallel_state.py
        echo   ^(NCCL-^>gloo^) or apps\dreamverse\dreamverse\av_streaming.py
        echo   ^(fcntl import guard^).
        exit /b 1
    )
)

echo.
echo ============================================================
echo Dreamverse setup complete.
echo Next:  run.bat                     full backend + frontend
echo        run.bat --mock              UI only, no GPU
echo        run.bat --no-frontend       backend only
echo ============================================================
exit /b 0


:help
echo.
echo setup.bat -- Dreamverse install pipeline ^(Windows^)
echo.
echo Args:
echo   --with-fp4         install flashinfer for NVFP4-quantized models
echo                      ^(needs Windows Developer Mode for symlinks^)
echo   --skip-fastvideo   skip editable fastvideo install
echo   --skip-smoke       skip the phase A,B,C smoke validation
echo.
echo Env overrides: none.
exit /b 0
