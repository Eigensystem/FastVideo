@echo off
:: run_matrixgame2_gradio.bat -- launch the matrixgame2 Gradio demo with the
:: Windows-specific env tweaks the upstream example doesn't set, plus the
:: in-script handle-leak mitigations (serial queue + uvicorn loop=asyncio +
:: lifespan=off) that are already baked into the .py.
::
:: Usage:
::   run_matrixgame2_gradio.bat                 base variant, port 7860
::   run_matrixgame2_gradio.bat --variant temple
::   run_matrixgame2_gradio.bat --port 8080
::   run_matrixgame2_gradio.bat --no-offline    allow HF Hub network fetch
::
:: Variants:    base ^| templerun                (HF-style names auto-mapped)
::
:: Pre-req: HF snapshot in cache (run once with --no-offline to fetch).

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "!VENV_PY!" (
    echo ERROR: venv python not found: !VENV_PY!
    exit /b 1
)

:: Strip ambient venv state so the spawned interpreter doesn't graft
:: a different venv's stdlib path onto this one (SRE / _sre mismatch).
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

:: --- Windows env tweaks ---
:: gloo can't auto-pick a Windows NIC; nudge it at a real adapter.
if not defined GLOO_SOCKET_IFNAME set "GLOO_SOCKET_IFNAME=Wi-Fi"
:: transformers' async shard loader segfaults on Win + sm_120 mid-shard.
set "HF_DEACTIVATE_ASYNC_LOAD=1"
:: hf_transfer's mmap buffers compete with the DiT mmap for Win address space.
set "HF_HUB_ENABLE_HF_TRANSFER=0"
:: PyTorch Win wheels are built without libuv; force gloo's default off.
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"
:: UTF-8 stdio so emoji prints don't crash cp1252.
set "PYTHONIOENCODING=utf-8"

:: --- arg parse ---
set "VARIANT=base_distilled_model"
set "GUI_PORT=7860"
set "OFFLINE=1"
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--variant"    ( set "VARIANT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--port"       ( set "GUI_PORT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--no-offline" ( set "OFFLINE=0" & shift & goto parse )
if /I "%~1"=="--help"       goto :help
if /I "%~1"=="-h"           goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: HF Hub OFFLINE: works around the Py3.12 + Windows thread-join atexit bug
:: in snapshot_download. Pre-req: model already cached. Use --no-offline once
:: to fetch a new variant.
if "%OFFLINE%"=="1" (
    set "HF_HUB_OFFLINE=1"
) else (
    set "HF_HUB_OFFLINE="
)

:: --- variant aliases + HF-style name mapping ---
if /I "%VARIANT%"=="base"       set "VARIANT=base_distilled_model"
if /I "%VARIANT%"=="templerun"  set "VARIANT=templerun_distilled_model"
if /I "%VARIANT%"=="temple"     set "VARIANT=templerun_distilled_model"

if /I "%VARIANT%"=="base_distilled_model"       set "GUI_MODEL=Matrix-Game-2.0-Base"
if /I "%VARIANT%"=="templerun_distilled_model"  set "GUI_MODEL=Matrix-Game-2.0-TempleRun"
if not defined GUI_MODEL (
    echo ERROR: --variant must be one of: base ^| templerun ^(got: %VARIANT%^)
    exit /b 2
)

set "EXAMPLE=examples\inference\gradio\local\gradio_local_demo_matrixgame2.py"
if not exist "%~dp0!EXAMPLE!" (
    echo ERROR: example not found: %~dp0!EXAMPLE!
    exit /b 2
)

echo ============================================================
echo FastVideo matrixgame2 Gradio demo
echo ============================================================
echo   python   : !VENV_PY!
echo   example  : %~dp0!EXAMPLE!
echo   variant  : !VARIANT!  ^(--model !GUI_MODEL!^)
echo   port     : !GUI_PORT!  ^(open http://localhost:!GUI_PORT!/^)
echo   offline  : %OFFLINE%
echo   gloo if  : !GLOO_SOCKET_IFNAME!
echo ============================================================
echo.
echo NOTE: Windows handle-leak mitigation is baked into the .py
echo ^(serial-queue + uvicorn loop=asyncio + lifespan=off^). A long
echo session can still drift; restart the process if you see
echo "can't allocate lock". Use the basic CLI script for batch runs.
echo.

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_START=%%T"
echo Started:  %DATE% %TIME%

"!VENV_PY!" -X utf8 "%~dp0!EXAMPLE!" --model "!GUI_MODEL!" --host 127.0.0.1 --port !GUI_PORT!
set "EXIT_CODE=!ERRORLEVEL!"

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_END=%%T"
echo Finished: %DATE% %TIME%
set /a TOTAL_S=T_END-T_START
echo Elapsed   : %TOTAL_S%s

if not !EXIT_CODE!==0 (
    echo ERROR: gradio demo exited with code !EXIT_CODE!
    if "%OFFLINE%"=="1" (
        echo Hint: variant may not be cached. Try once with --no-offline:
        echo   run_matrixgame2_gradio.bat --variant %VARIANT% --no-offline
    )
    exit /b !EXIT_CODE!
)

echo.
echo --- Gradio shut down ---
exit /b 0

:help
echo Usage:
echo   run_matrixgame2_gradio.bat                 base variant, port 7860
echo   run_matrixgame2_gradio.bat --variant temple   Temple Run variant
echo   run_matrixgame2_gradio.bat --port 8080      custom port
echo   run_matrixgame2_gradio.bat --no-offline     allow HF Hub network fetch
echo.
echo Variants:  base ^| templerun
echo Env:       GLOO_SOCKET_IFNAME ^(default Wi-Fi^)
exit /b 0
