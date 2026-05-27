@echo off
:: run_cosmos.bat -- launch Cosmos-Predict2.5 inference examples with the
:: Windows-specific env tweaks the upstream example files don't set.
:: Mirrors run_matrixgame2.bat.
::
:: Usage:
::   run_cosmos.bat                       text-to-world (default)
::   run_cosmos.bat --t2w                 text-to-world
::   run_cosmos.bat --i2w                 image-to-world
::   run_cosmos.bat --v2w                 video-to-world (continuation)
::   run_cosmos.bat --no-offline          allow HF Hub network fetch
::                                        (default: HF_HUB_OFFLINE=1)
::
:: Required pre-reqs:
::   1) FastVideo .venv exists with torch + fastvideo editable installed
::      (see setup.bat).
::   2) The Cosmos-Predict2.5-2B HF snapshot is in the cache. If missing,
::      pre-fetch with --no-offline once, then subsequent runs default to
::      offline (works around the Py3.12 atexit thread-join bug).

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "!VENV_PY!" (
    echo ERROR: venv python not found: !VENV_PY!
    echo Run setup.bat first to create the venv and install fastvideo.
    exit /b 1
)

:: Strip ambient venv state so the spawned interpreter doesn't graft
:: a different venv's stdlib path onto this one (the SRE / _sre mismatch).
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

:: --- Windows gloo fix ---
:: torch.distributed's gloo backend can't auto-pick a network adapter on
:: Windows. Set GLOO_SOCKET_IFNAME so any code path that still hits gloo
:: has a fallback.
if not defined GLOO_SOCKET_IFNAME set "GLOO_SOCKET_IFNAME=Wi-Fi"

:: --- Windows shard-load stability ---
:: transformers' async/threaded shard loader segfaults on Windows + sm_120
:: mid-shard (0xC0000005 ACCESS_VIOLATION at meta-init -> param copy).
set "HF_DEACTIVATE_ASYNC_LOAD=1"

:: hf_transfer's extra mmap'd download buffers compete with the
:: diffusion-checkpoint mmap for Windows virtual address space. Off.
set "HF_HUB_ENABLE_HF_TRANSFER=0"

:: torch.distributed on Windows: PyTorch wheels are built without libuv.
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"

:: HF Hub on Py3.12 + Windows: snapshot_download's ThreadPoolExecutor
:: atexit handler raises RuntimeError("cannot join thread before it is
:: started") and re-wraps the real error into a misleading "model not
:: found". OFFLINE mode skips the network step entirely so no executor
:: is created. Pre-req: model must already be in cache (use --no-offline
:: once to populate).
set "OFFLINE=1"

:: UTF-8 stdio for any emoji prints (matches HY-WorldPlay / matrixgame2).
set "PYTHONIOENCODING=utf-8"

:: --- arg parse ---
set "MODE=t2w"
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--t2w"        ( set "MODE=t2w" & shift & goto parse )
if /I "%~1"=="--i2w"        ( set "MODE=i2w" & shift & goto parse )
if /I "%~1"=="--v2w"        ( set "MODE=v2w" & shift & goto parse )
if /I "%~1"=="--no-offline" ( set "OFFLINE=0" & shift & goto parse )
if /I "%~1"=="--help"       goto :help
if /I "%~1"=="-h"           goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

if "%OFFLINE%"=="1" (
    set "HF_HUB_OFFLINE=1"
) else (
    set "HF_HUB_OFFLINE="
)

:: --- mode -> example file ---
if /I "%MODE%"=="t2w" set "EXAMPLE=examples\inference\basic\basic_cosmos2_5_t2w.py"
if /I "%MODE%"=="i2w" set "EXAMPLE=examples\inference\basic\basic_cosmos2_5_i2w.py"
if /I "%MODE%"=="v2w" set "EXAMPLE=examples\inference\basic\basic_cosmos2_5_v2w.py"

if not exist "%~dp0!EXAMPLE!" (
    echo ERROR: example not found: %~dp0!EXAMPLE!
    exit /b 2
)

echo ============================================================
echo FastVideo Cosmos-Predict 2.5 inference
echo ============================================================
echo   python   : !VENV_PY!
echo   mode     : !MODE!  ^(t2w^|i2w^|v2w^)
echo   example  : %~dp0!EXAMPLE!
echo   offline  : %OFFLINE%
echo   gloo if  : !GLOO_SOCKET_IFNAME!
echo ============================================================

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_START=%%T"
echo Started:  %DATE% %TIME%

"!VENV_PY!" -X utf8 "%~dp0!EXAMPLE!"
set "EXIT_CODE=!ERRORLEVEL!"

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_END=%%T"
echo Finished: %DATE% %TIME%
set /a TOTAL_S=T_END-T_START
echo Elapsed   : %TOTAL_S%s

if not !EXIT_CODE!==0 (
    echo ERROR: example exited with code !EXIT_CODE!
    if "%OFFLINE%"=="1" (
        echo Hint: model may not be cached. Try once with --no-offline to fetch:
        echo   run_cosmos.bat --!MODE! --no-offline
    )
    exit /b !EXIT_CODE!
)

echo.
echo --- done ---
echo videos at: %~dp0outputs_video\
exit /b 0

:help
echo Usage:
echo   run_cosmos.bat                       text-to-world ^(default^)
echo   run_cosmos.bat --t2w                 text-to-world
echo   run_cosmos.bat --i2w                 image-to-world
echo   run_cosmos.bat --v2w                 video-to-world ^(continuation^)
echo   run_cosmos.bat --no-offline          allow HF Hub network fetch
echo.
echo Env overrides:
echo   GLOO_SOCKET_IFNAME   network adapter name ^(default: Wi-Fi^)
echo.
echo Model: KyleShao/Cosmos-Predict2.5-2B-Diffusers
echo Output: outputs_video\cosmos2_5_^<mode^>.mp4
exit /b 0
