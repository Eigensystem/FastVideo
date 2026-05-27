@echo off
:: run_matrixgame2.bat -- launch examples/inference/basic/basic_matrixgame2.py
:: (or its streaming / gradio variants) with the Windows-specific env tweaks
:: that the example files themselves don't set (because they ship in
:: upstream FastVideo). Mirrors the pattern used by HY-WorldPlay/run.bat.
::
:: Usage:
::   run_matrixgame2.bat                       base one-shot (default)
::   run_matrixgame2.bat --variant templerun   Temple Run variant
::   run_matrixgame2.bat --streaming           interactive WASD streaming (stdin)
::   run_matrixgame2.bat --gui                 Gradio web UI at http://localhost:7860
::   run_matrixgame2.bat --gui --port 8080     custom Gradio port
::
:: Required pre-reqs:
::   1) FastVideo .venv exists with torch + fastvideo editable installed
::      (see setup.bat).
::   2) The corresponding HF snapshot is in the cache (run download_models.bat
::      mg-base / mg-temple). The example will fetch on first use if missing.

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
:: (we patched fastvideo to use backend=fake on win+world=1, but eval
:: code paths might still touch gloo) has a fallback. Override at the
:: call site via:  set GLOO_SOCKET_IFNAME=Ethernet
if not defined GLOO_SOCKET_IFNAME set "GLOO_SOCKET_IFNAME=Wi-Fi"

:: --- Windows shard-load stability ---
:: transformers' async/threaded shard loader segfaults on Windows + sm_120
:: mid-shard (0xC0000005 ACCESS_VIOLATION at meta-init -> param copy).
:: Forcing the serial loader avoids it; matches HY-WorldPlay/run.bat fix.
set "HF_DEACTIVATE_ASYNC_LOAD=1"

:: hf_transfer's extra mmap'd download buffers compete with the
:: diffusion-checkpoint mmap for Windows virtual address space. Off.
set "HF_HUB_ENABLE_HF_TRANSFER=0"

:: torch.distributed on Windows: PyTorch wheels are built without libuv, but
:: TCPStore defaults to use_libuv=1. Disable both var names so the override
:: is version-agnostic.
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"

:: --- arg parse ---
:: MODE = basic | streaming | gui. Default gui (Gradio web UI is the
:: friendliest entrypoint; CLI modes available via --basic / --streaming).
set "MODE=gui"
set "VARIANT=base_distilled_model"
set "GUI_PORT=7860"
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--variant"   ( set "VARIANT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--streaming" ( set "MODE=streaming" & shift & goto parse )
if /I "%~1"=="--gui"       ( set "MODE=gui" & shift & goto parse )
if /I "%~1"=="--port"      ( set "GUI_PORT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--help"      goto :help
if /I "%~1"=="-h"          goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: --- variant normalization ---
:: Accept short aliases for the two supported variants.
if /I "%VARIANT%"=="base"       set "VARIANT=base_distilled_model"
if /I "%VARIANT%"=="templerun"  set "VARIANT=templerun_distilled_model"
if /I "%VARIANT%"=="temple"     set "VARIANT=templerun_distilled_model"

if not "%VARIANT%"=="base_distilled_model" if not "%VARIANT%"=="templerun_distilled_model" (
    echo ERROR: --variant must be one of: base ^| templerun ^(got: %VARIANT%^)
    echo Note: gta variant has been removed from this launcher.
    exit /b 2
)

:: --- per-mode example file + variant mapping ---
:: The basic and streaming examples use snake_case variant names
:: (base_distilled_model, templerun_distilled_model). The gradio demo uses
:: hyphenated HF-style names (Matrix-Game-2.0-Base, Matrix-Game-2.0-TempleRun)
:: and accepts --model as a CLI arg, so it doesn't need source patching.
if /I "%MODE%"=="basic" (
    set "EXAMPLE=examples\inference\basic\basic_matrixgame2.py"
) else if /I "%MODE%"=="streaming" (
    set "EXAMPLE=examples\inference\basic\basic_matrixgame2_streaming.py"
) else if /I "%MODE%"=="gui" (
    set "EXAMPLE=examples\inference\gradio\local\gradio_local_demo_matrixgame2.py"
    if /I "%VARIANT%"=="base_distilled_model"       set "GUI_MODEL=Matrix-Game-2.0-Base"
    if /I "%VARIANT%"=="templerun_distilled_model"  set "GUI_MODEL=Matrix-Game-2.0-TempleRun"
)

if not exist "%~dp0!EXAMPLE!" (
    echo ERROR: example not found: %~dp0!EXAMPLE!
    exit /b 2
)

echo ============================================================
echo FastVideo matrixgame2 inference
echo ============================================================
echo   python   : !VENV_PY!
echo   mode     : !MODE!
echo   example  : %~dp0!EXAMPLE!
echo   variant  : !VARIANT!
if /I "%MODE%"=="gui" echo   port     : !GUI_PORT! ^(open http://localhost:!GUI_PORT!/^)
echo   gloo if  : !GLOO_SOCKET_IFNAME!
echo ============================================================

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_START=%%T"
echo Started:  %DATE% %TIME%

:: Gui mode: the gradio script uses argparse, so we just invoke it directly
:: with --model and --port. No source patching needed.
:: Basic/streaming: the upstream files hardcode MODEL_VARIANT at module level,
:: so we read the source, patch it in memory, and exec under __name__='__main__'.
if /I "%MODE%"=="gui" (
    "!VENV_PY!" -X utf8 "%~dp0!EXAMPLE!" --model "!GUI_MODEL!" --host 127.0.0.1 --port !GUI_PORT!
) else (
    "!VENV_PY!" -X utf8 -c "import sys, pathlib, runpy; p=pathlib.Path(r'%~dp0!EXAMPLE!'); src=p.read_text(encoding='utf-8'); patched=src.replace('MODEL_VARIANT = \"base_distilled_model\"', 'MODEL_VARIANT = \"%VARIANT%\"'); g={'__name__':'__main__','__file__':str(p)}; exec(compile(patched, str(p), 'exec'), g)"
)
set "EXIT_CODE=!ERRORLEVEL!"

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_END=%%T"
echo Finished: %DATE% %TIME%
set /a TOTAL_S=T_END-T_START
echo Elapsed   : %TOTAL_S%s

if not !EXIT_CODE!==0 (
    echo ERROR: example exited with code !EXIT_CODE!
    exit /b !EXIT_CODE!
)

echo.
echo --- done ---
if /I "%MODE%"=="gui" (
    echo Gradio shut down.
) else (
    echo videos at: %~dp0video_samples_matrixgame2\
)
exit /b 0

:help
echo Usage:
echo   run_matrixgame2.bat                       base one-shot ^(default^)
echo   run_matrixgame2.bat --variant templerun   Temple Run variant
echo   run_matrixgame2.bat --streaming           interactive WASD streaming demo
echo   run_matrixgame2.bat --gui                 Gradio web UI at http://localhost:7860
echo   run_matrixgame2.bat --gui --port 8080     custom Gradio port
echo.
echo Variants:    base ^| templerun     ^(gta removed^)
echo Modes:       basic ^(default^) ^| streaming ^| gui
echo.
echo Env overrides:
echo   GLOO_SOCKET_IFNAME   network adapter name ^(default: Wi-Fi^)
exit /b 0
