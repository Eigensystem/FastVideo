#!/usr/bin/env bash
# run_matrixgame2.sh -- WSL/Linux companion to run_matrixgame2.bat.
# Launches examples/inference/basic/basic_matrixgame2.py (or streaming / gradio
# variants) under the FastVideo venv with env tweaks that the example files
# themselves don't set.
#
# Usage:
#   ./run_matrixgame2.sh                       base one-shot (default)
#   ./run_matrixgame2.sh --variant templerun   Temple Run variant
#   ./run_matrixgame2.sh --streaming           interactive WASD streaming (stdin)
#   ./run_matrixgame2.sh --gui                 Gradio web UI at http://localhost:7860
#   ./run_matrixgame2.sh --gui --port 8080     custom Gradio port
#
# Required pre-reqs:
#   1) FastVideo Linux venv exists at ./.venv/bin/python with torch + fastvideo
#      editable installed.  Override via FASTVIDEO_VENV_PY env var.
#   2) The corresponding HF snapshot is in the cache.  When invoked from WSL
#      against a Windows-side HF cache, point HF_HOME at it, e.g.
#      export HF_HOME=/mnt/c/Users/$USER/.cache/huggingface
#      (slow 9p reads but avoids re-downloading the ~tens of GB models).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY="${FASTVIDEO_VENV_PY:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found: $VENV_PY" >&2
    echo "Create the venv first, or set FASTVIDEO_VENV_PY." >&2
    exit 1
fi

# Strip ambient venv state so the spawned interpreter doesn't graft a different
# venv's stdlib path onto this one (the SRE/_sre module-mismatch crash that hits
# when the host shell already has VIRTUAL_ENV pointing at another env).
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

# Most Windows-specific env tweaks from run_matrixgame2.bat are intentionally
# NOT set here:
#   * GLOO_SOCKET_IFNAME -- gloo auto-picks on Linux; only set if you must pin.
#   * HF_DEACTIVATE_ASYNC_LOAD -- transformers async shard loader is stable on
#     Linux; leave unset for speed.
#   * HF_HUB_ENABLE_HF_TRANSFER=0 -- mmap-VA competition is a Windows problem;
#     on Linux hf_transfer is fine (and faster) if installed.
#   * HF_HUB_OFFLINE=1 -- the Python 3.12 + atexit-thread-join bug is Windows-
#     specific.  Online resolution works on Linux.
#   * USE_LIBUV=0 / TORCH_TCPSTORE_USE_LIBUV=0 -- Linux torch wheels are built
#     with libuv; the default is correct.

# arg parse
MODE="basic"
VARIANT="base_distilled_model"
GUI_PORT="7860"

usage() {
    cat <<'EOF'
Usage:
  ./run_matrixgame2.sh                       base one-shot (default)
  ./run_matrixgame2.sh --variant templerun   Temple Run variant
  ./run_matrixgame2.sh --streaming           interactive WASD streaming demo
  ./run_matrixgame2.sh --gui                 Gradio web UI at http://localhost:7860
  ./run_matrixgame2.sh --gui --port 8080     custom Gradio port

Variants:    base | templerun     (gta removed)
Modes:       basic (default) | streaming | gui

Env:
  FASTVIDEO_VENV_PY   override path to python (default: ./.venv/bin/python)
  HF_HOME             HuggingFace cache root (point at /mnt/c/... to reuse the
                      Windows-side cache from WSL)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)   VARIANT="$2"; shift 2 ;;
        --basic)     MODE="basic"; shift ;;
        --streaming) MODE="streaming"; shift ;;
        --gui)       MODE="gui"; shift ;;
        --port)      GUI_PORT="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "ERROR: unknown arg $1" >&2; exit 2 ;;
    esac
done

# variant aliases
case "$VARIANT" in
    base)              VARIANT="base_distilled_model" ;;
    temple|templerun)  VARIANT="templerun_distilled_model" ;;
esac

if [[ "$VARIANT" != "base_distilled_model" && "$VARIANT" != "templerun_distilled_model" ]]; then
    echo "ERROR: --variant must be one of: base | templerun (got: $VARIANT)" >&2
    echo "Note: gta variant has been removed from this launcher." >&2
    exit 2
fi

# per-mode example + GUI model mapping.  basic/streaming hardcode MODEL_VARIANT
# at module level (so we patch in-memory before exec); the gradio demo uses
# argparse with hyphenated HF-style names.
case "$MODE" in
    basic)
        EXAMPLE="examples/inference/basic/basic_matrixgame2.py"
        ;;
    streaming)
        EXAMPLE="examples/inference/basic/basic_matrixgame2_streaming.py"
        ;;
    gui)
        EXAMPLE="examples/inference/gradio/local/gradio_local_demo_matrixgame2.py"
        case "$VARIANT" in
            base_distilled_model)       GUI_MODEL="Matrix-Game-2.0-Base" ;;
            templerun_distilled_model)  GUI_MODEL="Matrix-Game-2.0-TempleRun" ;;
        esac
        ;;
esac

if [[ ! -f "$SCRIPT_DIR/$EXAMPLE" ]]; then
    echo "ERROR: example not found: $SCRIPT_DIR/$EXAMPLE" >&2
    exit 2
fi

echo "============================================================"
echo "FastVideo matrixgame2 inference"
echo "============================================================"
echo "  python   : $VENV_PY"
echo "  mode     : $MODE"
echo "  example  : $SCRIPT_DIR/$EXAMPLE"
echo "  variant  : $VARIANT"
[[ "$MODE" == "gui" ]] && echo "  port     : $GUI_PORT (open http://localhost:$GUI_PORT/)"
echo "============================================================"

T_START=$(date +%s)
echo "Started:  $(date)"

EXIT_CODE=0
if [[ "$MODE" == "gui" ]]; then
    "$VENV_PY" -X utf8 "$SCRIPT_DIR/$EXAMPLE" --model "$GUI_MODEL" --host 127.0.0.1 --port "$GUI_PORT" || EXIT_CODE=$?
else
    "$VENV_PY" -X utf8 -c "import pathlib; p=pathlib.Path(r'$SCRIPT_DIR/$EXAMPLE'); src=p.read_text(encoding='utf-8'); patched=src.replace('MODEL_VARIANT = \"base_distilled_model\"', 'MODEL_VARIANT = \"$VARIANT\"'); g={'__name__':'__main__','__file__':str(p)}; exec(compile(patched, str(p), 'exec'), g)" || EXIT_CODE=$?
fi

T_END=$(date +%s)
echo "Finished: $(date)"
echo "Elapsed   : $((T_END - T_START))s"

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "ERROR: example exited with code $EXIT_CODE" >&2
    exit "$EXIT_CODE"
fi

echo
echo "--- done ---"
if [[ "$MODE" == "gui" ]]; then
    echo "Gradio shut down."
else
    echo "videos at: $SCRIPT_DIR/video_samples_matrixgame2/"
fi
