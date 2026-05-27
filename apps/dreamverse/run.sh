#!/usr/bin/env bash
# Linux/WSL launcher for the Dreamverse demo (mirrors run.bat).
#
# Runs backend + frontend as background processes in this terminal. Logs go
# to backend.log / frontend.log next to this script. Ctrl-C in this shell
# stops both processes via the EXIT trap.
#
# Usage:
#   ./run.sh                    backend on :8009, frontend (devtools) on :5299
#   ./run.sh --mock             use dreamverse-mock-server (no GPU, UI-dev mode)
#   ./run.sh --no-frontend      backend only
#   ./run.sh --no-browser       skip opening the browser
#
# Env overrides:
#   BE_PORT=8010 ./run.sh
#   FRONTEND_MODE=dev ./run.sh          (no devtools)
#   FRONTEND_MODE=single5s ./run.sh
#   FASTVIDEO_VENV=$HOME/fastvideo-venv path to Linux venv (default: ./.venv up two levels)
#   READY_TIMEOUT_S=300                 backend readiness poll cap (sec)
#   CEREBRAS_API_KEY=...                (LLM features off without these)
#   GROQ_API_KEY=...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- parse flags ---
MOCK=0
NO_FRONTEND=0
NO_BROWSER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)         MOCK=1;        shift ;;
        --no-frontend)  NO_FRONTEND=1; shift ;;
        --no-browser)   NO_BROWSER=1;  shift ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "WARN: ignoring unknown arg $1" >&2
            shift
            ;;
    esac
done

# --- defaults ---
BE_PORT="${BE_PORT:-8009}"
FE_PORT="${FE_PORT:-5299}"
FRONTEND_MODE="${FRONTEND_MODE:-devtools}"

# NVFP4 quantization: flashinfer has Linux wheels for cu128 but the build still
# needs a recent CUDA toolkit and the right Blackwell support, so default OFF
# for the same reason as the .bat -- bf16 fallback Just Works. Flip to 0 when
# you've verified flashinfer is installed.
DREAMVERSE_DISABLE_NVFP4="${DREAMVERSE_DISABLE_NVFP4:-1}"
export DREAMVERSE_DISABLE_NVFP4

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${FASTVIDEO_VENV:-$REPO_ROOT/.venv}"
WEB="$SCRIPT_DIR/web"

# --- pre-flight checks ---
if [[ "$MOCK" == "1" ]]; then
    BE_EXE="$VENV/bin/dreamverse-mock-server"
else
    BE_EXE="$VENV/bin/dreamverse-server"
fi
if [[ ! -x "$BE_EXE" ]]; then
    echo "ERROR: $BE_EXE not found" >&2
    echo "Did you install fastvideo[dreamverse] into $VENV?" >&2
    echo "Try: uv pip install --python $VENV -e \"$REPO_ROOT[dreamverse]\"" >&2
    exit 2
fi
if [[ "$NO_FRONTEND" == "0" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
        echo "ERROR: npm not on PATH (install Node.js)" >&2
        exit 2
    fi
    if [[ ! -d "$WEB/node_modules" ]]; then
        echo "=== installing frontend deps (one-time) ==="
        ( cd "$WEB" && npm ci ) || { echo "ERROR: npm ci failed" >&2; exit 1; }
    fi
fi

# --- env-var warnings ---
[[ -z "${CEREBRAS_API_KEY:-}" ]] && echo "WARN: CEREBRAS_API_KEY not set (LLM-routed features will be disabled)"
[[ -z "${GROQ_API_KEY:-}" ]] && echo "WARN: GROQ_API_KEY not set (LLM-routed features will be disabled)"

# --- map FRONTEND_MODE -> npm script + env ---
case "${FRONTEND_MODE,,}" in
    devtools)
        FE_ENV_KEY="NEXT_PUBLIC_INCLUDE_DEVTOOLS"; FE_ENV_VAL="1"; FE_NPM="dev" ;;
    dev)
        FE_ENV_KEY=""; FE_ENV_VAL=""; FE_NPM="dev" ;;
    single5s)
        FE_ENV_KEY="NEXT_PUBLIC_PRODUCT_MODE"; FE_ENV_VAL="single5s"; FE_NPM="dev" ;;
    *)
        echo "ERROR: FRONTEND_MODE must be devtools|dev|single5s (got $FRONTEND_MODE)" >&2
        exit 2
        ;;
esac

echo "============================================================"
echo "Dreamverse demo"
echo "============================================================"
echo "  backend     : $BE_EXE --port $BE_PORT"
echo "  backend URL : http://localhost:$BE_PORT"
echo "  NVFP4       : DREAMVERSE_DISABLE_NVFP4=$DREAMVERSE_DISABLE_NVFP4  (1=skip (bf16 fallback), 0=use NVFP4)"
if [[ "$NO_FRONTEND" == "0" ]]; then
    echo "  frontend    : npm run $FE_NPM (mode=$FRONTEND_MODE)"
    echo "  frontend URL: http://localhost:$FE_PORT"
fi
echo "============================================================"

BE_LOG="$SCRIPT_DIR/backend.log"
FE_LOG="$SCRIPT_DIR/frontend.log"
: > "$BE_LOG"
: > "$FE_LOG"

# Track pids so the EXIT trap can stop them.
BE_PID=""
FE_PID=""
cleanup() {
    echo
    echo "Stopping background processes..."
    if [[ -n "$FE_PID" ]] && kill -0 "$FE_PID" 2>/dev/null; then
        kill -- -"$FE_PID" 2>/dev/null || kill "$FE_PID" 2>/dev/null || true
    fi
    if [[ -n "$BE_PID" ]] && kill -0 "$BE_PID" 2>/dev/null; then
        kill -- -"$BE_PID" 2>/dev/null || kill "$BE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- launch backend ---
echo "Launching backend -> $BE_LOG"
( "$BE_EXE" --port "$BE_PORT" >>"$BE_LOG" 2>&1 ) &
BE_PID=$!

# --- wait for backend /readyz ---
READY_TIMEOUT_S="${READY_TIMEOUT_S:-300}"
echo
echo "Waiting for backend /readyz on :$BE_PORT (up to ${READY_TIMEOUT_S}s) ..."
BE_OK=0
for ((i=1; i<=READY_TIMEOUT_S; i++)); do
    if ! kill -0 "$BE_PID" 2>/dev/null; then
        echo "ERROR: backend process exited; see $BE_LOG"
        exit 1
    fi
    if curl -fsS -m 2 "http://localhost:$BE_PORT/readyz" >/dev/null 2>&1; then
        BE_OK=1
        echo "  /readyz 200 after ${i} s"
        break
    fi
    sleep 1
done
if [[ "$BE_OK" == "0" ]]; then
    echo
    echo "ERROR: backend /readyz did not return 200 within ${READY_TIMEOUT_S}s." >&2
    echo "  Check $BE_LOG for import / load errors." >&2
    echo "  Common causes: missing CEREBRAS_API_KEY/GROQ_API_KEY, broken venv install," >&2
    echo "  GPU OOM, or the model snapshot still downloading." >&2
    exit 1
fi

# --- launch frontend ---
if [[ "$NO_FRONTEND" == "0" ]]; then
    echo "Launching frontend -> $FE_LOG"
    (
        cd "$WEB"
        if [[ -n "$FE_ENV_KEY" ]]; then
            export "$FE_ENV_KEY"="$FE_ENV_VAL"
        fi
        npm run "$FE_NPM" >>"$FE_LOG" 2>&1
    ) &
    FE_PID=$!

    if [[ "$NO_BROWSER" == "0" ]]; then
        # Give Next.js a couple seconds to bind before opening the tab.
        sleep 3
        url="http://localhost:$FE_PORT"
        if grep -qi microsoft /proc/version 2>/dev/null; then
            # WSL: launch the URL with the Windows default browser.
            cmd.exe /c start "$url" >/dev/null 2>&1 &
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" >/dev/null 2>&1 &
        else
            echo "  (no browser launcher found; open $url manually)"
        fi
    fi
fi

echo
echo "--- launched ---"
echo "  backend  : http://localhost:$BE_PORT  (/healthz, /readyz, /status)  [pid $BE_PID, log: $BE_LOG]"
if [[ "$NO_FRONTEND" == "0" ]]; then
    echo "  frontend : http://localhost:$FE_PORT  [pid $FE_PID, log: $FE_LOG]"
fi
echo
echo "Stop with Ctrl-C in this shell (cleanup trap will kill both)."
echo "Tail logs in another terminal with:"
echo "  tail -f \"$BE_LOG\""
[[ "$NO_FRONTEND" == "0" ]] && echo "  tail -f \"$FE_LOG\""

# Block until either child exits or the user hits Ctrl-C.
if [[ "$NO_FRONTEND" == "0" ]]; then
    wait -n "$BE_PID" "$FE_PID" 2>/dev/null || true
else
    wait "$BE_PID" 2>/dev/null || true
fi
