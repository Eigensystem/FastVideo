"""Staged smoke test for the dreamverse backend.

Designed to catch Windows-startup failure modes (fcntl import, asyncio
add_reader on ProactorEventLoop, torch.distributed NCCL backend, NVFP4
flashinfer dependency, etc.) at the earliest phase where they manifest.
Each phase runs only if the previous phase passed; on failure, the
traceback is printed and the script exits with a non-zero code that
identifies the failing phase.

Phases:
  A  Import smoke           — `import dreamverse.main` and its transitive deps. ~1s.
  B  Distributed-init smoke — calls init_distributed_environment(world_size=1)
                              directly to verify the platform-specific backend
                              selection (Windows → gloo, Linux → nccl). ~3s.
  C  CLI smoke              — runs `dreamverse-server --help` as a subprocess
                              and checks exit code. Catches argparse/wiring
                              regressions. ~10s (one-time torch import).
  D  Mock-server /healthz   — launches `dreamverse-mock-server` and polls
                              /healthz. Validates FastAPI lifespan and the
                              health-route registration. No GPU needed. ~15s.
  E  Real /readyz           — opt-in via --with-readyz. Launches the real
                              backend and polls /readyz with the configured
                              timeout (default 600s). Catches NCCL/NVFP4/model
                              load. Heavy — only run when the cache is warm.

Usage:
  python apps/dreamverse/scripts/smoke_backend.py                  # phases A-D
  python apps/dreamverse/scripts/smoke_backend.py --with-readyz    # plus E
  python apps/dreamverse/scripts/smoke_backend.py --only A,B       # subset
  python apps/dreamverse/scripts/smoke_backend.py --readyz-timeout 1800
"""

from __future__ import annotations

import argparse
import http.client
import os
import socket
import subprocess
import sys
import time
import traceback
from contextlib import closing
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
DREAMVERSE_DIR = Path(__file__).resolve().parents[1]


def _print(tag: str, msg: str) -> None:
    print(f"[smoke] {tag} {msg}", flush=True)


def _exit(code: int, phase: str, why: str) -> None:
    _print("FAIL", f"phase={phase} reason={why}")
    sys.exit(code)


def _pick_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_get(host: str, port: int, path: str, timeout: float = 2.0) -> tuple[int, str]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", errors="replace")
    finally:
        conn.close()


def _poll_endpoint(host: str, port: int, path: str, timeout_s: float,
                   poll_s: float, proc: subprocess.Popen | None,
                   expect_status: int = 200) -> tuple[bool, str]:
    """Poll <host>:<port><path> until expect_status, timeout, or proc dies."""
    deadline = time.monotonic() + timeout_s
    last_err = "no attempts made"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False, f"server exited with code {proc.returncode} before ready"
        try:
            status, body = _http_get(host, port, path, timeout=2.0)
            if status == expect_status:
                return True, body
            last_err = f"status={status} body={body[:120]!r}"
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            last_err = type(e).__name__ + f": {e}"
        time.sleep(poll_s)
    return False, f"timeout after {timeout_s}s — last={last_err}"


# ---------- Phase A: import smoke ----------

def phase_a_import() -> None:
    """Walk the import graph the same way `dreamverse-server` does at startup.

    Catches: fcntl/posix-only imports, missing top-level deps. Won't catch
    runtime-only failures (NCCL, model load).
    """
    _print("A", "importing dreamverse.main and transitive modules")
    # Order chosen to mirror what server_entry.cli does. If any of these
    # carry an unguarded posix-only import, the failure is here.
    import dreamverse.av_streaming  # noqa: F401  fcntl was the gotcha here
    import dreamverse.gpu_pool  # noqa: F401  asyncio.add_reader was here
    import dreamverse.video_generation  # noqa: F401  pulls fastvideo.entrypoints
    import dreamverse.main  # noqa: F401  full server module
    import dreamverse.server_entry  # noqa: F401  CLI entry
    _print("A", "ok — all dreamverse modules import cleanly")


# ---------- Phase B: distributed init smoke ----------

def phase_b_distributed() -> None:
    """Force init_process_group at world_size=1 to verify backend selection.

    Catches: NCCL on Windows, gloo path regressions, master-addr/port issues.
    """
    _print("B", "validating distributed init at world_size=1")
    # Run in a subprocess because torch.distributed has process-global state
    # and we don't want to dirty the smoke runner.
    code = (
        "import os, sys\n"
        "os.environ.setdefault('MASTER_ADDR', '127.0.0.1')\n"
        "os.environ.setdefault('MASTER_PORT', '0')\n"
        "from fastvideo.distributed.parallel_state import init_distributed_environment\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    print('SKIP: no CUDA device', flush=True)\n"
        "    sys.exit(0)\n"
        "init_distributed_environment(world_size=1, rank=0, local_rank=0,\n"
        "                              distributed_init_method='tcp://127.0.0.1:0')\n"
        "import torch.distributed as dist\n"
        "be = dist.get_backend() if dist.is_initialized() else 'uninitialized'\n"
        "print(f'OK backend={be} platform={sys.platform}', flush=True)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=60)
    if res.returncode != 0:
        _print("B", f"stdout={res.stdout!r}")
        _print("B", f"stderr={res.stderr!r}")
        _exit(2, "B", f"distributed init failed (rc={res.returncode})")
    _print("B", f"ok — {res.stdout.strip()}")


# ---------- Phase C: CLI smoke ----------

def phase_c_cli() -> None:
    """Run `dreamverse-server --help` and check exit code."""
    _print("C", "invoking dreamverse-server --help")
    # Resolve the entry from the active interpreter rather than relying on PATH.
    res = subprocess.run(
        [sys.executable, "-m", "dreamverse.server_entry", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        _print("C", f"stdout={res.stdout[-500:]!r}")
        _print("C", f"stderr={res.stderr[-500:]!r}")
        _exit(3, "C", f"--help exited rc={res.returncode}")
    if "usage: " not in res.stdout.lower() and "usage: " not in res.stderr.lower():
        _exit(3, "C", "--help did not produce usage banner")
    _print("C", "ok — CLI loads + prints usage")


# ---------- Phase D: mock server /healthz ----------

def phase_d_mock_health(host: str = "127.0.0.1", timeout_s: float = 30.0) -> None:
    """Launch mock server, poll /healthz, kill it."""
    # Pre-flight: both mock + real server require ffmpeg for fMP4 muxing.
    # av_streaming.py: shutil.which(os.getenv("FASTVIDEO_FFMPEG_BIN", "ffmpeg")).
    # Failing here with a specific message is more useful than letting the
    # mock server spawn and crash inside its FastAPI lifespan.
    import shutil as _shutil
    ffmpeg_bin = os.environ.get("FASTVIDEO_FFMPEG_BIN", "ffmpeg")
    if _shutil.which(ffmpeg_bin) is None:
        _exit(4, "D",
              f"ffmpeg not found on PATH (looked for {ffmpeg_bin!r}). "
              "Install via `winget install ffmpeg` / `choco install ffmpeg` / "
              "apt-get / brew, or set FASTVIDEO_FFMPEG_BIN to an explicit path.")
    port = _pick_free_port()
    _print("D", f"launching dreamverse-mock-server on {host}:{port}")
    env = os.environ.copy()
    # Strip any inherited venv pointers that could send the subprocess to a
    # different interpreter.
    for k in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    # mock_server.py only exposes --latency and --port; it always binds 0.0.0.0
    # internally (see uvicorn.run(... host="0.0.0.0", ...)), so we poll on
    # 127.0.0.1 regardless of the `host` argument passed to us.
    proc = subprocess.Popen(
        [sys.executable, "-m", "dreamverse.mock_server", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        ok, info = _poll_endpoint(host, port, "/healthz", timeout_s=timeout_s,
                                   poll_s=1.0, proc=proc)
        if not ok:
            # Surface a tail of the server's own log so we can see why it
            # didn't start.
            tail = ""
            if proc.stdout is not None:
                try:
                    proc.terminate()
                    out, _ = proc.communicate(timeout=5)
                    tail = (out or "")[-1200:]
                except Exception:
                    pass
            _print("D", f"server tail:\n{tail}")
            _exit(4, "D", info)
        _print("D", f"ok — /healthz returned 200 ({len(info)}b)")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------- Phase E: real /readyz (opt-in) ----------

def phase_e_real_readyz(host: str = "127.0.0.1", timeout_s: float = 600.0) -> None:
    """Launch real backend, poll /readyz. Heavy — needs warm model cache."""
    port = _pick_free_port()
    _print("E", f"launching dreamverse-server on {host}:{port} (timeout={timeout_s}s)")
    env = os.environ.copy()
    for k in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    # Give the worker some headroom for model download/load.
    env.setdefault("DREAMVERSE_INIT_TIMEOUT_S", str(max(int(timeout_s * 2), 1200)))
    proc = subprocess.Popen(
        [sys.executable, "-m", "dreamverse.server_entry", "--port", str(port),
         "--host", host],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        ok, info = _poll_endpoint(host, port, "/readyz", timeout_s=timeout_s,
                                   poll_s=3.0, proc=proc)
        if not ok:
            tail = ""
            if proc.stdout is not None:
                try:
                    proc.terminate()
                    out, _ = proc.communicate(timeout=10)
                    tail = (out or "")[-2500:]
                except Exception:
                    pass
            _print("E", f"server tail:\n{tail}")
            _exit(5, "E", info)
        _print("E", "ok — /readyz returned 200")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


PHASES: dict[str, Callable[[], None]] = {
    "A": phase_a_import,
    "B": phase_b_distributed,
    "C": phase_c_cli,
    "D": phase_d_mock_health,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Comma-separated subset (e.g. A,B,C)")
    p.add_argument("--with-readyz", action="store_true",
                   help="Also run phase E (heavy real-server /readyz poll)")
    p.add_argument("--readyz-timeout", type=float, default=600.0,
                   help="Phase E /readyz timeout seconds (default 600)")
    p.add_argument("--mock-timeout", type=float, default=30.0,
                   help="Phase D /healthz timeout seconds (default 30)")
    args = p.parse_args()

    phases = list(PHASES.keys())
    if args.only:
        phases = [c.strip().upper() for c in args.only.split(",") if c.strip()]
        unknown = [c for c in phases if c not in PHASES and c != "E"]
        if unknown:
            print(f"[smoke] unknown phases: {unknown}", file=sys.stderr)
            return 1

    started = time.monotonic()
    for code in phases:
        if code == "E":
            continue  # E only via --with-readyz
        fn = PHASES[code]
        t0 = time.monotonic()
        try:
            if code == "D":
                phase_d_mock_health(timeout_s=args.mock_timeout)
            else:
                fn()
        except SystemExit:
            raise
        except Exception:
            _print(code, "FAIL with exception:")
            traceback.print_exc()
            _exit(10 + ord(code) - ord("A"), code, "unhandled exception")
        _print(code, f"phase took {time.monotonic() - t0:.1f}s")

    if args.with_readyz:
        t0 = time.monotonic()
        try:
            phase_e_real_readyz(timeout_s=args.readyz_timeout)
        except SystemExit:
            raise
        except Exception:
            _print("E", "FAIL with exception:")
            traceback.print_exc()
            _exit(15, "E", "unhandled exception")
        _print("E", f"phase took {time.monotonic() - t0:.1f}s")

    _print("PASS", f"all phases ok in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
