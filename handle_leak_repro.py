"""Isolation test for the Windows kernel-handle leak that crashes the gradio gui.

The full failure mode in gradio_local_demo_matrixgame2.py was:
    Browser connects -> anyio worker threads spawn -> threads do C-extension
    work -> ~970K kernel handles allocated per thread per 10 sec -> process
    hits 16.7M handle cap -> threading.Condition.wait()'s _allocate_lock()
    fails -> every thread crashes with RuntimeError("can't allocate lock").

This script reproduces the same execution shape (anyio worker pool driven by
async tasks) but layers C-extension calls progressively, so we can localize
*which* layer leaks. Run each mode, watch the handle counter:

    python handle_leak_repro.py --mode plain      # no C work; expected: flat
    python handle_leak_repro.py --mode cuda       # CUDA stream/event per task
    python handle_leak_repro.py --mode shmem      # mmap'd shared memory per task
    python handle_leak_repro.py --mode fastvideo  # full FastVideo step_async path

If a mode shows linear handle growth across iterations, that mode contains the
leaking primitive. The mode that DOES leak isolates the bug to its specific
C-extension call chain.

Why this matters: on Windows the per-process handle cap is 2^24 ~= 16.7M.
Once a process hits it, ALL threading.Condition / threading.Lock allocations
across the whole process fail -- and on a busy host that cascades into
"can't allocate lock" / "cannot join thread before it is started" errors in
*other* unrelated processes too, until the leaker is killed.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
import time
from typing import Callable

try:
    import anyio.to_thread
    import psutil
except ImportError as e:
    print(f"FATAL: missing dependency ({e}). Install psutil + anyio first.", file=sys.stderr)
    sys.exit(2)

_PROC = psutil.Process(os.getpid())


def _snapshot(label: str) -> None:
    """Print a one-line resource snapshot (handles + threads + RSS)."""
    handles = _PROC.num_handles()
    threads = _PROC.num_threads()
    rss = _PROC.memory_info().rss / (1024**3)
    print(f"  [{label}] handles={handles:,}  threads={threads}  rss={rss:.2f}GB", flush=True)


# --- mode: plain ---
# Pure-Python work: read a tiny int, return it. Establishes a baseline for
# the cost of the anyio worker-thread plumbing alone (Queue.get/put, the
# Condition wait/notify pairs). If THIS leaks, the bug is in Python stdlib
# or anyio itself.
def _work_plain(i: int) -> int:
    return i * 2


# --- mode: cuda ---
# Each task creates a small CUDA tensor and forces a stream-sync. Each tensor
# alloc uses cudaMalloc; each stream-sync may register a cudaEvent. On
# Windows, cudaEvent handles count against the per-process handle table, so
# leaking events looks identical to the gradio symptom.
def _work_cuda(i: int):
    import torch
    if not torch.cuda.is_available():
        return None
    # Force a kernel + sync so an event/stream is implicitly used.
    x = torch.zeros(64, device="cuda")
    x = x + i
    torch.cuda.synchronize()
    return x.sum().item()


# --- mode: shmem ---
# Each task allocates a small shared-memory segment via multiprocessing's
# shared_memory module, then closes it. Tests whether multiprocessing's
# SHM layer leaks SECTION handles on Windows under the anyio thread pool
# (a plausible MultiprocExecutor leak source).
def _work_shmem(i: int):
    from multiprocessing import shared_memory
    name = f"hlk_{os.getpid()}_{i}_{time.monotonic_ns()}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=1024)
    try:
        shm.buf[0] = i % 255
    finally:
        shm.close()
        shm.unlink()


# --- mode: fastvideo ---
# Exercises the full FastVideo StreamingVideoGenerator -> MultiprocExecutor
# IPC path that the gradio demo uses. Heaviest mode; loads the actual model.
# If this leaks but `cuda` and `shmem` don't, the bug is in FastVideo's
# per-step IPC marshalling.
_FV_GENERATOR = None


def _work_fastvideo(i: int):
    """Run a single FastVideo step. Generator is cached at module level."""
    import torch
    global _FV_GENERATOR
    if _FV_GENERATOR is None:
        from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
        from fastvideo.models.dits.matrixgame2.utils import expand_action_to_frames
        _FV_GENERATOR = {
            "gen": StreamingVideoGenerator.from_pretrained(
                "FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers",
                num_gpus=1,
                use_fsdp_inference=False,
                dit_layerwise_offload=False,
                dit_cpu_offload=True,
                vae_cpu_offload=False,
                text_encoder_cpu_offload=True,
                pin_cpu_memory=True,
            ),
            "expand": expand_action_to_frames,
        }
        # First step requires a reset() with an initial image. Use a tiny
        # solid-color image to keep this isolation test light.
        gen = _FV_GENERATOR["gen"]
        # Skip reset() complexity — if there's no public no-op step API we
        # leave this branch as "not implemented" and rely on the other modes.
        raise NotImplementedError(
            "fastvideo mode would need a full reset() with image_path; for now "
            "use the gradio demo with the health.log monitor instead."
        )


_MODES: dict[str, Callable[[int], object]] = {
    "plain": _work_plain,
    "cuda": _work_cuda,
    "shmem": _work_shmem,
    "fastvideo": _work_fastvideo,
    # Phase-1 localization modes (handled specially in _driver):
    "pinned": _work_plain,        # plain work, but pinned to 1 shared worker
    "bare-thread": _work_plain,   # plain work, no anyio at all
}


async def _driver_anyio(mode: str, iterations: int, snapshot_every: int,
                         concurrency: int, pinned_limiter: object | None = None) -> None:
    """Fire `iterations` work items through anyio's worker-thread pool.

    `concurrency` matches the 6 simultaneous anyio worker threads that
    gradio's request handlers spawn in the wild crash.

    `pinned_limiter`: if non-None, anyio forces ALL run_sync calls onto a
    single shared worker thread (no per-call spawning). This is the key
    A/B for Phase 1 localization.
    """
    work = _MODES[mode]
    _snapshot("baseline")

    sem = asyncio.Semaphore(concurrency)

    async def _one(i: int) -> None:
        async with sem:
            if pinned_limiter is not None:
                await anyio.to_thread.run_sync(work, i, limiter=pinned_limiter)
            else:
                await anyio.to_thread.run_sync(work, i)

    for i in range(iterations):
        await _one(i)
        if (i + 1) % snapshot_every == 0:
            _snapshot(f"iter {i + 1}")

    await asyncio.sleep(0.2)
    gc.collect()
    _snapshot("final")


def _driver_bare_thread(iterations: int, snapshot_every: int, concurrency: int) -> None:
    """Stdlib-only reproducer: no anyio, no asyncio.

    Spawns `concurrency` workers each pulling from a shared queue.Queue. If
    THIS leaks too, the bug is in CPython 3.12's threading internals, not
    anyio. If it doesn't, the leak lives in anyio's WorkerThread spawn path.
    """
    import queue
    import threading

    work = _work_plain
    q: queue.Queue = queue.Queue()
    done = queue.Queue()
    stop_sentinel = object()

    def _worker():
        while True:
            item = q.get()
            if item is stop_sentinel:
                return
            done.put(work(item))

    _snapshot("baseline")

    workers = [threading.Thread(target=_worker, name=f"bare-{i}", daemon=True)
               for i in range(concurrency)]
    for t in workers:
        t.start()
    _snapshot("workers-spawned")

    for i in range(iterations):
        q.put(i)
        done.get()  # synchronous: 1 in, 1 out -- matches the anyio shape
        if (i + 1) % snapshot_every == 0:
            _snapshot(f"iter {i + 1}")

    for _ in workers:
        q.put(stop_sentinel)
    for t in workers:
        t.join()
    gc.collect()
    _snapshot("final")


async def _driver(mode: str, iterations: int, snapshot_every: int, concurrency: int) -> None:
    """Dispatch to the right driver based on mode."""
    if mode == "bare-thread":
        # Sync driver — exits cleanly to caller. Caller wraps everything in
        # asyncio.run() for the other modes, but bare-thread doesn't need it.
        _driver_bare_thread(iterations, snapshot_every, concurrency)
        return

    pinned_limiter = None
    if mode == "pinned":
        # CapacityLimiter(1) forces a single shared worker across ALL calls.
        # If anyio's WorkerThread spawn-and-die is the leak source, this mode
        # should be flat.
        pinned_limiter = anyio.CapacityLimiter(1)

    await _driver_anyio(mode, iterations, snapshot_every, concurrency, pinned_limiter)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(_MODES.keys()), default="plain",
                   help="Which workload to put through the anyio worker pool.")
    p.add_argument("--iterations", type=int, default=200,
                   help="Total work items to dispatch (default 200).")
    p.add_argument("--snapshot-every", type=int, default=20,
                   help="Print a handle/thread snapshot every N iterations.")
    p.add_argument("--concurrency", type=int, default=6,
                   help="Max concurrent anyio worker calls (default 6 -- "
                        "matches gradio's wild crash).")
    args = p.parse_args()

    print(f"=== handle_leak_repro mode={args.mode} iterations={args.iterations} "
          f"concurrency={args.concurrency} ===", flush=True)
    # bare-thread mode skips asyncio.run entirely so even the asyncio event
    # loop's own handle allocations don't pollute the measurement.
    if args.mode == "bare-thread":
        _driver_bare_thread(args.iterations, args.snapshot_every, args.concurrency)
    else:
        asyncio.run(_driver(args.mode, args.iterations, args.snapshot_every, args.concurrency))
    print("=== done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
