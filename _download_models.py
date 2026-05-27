"""Predownload FastVideo model snapshots into the HuggingFace cache.

Driven by download_models.bat.  Resolves a friendly short name -> full
HF repo id, then calls snapshot_download for each.  Storage lands in the
default HF cache (`%USERPROFILE%\\.cache\\huggingface\\hub`) so dreamverse
and game-streaming-poc pick it up at runtime with no further config.

Usage examples (from the .bat):
    download_models.bat                       # default: ltx2 only
    download_models.bat --all                 # all five
    download_models.bat ltx2 ltx23            # subset by short name
    download_models.bat --list                # print the registry

Run safely: snapshot_download resumes partial files.  max_workers=1 avoids
the Windows + py3.10/3.11 thread-shutdown race on Ctrl-C.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import snapshot_download


# Short name -> (list_of_repo_ids, used_by, audio).  Most entries are a
# single repo; Matrix-Game-2.0 is grouped as one bundle because
# game-streaming-poc selects between its three variants via --variant
# rather than fetching them as distinct shorts.
# `audio` describes what the entry contributes to the audio side of a
# pipeline: "bundled" = audio_vae + vocoder ship inside the same repo,
# "v2a" = standalone video->audio model, "" = silent / image-only.
REGISTRY: dict[str, tuple[list[str], str, str]] = {
    "ltx2":    (["FastVideo/LTX2-Distilled-Diffusers"],         "dreamverse (default)",      "bundled"),
    "ltx23":   (["FastVideo/LTX-2.3-Distilled-Diffusers"],      "dreamverse",                "bundled"),
    "mg":      ([
                    "FastVideo/Matrix-Game-2.0-Base-Diffusers",
                    "FastVideo/Matrix-Game-2.0-GTA-Diffusers",
                    "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers",
                ], "game-streaming-poc (legacy non-distilled, all 3 variants)", ""),
    # Distilled Matrix-Game-2.0 variants — what basic_matrixgame2.py and
    # fastvideo/registry.py expect by default. Pick only the variant you
    # need; no reason to download all three unless you're switching between
    # them at runtime.
    "mg-base":   (["FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"],      "basic_matrixgame2.py (universal/WASD)", ""),
    "mg-gta":    (["FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers"],       "basic_matrixgame2.py (gta_drive)",      ""),
    "mg-temple": (["FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers"], "basic_matrixgame2.py (templerun)",      ""),
    "wan22":   (["Wan-AI/Wan2.2-TI2V-5B-Diffusers"],            "HY-WorldPlay WAN pipeline", ""),
    # NOTE: MMAudio (hkchengrex/MMAudio) is intentionally NOT in this
    # registry.  FastVideo references it only from the eval harness
    # (fastvideo/eval/metrics/audio/desync, clap_score, audiobox_aesthetics,
    # kl_divergence) to score video-audio sync / quality; there is no
    # from_pretrained path that loads MMAudio as a generative V2A backbone.
    # Its Synchformer checkpoint is fetched lazily by the eval metric on
    # first use, so no predownload step is needed.
}

DEFAULT_SET = ["ltx2"]
ALL_SET = list(REGISTRY)


def list_registry() -> None:
    print("FastVideo model registry:")
    print(f"  {'short':10s}  {'repo':55s}  {'audio':8s}  used_by")
    for short, (repos, who, audio) in REGISTRY.items():
        first = repos[0] if repos else ""
        extra = f"  (+{len(repos) - 1} bundled)" if len(repos) > 1 else ""
        print(f"  {short:10s}  {first:55s}  {audio:8s}  {who}{extra}")
        for r in repos[1:]:
            print(f"  {'':10s}  {r}")


def download(short_names: list[str]) -> int:
    failures: list[tuple[str, str]] = []
    # Flatten short_names into (short, repo) work items so a bundled short
    # like 'mg' triggers all three Matrix-Game variant downloads in one run.
    work: list[tuple[str, str, str]] = []  # (short, repo, who)
    for short in short_names:
        if short not in REGISTRY:
            print(f"unknown short name {short!r}; skipping.", flush=True)
            failures.append((short, "unknown short name"))
            continue
        repos, who, _audio = REGISTRY[short]
        for r in repos:
            work.append((short, r, who))
    for i, (short, repo, who) in enumerate(work, 1):
        print(f"\n[{i}/{len(work)}] {short}  ->  {repo}  (used by {who})", flush=True)
        try:
            path = snapshot_download(repo, max_workers=1)
            print(f"   ok   {path}", flush=True)
        except Exception as e:
            print(f"   FAIL {type(e).__name__}: {e}", flush=True)
            failures.append((f"{short}:{repo}", str(e)))
    if failures:
        print("\nSome downloads failed:")
        for short, err in failures:
            print(f"  {short}: {err}")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Predownload FastVideo HF model snapshots into the HF cache.")
    p.add_argument("names", nargs="*", help="Short names from --list (default: ltx2)")
    p.add_argument("--all", action="store_true", help="Download every model in the registry")
    p.add_argument("--list", action="store_true", help="Print the registry and exit")
    args = p.parse_args()

    if args.list:
        list_registry()
        return 0

    if args.all:
        targets = ALL_SET
    elif args.names:
        targets = args.names
    else:
        targets = DEFAULT_SET

    default_hf_home = os.path.expanduser("~/.cache/huggingface")
    hf_home = os.environ.get("HF_HOME") or default_hf_home
    print(f"Targets: {targets}")
    print(f"HF cache: {hf_home}")
    return download(targets)


if __name__ == "__main__":
    sys.exit(main())
