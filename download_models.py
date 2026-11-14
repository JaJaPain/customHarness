"""
Standalone CLI Downloader for the Local Godot 4 Agentic Harness.
Downloads and extracts:
  1. Portable Windows CUDA 12.x binaries (llama-server.exe, sd.exe + CUDA DLLs) -> ./bin/
  2. Primary VLM GGUF + mmproj weights (Ornith-1.5-9B-Q6_K.gguf / Gemma-4-12B-IT) -> ./models/
  3. 2D Image-Gen GGUF weights (FLUX.1-schnell-Q5_K_S.gguf) -> ./models/
  4. 2-in-1 Audio-Gen Diffusers FP16 weights (stabilityai/stable-audio-open-1.0) -> ./models/stable-audio-open-1.0/

Usage:
  python download_models.py --check
  python download_models.py --all [--hf-token hf_xxx]
  python download_models.py --category binary --id llama-server
  python download_models.py --category llm --id ornith-9b
  python download_models.py --category image_model --id flux-schnell
  python download_models.py --category audio_model --id stable-audio-open --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

from vram_orchestrator import VRAMOrchestrator


def _print_progress(job: Dict[str, Any]) -> None:
    status = job.get("status", "")
    fname = job.get("current_file") or job.get("item_id") or ""
    pct = job.get("percent")
    dl_mb = job.get("downloaded_mb")
    tot_mb = job.get("total_mb")
    spd = job.get("speed_mbps")
    if status == "downloading":
        pct_str = f"{pct:5.1f}%" if pct is not None else "  ..."
        mb_str = f"{dl_mb:.1f}/{tot_mb:.1f} MB" if (dl_mb is not None and tot_mb) else f"{dl_mb or 0:.1f} MB"
        spd_str = f"{spd:.2f} MB/s" if spd is not None else ""
        sys.stdout.write(f"\r  [DOWNLOADING] {fname}: {pct_str} ({mb_str}) {spd_str}      ")
        sys.stdout.flush()
    elif status in ("completed", "error"):
        sys.stdout.write("\n")
        print(f"  [{status.upper()}] {job.get('message') or job.get('error')}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Download local GGUF models & CUDA binaries for the Godot 4 Harness.")
    parser.add_argument("--check", action="store_true", help="Run health check and print missing/installed files.")
    parser.add_argument("--all", action="store_true", help="Download all default binaries & models.")
    parser.add_argument("--category", choices=["binary", "llm", "image_model", "audio_model"], help="Category to download.")
    parser.add_argument("--id", dest="item_id", help="Item ID (e.g. llama-server, sd, ornith-9b, flux-schnell, stable-audio-open).")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face token for gated repos.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    orch = VRAMOrchestrator(base_dir=base_dir)

    if args.check or (not args.all and not args.category):
        report = orch.run_setup_health_check()
        print("=== Local Godot 4 Harness — Setup & Model Health Check ===")
        print("\n[1] Portable Binaries (./bin):")
        for k, v in report["binaries"].items():
            mark = "READY" if v["exists"] else "MISSING (Fallback Active)"
            print(f"  - {k:18s}: {mark} ({v['path']})")
        print("\n[2] VLM GGUF Profiles (./models):")
        for k, v in report["llm_models"].items():
            mark = f"READY ({v['model_size_mb']} MB)" if v["model_exists"] else "MISSING"
            print(f"  - {k:18s}: {mark} -> {v['model_path']}")
        print("\n[3] 2D Image-Gen GGUF Models (./models):")
        for k, v in report["image_models"].items():
            mark = f"READY ({v['size_mb']} MB)" if v["exists"] else "MISSING"
            print(f"  - {k:18s}: {mark} -> {v['path']}")
        print("\n[4] 2-in-1 Audio-Gen Models (./models):")
        for k, v in report["audio_models"].items():
            mark = "READY" if v["exists"] else "MISSING"
            print(f"  - {k:18s}: {mark} -> {v['path']} ({v['repo_id']})")
        print("\nTip: Run `python download_models.py --all` or use the '⚙ Setup & Health' button in the web UI to download.")
        return 0

    targets = []
    if args.all:
        targets = [
            ("binary", "llama-server"),
            ("binary", "sd"),
            ("llm", "ornith-9b"),
            ("image_model", "flux-schnell"),
            ("audio_model", "stable-audio-open"),
        ]
    elif args.category and args.item_id:
        targets = [(args.category, args.item_id)]
    else:
        print("Error: Provide both --category and --id, or pass --all.")
        return 1

    for cat, iid in targets:
        print(f"\n>>> Starting download for [{cat}:{iid}]...")
        res = await orch.start_download_job(
            category=cat,
            item_id=iid,
            hf_token=args.hf_token,
            progress_cb=_print_progress,
        )
        if res.get("status") == "error":
            print(f"  [!] Failed to download {cat}:{iid} -> {res.get('error')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
