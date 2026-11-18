"""
Swap Models - move the harness's big AI models between the SSD (C:) and the hard disk (D:).

The chat model and the image models are read in full every time the harness loads them, so they
load several times faster from the SSD. This app moves them back and forth when you need the space,
and records where they are in cache/models_location.json, which the harness reads at startup.

Run it with "Swap Models.bat" (window) or from a terminal:
    python model_swap.py                 open the window
    python model_swap.py --status        print where the models are
    python model_swap.py --to ssd        move them to the SSD
    python model_swap.py --to hdd        move them to the hard disk

Moves are safe to interrupt: each file is copied, its size checked, and only then is the original
deleted, so a file is never missing from both places. Running the move again finishes the job.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "cache" / "models_location.json"
DEFAULT_SSD_DIR = Path(r"C:\HarnessModels")
HDD_DIR = BASE_DIR / "models"

# The models that load on every chat turn and every image: the chat model + its vision part, and FLUX.
# (Speech-to-text and audio stay on the hard disk; they are small or rarely used.)
SWAP_FILES = [
    "Ornith-1.5-9B-Q6_K.gguf",
    "mmproj-Ornith-1.5-9B-f16.gguf",
    "FLUX.1-schnell-Q5_K_S.gguf",
    "t5xxl_q5_k.gguf",
    "ae.safetensors",
    "clip_l.safetensors",
]
CHUNK = 16 * 1024 * 1024
SPACE_MARGIN = 2 * 1024**3  # keep at least 2 GB free on the destination


# --------------------------------------------------------------------------- state
def load_config() -> Dict[str, str]:
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg.setdefault("ssd_dir", str(DEFAULT_SSD_DIR))
    return cfg


def save_config(active: str, ssd_dir: Path) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"active": active, "ssd_dir": str(ssd_dir), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf-8",
    )


def locate(ssd_dir: Path, hdd_dir: Path = HDD_DIR) -> Dict[str, object]:
    """Where each model file is right now, and the overall location ('ssd', 'hdd', 'split' or 'missing')."""
    files = []
    for name in SWAP_FILES:
        on_ssd, on_hdd = (ssd_dir / name).exists(), (hdd_dir / name).exists()
        size = (ssd_dir / name).stat().st_size if on_ssd else (hdd_dir / name).stat().st_size if on_hdd else 0
        files.append({"name": name, "ssd": on_ssd, "hdd": on_hdd, "size": size})
    if all(f["ssd"] for f in files):
        where = "ssd"
    elif all(f["hdd"] for f in files):
        where = "hdd"
    elif any(f["ssd"] or f["hdd"] for f in files):
        where = "split"
    else:
        where = "missing"
    return {"where": where, "files": files, "total": sum(f["size"] for f in files)}


def harness_running(ssd_dir: Optional[Path] = None) -> Optional[str]:
    """
    A reason the model files are in use, or None. Only processes that belong to THIS harness count:
    one started from the harness folder, or whose command line names the harness folder or a model
    folder. Other apps on the same ports or with the same program names (Ollama, other servers) are ignored.
    """
    try:
        import psutil
    except Exception:
        return None
    markers = {str(BASE_DIR).lower(), str(HDD_DIR).lower(), str(ssd_dir or load_config()["ssd_dir"]).lower()}
    markers |= {m.replace("\\", "/") for m in list(markers)}
    # This app itself (and the .venv launcher process that started it) is not "the harness"
    me = {os.getpid()}
    try:
        me |= {parent.pid for parent in psutil.Process().parents()}
    except Exception:
        pass
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            if proc.info["pid"] in me:
                continue
            name = (proc.info.get("name") or "").lower()
            if not name.startswith(("python", "llama-server", "sd-cli", "sd.exe")):
                continue
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            cwd = (proc.info.get("cwd") or "").lower()
            if "model_swap.py" in cmd:
                continue
            uses_models = any(m in cmd for m in markers if m) or any(f.lower() in cmd for f in SWAP_FILES)
            harness_server = cwd.rstrip("\/") == str(BASE_DIR).lower() and "server.py" in cmd
            if uses_models or harness_server:
                what = "The harness" if harness_server or name.startswith("python") else proc.info.get("name")
                return f"{what} is running (process {proc.info['pid']}) and may have a model open. Close the harness first."
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue
    return None


def human(n: float) -> str:
    return f"{n / 1024**3:.1f} GB"


# --------------------------------------------------------------------------- the move
def move_models(
    to: str,
    ssd_dir: Path,
    hdd_dir: Path = HDD_DIR,
    progress: Callable[[int, int, str], None] = lambda done, total, msg: None,
    check_running: bool = True,
) -> str:
    """Move every model to `to` ('ssd' or 'hdd'), then record the location. Returns a summary line."""
    if to not in ("ssd", "hdd"):
        raise ValueError("to must be 'ssd' or 'hdd'")
    if check_running:
        reason = harness_running(ssd_dir)
        if reason:
            raise RuntimeError(reason)

    src_dir, dst_dir = (hdd_dir, ssd_dir) if to == "ssd" else (ssd_dir, hdd_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for name in SWAP_FILES:
        src, dst = src_dir / name, dst_dir / name
        if src.exists() and dst.exists() and src.stat().st_size == dst.stat().st_size:
            plan.append((src, dst, "delete-source"))  # an earlier move copied it but did not finish
        elif src.exists():
            plan.append((src, dst, "move"))
        elif not dst.exists():
            raise FileNotFoundError(f"{name} is in neither {src_dir} nor {dst_dir}.")

    to_copy = sum(s.stat().st_size for s, d, kind in plan if kind == "move")
    same_drive = os.path.splitdrive(str(src_dir.resolve()))[0].lower() == os.path.splitdrive(str(dst_dir.resolve()))[0].lower()
    if to_copy and not same_drive:
        free = shutil.disk_usage(dst_dir).free
        if free < to_copy + SPACE_MARGIN:
            raise RuntimeError(
                f"Not enough space on {dst_dir.drive or dst_dir}: the models need {human(to_copy)} "
                f"and only {human(free)} is free (keeping {human(SPACE_MARGIN)} spare)."
            )

    done = 0
    started = time.time()
    for src, dst, kind in plan:
        if kind == "delete-source":
            src.unlink()
            continue
        size = src.stat().st_size
        if same_drive:
            os.replace(src, dst)
            done += size
            progress(done, to_copy, f"Moved {src.name}")
            continue
        part = dst.with_name(dst.name + ".part")
        with open(src, "rb") as fin, open(part, "wb") as fout:
            while True:
                buf = fin.read(CHUNK)
                if not buf:
                    break
                fout.write(buf)
                done += len(buf)
                rate = done / max(0.001, time.time() - started)
                left = (to_copy - done) / rate if rate else 0
                progress(done, to_copy, f"Copying {src.name}  ·  {human(done)} of {human(to_copy)}  ·  about {left / 60:.0f} min left")
        if part.stat().st_size != size:
            part.unlink(missing_ok=True)
            raise IOError(f"Copy of {src.name} came out the wrong size; the original was left in place.")
        shutil.copystat(src, part)
        os.replace(part, dst)
        src.unlink()  # only after the copy is complete and checked

    save_config(to, ssd_dir)
    if to == "hdd":
        try:
            if ssd_dir.exists() and not any(ssd_dir.iterdir()):
                ssd_dir.rmdir()  # leave nothing behind on the SSD
        except Exception:
            pass
    label = "SSD" if to == "ssd" else "hard disk"
    return f"All models are on the {label} ({dst_dir}). Took {time.time() - started:.0f}s."


# --------------------------------------------------------------------------- window
def run_window() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    cfg = load_config()
    ssd_dir = Path(cfg["ssd_dir"])

    root = tk.Tk()
    root.title("Swap Models")
    root.resizable(False, False)
    pad = {"padx": 14, "pady": 6}

    status = tk.StringVar()
    detail = tk.StringVar()
    note = tk.StringVar()
    ttk.Label(root, text="Godot Harness - model location", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
    ttk.Label(root, textvariable=status, font=("Segoe UI", 10)).grid(row=1, column=0, columnspan=2, sticky="w", padx=14)
    ttk.Label(root, textvariable=detail, foreground="#555").grid(row=2, column=0, columnspan=2, sticky="w", padx=14)
    bar = ttk.Progressbar(root, length=420, mode="determinate")
    bar.grid(row=3, column=0, columnspan=2, **pad)
    ttk.Label(root, textvariable=note, foreground="#555", wraplength=420).grid(row=4, column=0, columnspan=2, sticky="w", padx=14)
    btn_ssd = ttk.Button(root, text="Move to SSD (C:)  -  faster")
    btn_hdd = ttk.Button(root, text="Move to hard disk (D:)  -  frees C:")
    btn_ssd.grid(row=5, column=0, sticky="ew", padx=(14, 4), pady=12)
    btn_hdd.grid(row=5, column=1, sticky="ew", padx=(4, 14), pady=12)

    def refresh() -> None:
        info = locate(ssd_dir)
        where = info["where"]
        c_free = shutil.disk_usage(ssd_dir.anchor or "C:\\").free
        text = {
            "ssd": f"Models are on the SSD  ({ssd_dir})",
            "hdd": f"Models are on the hard disk  ({HDD_DIR})",
            "split": "Models are split between both drives (an earlier move did not finish).",
            "missing": "Model files were not found on either drive.",
        }[where]
        status.set(text)
        detail.set(f"{human(info['total'])} of models  ·  {human(c_free)} free on {ssd_dir.drive or 'C:'}")
        btn_ssd.state(["disabled"] if where == "ssd" or where == "missing" else ["!disabled"])
        btn_hdd.state(["disabled"] if where == "hdd" or where == "missing" else ["!disabled"])
        if where == "split":
            note.set("Pick either button to finish moving everything to one place.")
        bar["value"] = 0

    def start(to: str) -> None:
        reason = harness_running(ssd_dir)
        if reason:
            messagebox.showwarning("Swap Models", reason)
            return
        btn_ssd.state(["disabled"])
        btn_hdd.state(["disabled"])
        note.set("Starting...")

        def report(done: int, total: int, msg: str) -> None:
            root.after(0, lambda: (bar.configure(value=100 * done / total if total else 100), note.set(msg)))

        def work() -> None:
            try:
                summary = move_models(to, ssd_dir, progress=report, check_running=False)
                root.after(0, lambda: (refresh(), note.set(summary + "  The harness will use the new location next time it starts.")))
            except Exception as exc:
                msg = str(exc)
                root.after(0, lambda: (refresh(), note.set("Move stopped: " + msg), messagebox.showerror("Swap Models", msg)))

        threading.Thread(target=work, daemon=True).start()

    btn_ssd.configure(command=lambda: start("ssd"))
    btn_hdd.configure(command=lambda: start("hdd"))
    note.set("Close the harness before moving. Moving between drives takes a few minutes.")
    refresh()
    root.mainloop()


# --------------------------------------------------------------------------- command line
def main() -> int:
    parser = argparse.ArgumentParser(description="Move the harness's AI models between the SSD and the hard disk.")
    parser.add_argument("--to", choices=["ssd", "hdd", "c", "d"], help="move the models (c = ssd, d = hdd)")
    parser.add_argument("--status", action="store_true", help="print where the models are")
    parser.add_argument("--ssd-dir", help=f"SSD folder (default {DEFAULT_SSD_DIR})")
    args = parser.parse_args()

    cfg = load_config()
    ssd_dir = Path(args.ssd_dir or cfg["ssd_dir"])

    if args.status or not args.to:
        if not args.status and not args.to:
            run_window()
            return 0
        info = locate(ssd_dir)
        print(f"Models are on: {info['where']}  ({human(info['total'])})")
        for f in info["files"]:
            print(f"  {f['name']:<34} {'SSD' if f['ssd'] else '   '} {'HDD' if f['hdd'] else '   '}")
        return 0

    to = {"c": "ssd", "d": "hdd"}.get(args.to, args.to)
    last = [0.0]

    def report(done: int, total: int, msg: str) -> None:
        if time.time() - last[0] > 2 or done == total:
            last[0] = time.time()
            print(msg, flush=True)

    try:
        print(move_models(to, ssd_dir, progress=report))
        return 0
    except Exception as exc:
        print(f"Move stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
