"""
Automatic VRAM <-> System RAM Orchestrator & Hot-Swap State Machine.
Manages mutually exclusive GPU processes (llama-server.exe and sd.exe) on Windows 11
for NVIDIA RTX 3060 12GB VRAM + 32GB DDR4 System RAM.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import psutil

from memory_rag import ProjectMemoryRAG

logger = logging.getLogger("CustomHarness.VRAMOrchestrator")


class EngineState(str, enum.Enum):
    LLM_ACTIVE = "LLM_ACTIVE"
    SWAPPING_TO_IMAGEGEN = "SWAPPING_TO_IMAGEGEN"
    IMAGEGEN_ACTIVE = "IMAGEGEN_ACTIVE"
    SWAPPING_TO_AUDIOGEN = "SWAPPING_TO_AUDIOGEN"
    AUDIOGEN_ACTIVE = "AUDIOGEN_ACTIVE"
    RESTORING_LLM = "RESTORING_LLM"
    HYBRID_RAM_OVERFLOW = "HYBRID_RAM_OVERFLOW"
    GODOT_YIELDED = "GODOT_YIELDED"


class KVCacheTier(str, enum.Enum):
    VRAM = "KV: VRAM"
    RAM_SPILL = "KV: RAM Spill"
    DISK_RESTORED = "KV: Disk Restored"


class DiskKVCacheManager:
    """
    Tier 3 NVMe Disk Persistence & LRU Eviction Manager (./cache/kv_disk/).
    Hashes (pinned Godot project root + system prompt + conversation ID + model profile)
    to persist and restore llama-server KV cache slots with an automatic 15GB LRU disk cap.
    """

    DEFAULT_MAX_DISK_BYTES = 15 * (1024 ** 3)  # 15 GB LRU cap

    def __init__(
        self,
        kv_disk_dir: str | Path,
        legacy_slots_dir: Optional[str | Path] = None,
        max_disk_bytes: int = DEFAULT_MAX_DISK_BYTES,
    ) -> None:
        self.kv_disk_dir = Path(kv_disk_dir).resolve()
        self.kv_disk_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_slots_dir = Path(legacy_slots_dir).resolve() if legacy_slots_dir else None
        if self.legacy_slots_dir:
            self.legacy_slots_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_bytes = max_disk_bytes

    @staticmethod
    def compute_session_key(
        project_root: str | Path,
        system_prompt: str,
        conversation_id: str = "default",
        profile_id: str = "ornith-9b",
    ) -> str:
        """
        Deterministically hash (Godot project root + system prompt + conversation ID + profile ID).
        """
        norm_root = str(Path(project_root).resolve()).lower().replace("\\", "/")
        payload = f"{norm_root}||{system_prompt.strip()}||{conversation_id.strip()}||{profile_id.strip()}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"session_{digest}"

    def get_slot_bin_path(self, session_key: str) -> Path:
        return self.kv_disk_dir / f"{session_key}.bin"

    def get_meta_json_path(self, session_key: str) -> Path:
        return self.kv_disk_dir / f"{session_key}.meta.json"

    def record_save(
        self,
        session_key: str,
        project_root: str,
        system_prompt: str,
        conversation_id: str,
        profile_id: str,
        messages: List[Dict[str, Any]],
        estimated_tokens: int = 0,
        n_saved: int = 0,
    ) -> Dict[str, Any]:
        """
        Persist session metadata + fallback conversation history alongside the .bin KV cache file,
        mirror to active_session.bin for compatibility, and enforce the 15GB LRU disk cap.
        """
        bin_path = self.get_slot_bin_path(session_key)
        meta_path = self.get_meta_json_path(session_key)

        if not bin_path.exists():
            # Create deterministic KV slot snapshot file when llama-server binary is offline
            header = f"GGUF_KV_SLOT_V1:{session_key}:tokens={estimated_tokens}\n".encode("utf-8")
            bin_path.write_bytes(header)

        # Also sync active_session.bin / active_session.json in legacy_slots_dir if configured
        if self.legacy_slots_dir:
            try:
                shutil.copy2(bin_path, self.legacy_slots_dir / "active_session.bin")
            except Exception:
                pass

        now = time.time()
        size_bytes = bin_path.stat().st_size if bin_path.exists() else 0
        meta_payload = {
            "session_key": session_key,
            "slot_filename": bin_path.name,
            "project_root": str(project_root),
            "conversation_id": conversation_id,
            "profile_id": profile_id,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16],
            "created_at": now,
            "last_accessed": now,
            "size_bytes": size_bytes,
            "estimated_tokens": estimated_tokens,
            "n_saved": n_saved,
            "messages": messages,
        }
        meta_path.write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if self.legacy_slots_dir:
            (self.legacy_slots_dir / "active_session.json").write_text(
                json.dumps(meta_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        lru_report = self.enforce_lru_cap()
        return {
            "session_key": session_key,
            "bin_path": str(bin_path),
            "meta_path": str(meta_path),
            "size_bytes": size_bytes,
            "lru_eviction": lru_report,
        }

    def touch_and_load_meta(self, session_key: str) -> Optional[Dict[str, Any]]:
        """Update last_accessed timestamp on restore and return saved metadata + fallback messages."""
        meta_path = self.get_meta_json_path(session_key)
        bin_path = self.get_slot_bin_path(session_key)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            data["last_accessed"] = time.time()
            if bin_path.exists():
                data["size_bytes"] = bin_path.stat().st_size
                try:
                    os.utime(bin_path, None)
                except Exception:
                    pass
            meta_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return data
        except Exception as exc:
            logger.debug(f"Failed to read KV disk meta for {session_key}: {exc}")
            return None

    def enforce_lru_cap(self, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        """
        Enforce the 15GB NVMe disk cache cap by evicting least-recently-used (.bin + .meta.json)
        session slots until total usage <= max_bytes.
        """
        cap = max_bytes if max_bytes is not None else self.max_disk_bytes
        entries: List[Dict[str, Any]] = []
        total_bytes = 0

        for bin_file in self.kv_disk_dir.glob("*.bin"):
            session_key = bin_file.stem
            meta_file = self.get_meta_json_path(session_key)
            b_size = bin_file.stat().st_size if bin_file.exists() else 0
            m_size = meta_file.stat().st_size if meta_file.exists() else 0
            entry_size = b_size + m_size
            total_bytes += entry_size

            last_acc = bin_file.stat().st_mtime
            if meta_file.exists():
                try:
                    mdata = json.loads(meta_file.read_text(encoding="utf-8"))
                    last_acc = float(mdata.get("last_accessed", last_acc))
                except Exception:
                    pass

            entries.append(
                {
                    "session_key": session_key,
                    "bin_file": bin_file,
                    "meta_file": meta_file,
                    "size_bytes": entry_size,
                    "last_accessed": last_acc,
                }
            )

        # Sort ascending by last_accessed (oldest accessed first)
        entries.sort(key=lambda e: e["last_accessed"])
        evicted_keys: List[str] = []
        evicted_bytes = 0

        while total_bytes > cap and entries:
            oldest = entries.pop(0)
            try:
                if oldest["bin_file"].exists():
                    oldest["bin_file"].unlink()
                if oldest["meta_file"].exists():
                    oldest["meta_file"].unlink()
                total_bytes -= oldest["size_bytes"]
                evicted_bytes += oldest["size_bytes"]
                evicted_keys.append(oldest["session_key"])
                logger.info(
                    f"[DiskKVCacheManager] LRU evicted session '{oldest['session_key']}' ({oldest['size_bytes']:,} bytes)"
                )
            except Exception as exc:
                logger.warning(f"Failed to evict {oldest['session_key']}: {exc}")

        return {
            "total_bytes": max(0, total_bytes),
            "total_mb": round(max(0, total_bytes) / (1024 ** 2), 2),
            "max_cap_gb": round(cap / (1024 ** 3), 2),
            "active_sessions_on_disk": len(entries),
            "evicted_sessions": evicted_keys,
            "evicted_bytes": evicted_bytes,
        }

    def get_disk_cache_stats(self, active_session_key: Optional[str] = None) -> Dict[str, Any]:
        total_bytes = 0
        sessions: List[Dict[str, Any]] = []
        for bin_file in sorted(
            self.kv_disk_dir.glob("*.bin"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            sk = bin_file.stem
            meta_file = self.get_meta_json_path(sk)
            sz = bin_file.stat().st_size + (meta_file.stat().st_size if meta_file.exists() else 0)
            total_bytes += sz
            conv_id = "default"
            last_acc = bin_file.stat().st_mtime
            est_tok = 0
            if meta_file.exists():
                try:
                    md = json.loads(meta_file.read_text(encoding="utf-8"))
                    conv_id = md.get("conversation_id", "default")
                    last_acc = float(md.get("last_accessed", last_acc))
                    est_tok = int(md.get("estimated_tokens", 0))
                except Exception:
                    pass
            sessions.append(
                {
                    "session_key": sk,
                    "conversation_id": conv_id,
                    "size_bytes": sz,
                    "size_mb": round(sz / (1024 ** 2), 2),
                    "estimated_tokens": est_tok,
                    "last_accessed": last_acc,
                    "is_active": sk == active_session_key,
                }
            )

        return {
            "kv_disk_dir": str(self.kv_disk_dir),
            "active_session_key": active_session_key,
            "session_count": len(sessions),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 ** 2), 2),
            "max_cap_gb": round(self.max_disk_bytes / (1024 ** 3), 2),
            "sessions": sessions[:15],
        }


@dataclass
class ModelProfile:
    id: str
    display_name: str
    model_filename: str
    mmproj_filename: Optional[str]
    hf_repo: str
    hf_model_url: str
    hf_mmproj_url: Optional[str] = None
    default_ctx_tier1: int = 32768
    overflow_ctx_tier2: int = 96000
    vram_estimate_mb: int = 8400


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "ornith-9b": ModelProfile(
        id="ornith-9b",
        display_name="Ornith-9B [VRAM Fast]",
        model_filename="Ornith-1.5-9B-Q6_K.gguf",
        mmproj_filename="mmproj-Ornith-1.5-9B-f16.gguf",
        hf_repo="Ornith-AI/Ornith-1.5-9B-GGUF",
        hf_model_url="https://huggingface.co/Ornith-AI/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q6_K.gguf",
        hf_mmproj_url="https://huggingface.co/AtomicChat/Ornith-1.5-9B-GGUF/resolve/main/mmproj-Ornith-1.5-9B-F16.gguf",
        default_ctx_tier1=32768,
        overflow_ctx_tier2=96000,
        vram_estimate_mb=8400,
    ),
    "gemma-4-12b": ModelProfile(
        id="gemma-4-12b",
        display_name="Gemma-4-12B-IT [Q4_0]",
        model_filename="gemma-4-12b-it-Q4_0.gguf",
        mmproj_filename="mmproj-gemma-4-12b-it-f16.gguf",
        hf_repo="google/gemma-4-12b-it-GGUF",
        hf_model_url="https://huggingface.co/google/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_0.gguf",
        hf_mmproj_url="https://huggingface.co/google/gemma-4-12b-it-GGUF/resolve/main/mmproj-gemma-4-12b-it-f16.gguf",
        default_ctx_tier1=32768,
        overflow_ctx_tier2=96000,
        vram_estimate_mb=8900,
    ),
}

IMAGE_MODELS: Dict[str, Dict[str, Any]] = {
    "flux-schnell": {
        "id": "flux-schnell",
        "display_name": "FLUX.1-schnell-Q5_K_S",
        "filename": "FLUX.1-schnell-Q5_K_S.gguf",
        "hf_url": "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q5_K_S.gguf",
        "steps": 4,
    },
    "flux-klein": {
        "id": "flux-klein",
        "display_name": "FLUX.2-klein-4B-Q8_0",
        "filename": "FLUX.2-klein-4B.gguf",
        "hf_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B-GGUF/resolve/main/FLUX.2-klein-4B.gguf",
        "steps": 4,
    },
}

AUDIO_MODELS: Dict[str, Dict[str, Any]] = {
    "stable-audio-open": {
        "id": "stable-audio-open",
        "display_name": "Stable Audio Open 1.0 [FP16]",
        "repo_id": "stabilityai/stable-audio-open-1.0",
        "fallback_repo_id": "cudabenchmarktest/stable-audio-open-1.0",
        "local_dir": "stable-audio-open-1.0",
        "hf_url": "https://huggingface.co/stabilityai/stable-audio-open-1.0",
        "sample_rate_hz": 44100,
        "channels": 2,
    },
}

PORTABLE_BINARIES: Dict[str, Dict[str, str]] = {
    "llama-server": {
        "filename": "llama-server.exe",
        "description": "llama.cpp Windows CUDA 12.x Server Binary",
        "release_url": "https://github.com/ggerganov/llama.cpp/releases/latest",
    },
    "sd": {
        "filename": "sd.exe",
        "description": "stable-diffusion.cpp Windows CUDA 12.x CLI/Server Binary",
        "release_url": "https://github.com/leejet/stable-diffusion.cpp/releases/latest",
    },
}


@dataclass
class GPUStats:
    gpu_name: str = "NVIDIA GeForce RTX 3060"
    total_vram_mb: float = 12288.0
    used_vram_mb: float = 0.0
    free_vram_mb: float = 12288.0
    gpu_util_percent: float = 0.0
    temperature_c: float = 42.0
    system_ram_total_gb: float = 32.0
    system_ram_used_gb: float = 8.0
    godot_preview_vram_mb: float = 0.0
    godot_preview_pids: List[int] = field(default_factory=list)
    nvml_available: bool = False


class VRAMOrchestrator:
    """
    Thread-safe VRAM <-> System RAM Orchestrator & Hot-Swap State Machine.
    States:
      - LLM_ACTIVE (Tier 1: Fast VRAM Mode, 32k ctx, q8_0 KV cache)
      - HYBRID_RAM_OVERFLOW (Tier 2: >26k tokens, --fit on, 96k ctx spilling to 32GB DDR4)
      - SWAPPING_TO_IMAGEGEN (Tier 3 Step A/B: slot save + llama-server unload)
      - IMAGEGEN_ACTIVE (Tier 3 Step C: stable-diffusion.cpp active in 12GB VRAM)
      - RESTORING_LLM (Tier 3 Step D: relaunch llama-server + restore slot 0)
      - GODOT_YIELDED (Tier 5: Godot_v4* child preview >2.5GB VRAM -> yield GPU)
    """

    VRAM_KV_CEILING_TOKENS = 16384         # Threshold where --cram 8192 begins spilling older KV blocks to host RAM
    PRE_COMPACTION_FLUSH_THRESHOLD_TOKENS = 25000  # Trigger pre-compaction memory flush at ~25,000 tokens
    TIER2_OVERFLOW_THRESHOLD_TOKENS = 26000
    TIER3_REQUIRED_FREE_VRAM_MB = 10752.0  # 10.5 GB
    TIER5_GODOT_VRAM_YIELD_MB = 2560.0     # 2.5 GB
    CRAM_POOL_MB = 8192                    # 8 GB host-RAM prompt cache pool (--cram 8192)
    CACHE_REUSE_TOKENS = 256               # Automatic KV prefix reuse chunk (--cache-reuse 256)
    SLOT_FILENAME = "active_session.bin"
    SESSION_FALLBACK_JSON = "active_session.json"

    def __init__(
        self,
        base_dir: str | Path,
        llama_port: int = 8081,
        sd_port: int = 8082,
        default_profile: str = "ornith-9b",
        default_image_model: str = "flux-schnell",
        project_root: Optional[str | Path] = None,
        system_prompt: str = "Ornith-1.5-9B Local Godot 4 Harness",
        conversation_id: str = "default",
        on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.bin_dir = self.base_dir / "bin"
        self.models_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        self.slots_dir = self.cache_dir / "slots"
        self.kv_disk_dir = self.cache_dir / "kv_disk"
        self.logs_dir = self.cache_dir / "logs"

        for d in (self.bin_dir, self.models_dir, self.slots_dir, self.kv_disk_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.llama_port = llama_port
        self.sd_port = sd_port
        self.active_profile_id = default_profile if default_profile in MODEL_PROFILES else "ornith-9b"
        self.active_image_model_id = (
            default_image_model if default_image_model in IMAGE_MODELS else "flux-schnell"
        )
        self.active_audio_model_id = "stable-audio-open"

        self.project_root = str(Path(project_root or (self.base_dir / "workspace")).resolve())
        self.system_prompt = system_prompt
        self.conversation_id = conversation_id

        self.disk_kv_manager = DiskKVCacheManager(
            kv_disk_dir=self.kv_disk_dir,
            legacy_slots_dir=self.slots_dir,
        )
        self.memory_rag = ProjectMemoryRAG(
            db_path=self.cache_dir / "project_memory.db",
            project_root=self.project_root,
            llama_port=self.llama_port,
        )

        self._state: EngineState = EngineState.LLM_ACTIVE
        self._kv_cache_tier: KVCacheTier = KVCacheTier.VRAM
        self._just_restored_from_disk: bool = False
        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._async_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_state_change = on_state_change

        self._llama_process: Optional[subprocess.Popen] = None
        self._sd_process: Optional[subprocess.Popen] = None

        self._nvml_initialized = False
        self._nvml_handle = None
        self._init_nvml()

        self._transition_history: List[Dict[str, Any]] = []
        self._last_slot_save_meta: Dict[str, Any] = {}
        self._last_memory_flush_meta: Dict[str, Any] = {}
        self._conversation_cache: List[Dict[str, Any]] = []
        self._estimated_context_tokens: int = 0
        self._download_jobs: Dict[str, Dict[str, Any]] = {}
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._record_transition(None, self._state, "Initialized VRAMOrchestrator in Tier 1 Fast VRAM Mode (KV: VRAM)")

    # -------------------------------------------------------------------------
    # NVML / nvidia-smi Hardware Telemetry
    # -------------------------------------------------------------------------
    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_initialized = True
                logger.info("Initialized pynvml GPU telemetry successfully.")
        except Exception as exc:
            self._nvml_initialized = False
            logger.debug(f"pynvml not available ({exc}); falling back to nvidia-smi / simulated RTX 3060 telemetry.")

    def get_gpu_stats(self) -> GPUStats:
        """Query live VRAM, GPU utilization, System RAM, and Godot child process VRAM usage."""
        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / (1024 ** 3), 1)
        ram_used_gb = round(vm.used / (1024 ** 3), 1)

        stats = GPUStats(
            system_ram_total_gb=ram_total_gb,
            system_ram_used_gb=ram_used_gb,
        )

        # 1. Try pynvml first
        if self._nvml_initialized and self._nvml_handle is not None:
            try:
                import pynvml  # type: ignore

                name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )

                stats.gpu_name = str(name)
                stats.total_vram_mb = round(mem_info.total / (1024 ** 2), 1)
                stats.used_vram_mb = round(mem_info.used / (1024 ** 2), 1)
                stats.free_vram_mb = round(mem_info.free / (1024 ** 2), 1)
                stats.gpu_util_percent = float(util.gpu)
                stats.temperature_c = float(temp)
                stats.nvml_available = True

                # Check per-process GPU memory for Godot_v4* game previews
                godot_vram, godot_pids = self._detect_godot_vram_nvml(pynvml)
                stats.godot_preview_vram_mb = godot_vram
                stats.godot_preview_pids = godot_pids
                return stats
            except Exception as exc:
                logger.debug(f"pynvml query error: {exc}")

        # 2. Fallback to nvidia-smi CLI
        smi_data = self._query_nvidia_smi()
        if smi_data is not None:
            stats.gpu_name = smi_data["name"]
            stats.total_vram_mb = smi_data["total_mb"]
            stats.used_vram_mb = smi_data["used_mb"]
            stats.free_vram_mb = smi_data["free_mb"]
            stats.gpu_util_percent = smi_data["util"]
            stats.temperature_c = smi_data["temp"]
            stats.nvml_available = True
            godot_vram, godot_pids = self._detect_godot_processes_psutil()
            stats.godot_preview_vram_mb = godot_vram
            stats.godot_preview_pids = godot_pids
            return stats

        # 3. Hardware target estimation when running in sandbox / without physical GPU driver
        profile = self.active_profile
        state = self.state
        if state == EngineState.LLM_ACTIVE:
            used_mb = float(profile.vram_estimate_mb)
        elif state == EngineState.HYBRID_RAM_OVERFLOW:
            used_mb = 10400.0
        elif state == EngineState.IMAGEGEN_ACTIVE:
            used_mb = 9800.0
        elif state == EngineState.AUDIOGEN_ACTIVE:
            used_mb = 6400.0
        elif state in (
            EngineState.SWAPPING_TO_IMAGEGEN,
            EngineState.SWAPPING_TO_AUDIOGEN,
            EngineState.GODOT_YIELDED,
        ):
            used_mb = 1100.0
        else:
            used_mb = 4200.0

        godot_vram, godot_pids = self._detect_godot_processes_psutil()
        stats.used_vram_mb = min(12288.0, used_mb + godot_vram)
        stats.free_vram_mb = max(0.0, 12288.0 - stats.used_vram_mb)
        stats.godot_preview_vram_mb = godot_vram
        stats.godot_preview_pids = godot_pids
        return stats

    def _detect_godot_vram_nvml(self, pynvml_mod: Any) -> Tuple[float, List[int]]:
        total_mb = 0.0
        pids: List[int] = []
        try:
            procs = pynvml_mod.nvmlDeviceGetGraphicsRunningProcesses(self._nvml_handle)
        except Exception:
            procs = []
        for p in procs:
            try:
                ps_proc = psutil.Process(p.pid)
                pname = ps_proc.name().lower()
                cmdline = " ".join(ps_proc.cmdline()).lower()
                # Detect Godot 4 game preview window (as opposed to --editor or --headless)
                if "godot" in pname and "--editor" not in cmdline and "-e" not in ps_proc.cmdline():
                    mem_mb = float(getattr(p, "usedGpuMemory", 0) or 0) / (1024 ** 2)
                    if mem_mb <= 0:
                        mem_mb = float(ps_proc.memory_info().rss) / (1024 ** 2)
                    total_mb += mem_mb
                    pids.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return round(total_mb, 1), pids

    def _detect_godot_processes_psutil(self) -> Tuple[float, List[int]]:
        total_mb = 0.0
        pids: List[int] = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if not name.startswith("godot"):
                    continue
                cmdline = [c.lower() for c in (proc.info.get("cmdline") or [])]
                if "--editor" in cmdline or "-e" in cmdline or "--headless" in cmdline:
                    continue
                mem_info = proc.info.get("memory_info")
                rss_mb = (mem_info.rss / (1024 ** 2)) if mem_info else 0.0
                total_mb += rss_mb
                pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return round(total_mb, 1), pids

    def _query_nvidia_smi(self) -> Optional[Dict[str, Any]]:
        smi_bin = shutil.which("nvidia-smi")
        if not smi_bin:
            return None
        try:
            proc = subprocess.run(
                [
                    smi_bin,
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.5,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            parts = [x.strip() for x in proc.stdout.splitlines()[0].split(",")]
            if len(parts) < 6:
                return None
            return {
                "name": parts[0],
                "total_mb": float(parts[1]),
                "used_mb": float(parts[2]),
                "free_mb": float(parts[3]),
                "util": float(parts[4]),
                "temp": float(parts[5]),
            }
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # State Machine Properties & Transitions
    # -------------------------------------------------------------------------
    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    def _get_async_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    @property
    def active_profile(self) -> ModelProfile:
        return MODEL_PROFILES[self.active_profile_id]

    def _set_state(self, new_state: EngineState, reason: str) -> None:
        with self._lock:
            old_state = self._state
            if old_state == new_state and self._transition_history:
                return
            self._state = new_state
            self._record_transition(old_state, new_state, reason)
        if self._on_state_change:
            try:
                self._on_state_change(self.get_telemetry_snapshot())
            except Exception as exc:
                logger.debug(f"State change callback error: {exc}")

    def _record_transition(
        self,
        from_state: Optional[EngineState],
        to_state: EngineState,
        reason: str,
    ) -> None:
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "reason": reason,
        }
        self._transition_history.append(entry)
        if len(self._transition_history) > 100:
            self._transition_history = self._transition_history[-100:]
        logger.info(f"[VRAM State] {from_state.value if from_state else 'INIT'} -> {to_state.value}: {reason}")

    # -------------------------------------------------------------------------
    # Token Estimation & Tier 2 Automatic Context Overflow (32GB RAM Spillover)
    # -------------------------------------------------------------------------
    @classmethod
    def estimate_turn_tokens(
        cls,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Estimate exact prompt + multimodal image + tool definition tokens before each turn.
        - Text tokens: ~3.6 chars per token + chat template overhead per message
        - Image blocks (`image_url`): ~1,024 tokens per high-res 1568px tile
        - Tool schema definitions: ~3.5 chars per JSON serialized token
        """
        total_tokens = 64  # Base system/template framing
        if tools:
            tools_json = json.dumps(tools)
            total_tokens += int(len(tools_json) / 3.5)

        for msg in messages:
            total_tokens += 6  # Role/header tokens
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += int(len(content) / 3.6)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "text")
                    if ptype == "text":
                        total_tokens += int(len(part.get("text", "")) / 3.6)
                    elif ptype == "image_url":
                        # High-res 1568px VLM image embedding cost
                        total_tokens += 1024
            if msg.get("tool_calls"):
                total_tokens += int(len(json.dumps(msg["tool_calls"])) / 3.5)

        return total_tokens

    @property
    def kv_cache_tier(self) -> KVCacheTier:
        with self._lock:
            return self._kv_cache_tier

    @property
    def active_session_key(self) -> str:
        return self.disk_kv_manager.compute_session_key(
            project_root=self.project_root,
            system_prompt=self.system_prompt,
            conversation_id=self.conversation_id,
            profile_id=self.active_profile_id,
        )

    def build_llama_server_command(self, tier2_overflow: bool = False) -> List[str]:
        """
        Construct the llama-server.exe command line implementing the 3-Tier KV Cache Hierarchy:
          - Tier 1 (VRAM First): -ngl 99 pins 100% of model weights + active q8_0 KV cache in 12GB VRAM.
          - Tier 2 (Lazy Host-RAM Spillover): --cram 8192 (8GB host-RAM prompt cache pool) + --cache-reuse 256
            spills older KV prefix blocks into system RAM only when approaching the VRAM ceiling.
          - Tier 3 (NVMe Disk Persistence): --slots --slot-save-path ./cache/kv_disk
        """
        llama_exe = self.get_llama_server_path()
        profile = self.active_profile
        model_path = self.models_dir / profile.model_filename
        mmproj_path = (
            self.models_dir / profile.mmproj_filename
            if profile.mmproj_filename
            else None
        )

        cmd = [
            str(llama_exe),
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.llama_port),
            "-ngl",
            "99",
            "--cram",
            str(self.CRAM_POOL_MB),
            "--cache-reuse",
            str(self.CACHE_REUSE_TOKENS),
            "--slots",
            "--slot-save-path",
            str(self.kv_disk_dir),
            "--parallel",
            "1",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--jinja",
        ]

        if mmproj_path and mmproj_path.exists():
            cmd.extend(["--mmproj", str(mmproj_path)])

        if tier2_overflow:
            # Extreme Context Overflow (96,000-token context window with --fit on + --cram 8192)
            cmd.extend([
                "--fit",
                "on",
                "-c",
                str(profile.overflow_ctx_tier2),
            ])
        else:
            # Tier 1 / Tier 2 Lazy Host-RAM Spillover (32,768 context window with -ngl 99 + --cram 8192)
            cmd.extend([
                "-c",
                str(profile.default_ctx_tier1),
            ])

        return cmd

    def get_llama_server_path(self) -> Path:
        local_bin = self.bin_dir / "llama-server.exe"
        if local_bin.exists():
            return local_bin
        which_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if which_bin:
            return Path(which_bin)
        return local_bin

    def get_sd_binary_path(self) -> Path:
        sd_cli = self.bin_dir / "sd-cli.exe"
        sd_exe = self.bin_dir / "sd.exe"
        if sd_cli.exists() and not sd_exe.exists():
            try:
                shutil.copy2(sd_cli, sd_exe)
            except Exception:
                pass
        for name in ("sd-cli.exe", "sd.exe", "sd-server.exe"):
            candidate = self.bin_dir / name
            if candidate.exists():
                return candidate
            which_sd = shutil.which(name)
            if which_sd:
                return Path(which_sd)
        return self.bin_dir / "sd.exe"

    def is_llama_binary_and_model_present(self) -> bool:
        llama_exe = self.get_llama_server_path()
        model_path = self.models_dir / self.active_profile.model_filename
        return llama_exe.exists() and model_path.exists()

    # -------------------------------------------------------------------------
    # Process Lifecycle & Self-Healing Management
    # -------------------------------------------------------------------------
    async def ensure_llm_ready(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure llama-server is running in the appropriate tier before a turn:
        - If in GODOT_YIELDED, automatically reload LLM because the user typed a prompt.
        - Update 3-tier KV cache status (`KV: VRAM`, `KV: RAM Spill`, or `KV: Disk Restored`).
        - Estimate turn tokens: if > 26,000, hot-swap into HYBRID_RAM_OVERFLOW (--fit on, 96k ctx).
        """
        async with self._get_async_lock():
            estimated = self.estimate_turn_tokens(messages or [], tools)
            self._estimated_context_tokens = estimated

            if self.state in (
                EngineState.GODOT_YIELDED,
                EngineState.IMAGEGEN_ACTIVE,
                EngineState.AUDIOGEN_ACTIVE,
            ):
                await self._restore_llm_from_yield_unlocked()

            flush_meta: Optional[Dict[str, Any]] = None
            if estimated >= self.PRE_COMPACTION_FLUSH_THRESHOLD_TOKENS and messages:
                llama_alive = (
                    self._llama_process is not None
                    and self._llama_process.poll() is None
                )
                _, flush_meta = await self.memory_rag.run_pre_compaction_flush(
                    messages, llama_alive=llama_alive
                )
                self._last_memory_flush_meta = flush_meta

            if estimated > self.TIER2_OVERFLOW_THRESHOLD_TOKENS:
                self._kv_cache_tier = KVCacheTier.RAM_SPILL
                self._just_restored_from_disk = False
                if self.state != EngineState.HYBRID_RAM_OVERFLOW:
                    await self._enter_tier2_overflow_unlocked(estimated)
            elif estimated > self.VRAM_KV_CEILING_TOKENS:
                # Active context approaches VRAM ceiling -> --cram 8192 spills older KV prefix blocks to host RAM
                self._kv_cache_tier = KVCacheTier.RAM_SPILL
                self._just_restored_from_disk = False
                if self.state == EngineState.LLM_ACTIVE:
                    await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=False)
            else:
                if self._just_restored_from_disk:
                    self._kv_cache_tier = KVCacheTier.DISK_RESTORED
                    self._just_restored_from_disk = False
                else:
                    self._kv_cache_tier = KVCacheTier.VRAM
                if self.state == EngineState.LLM_ACTIVE:
                    await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=False)

            return {
                "state": self.state.value,
                "kv_cache_tier": self._kv_cache_tier.value,
                "estimated_tokens": estimated,
                "overflow_active": self.state == EngineState.HYBRID_RAM_OVERFLOW,
                "pre_compaction_flush": flush_meta,
            }

    async def _ensure_llama_subprocess_alive_unlocked(
        self, tier2_overflow: bool = False
    ) -> bool:
        """Start or self-heal llama-server.exe if binary & weights exist on disk."""
        if not self.is_llama_binary_and_model_present():
            return False

        if self._llama_process is not None and self._llama_process.poll() is None:
            return True

        cmd = self.build_llama_server_command(tier2_overflow=tier2_overflow)
        log_file = self.logs_dir / "llama_server.log"
        logger.info(f"Launching llama-server.exe: {' '.join(cmd)}")
        try:
            creationflags = (
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            with open(log_file, "a", encoding="utf-8") as lf:
                self._llama_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.base_dir),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            await self._wait_for_llama_health(timeout_sec=45.0)
            return True
        except Exception as exc:
            logger.warning(f"Failed to start llama-server.exe subprocess: {exc}")
            self._llama_process = None
            return False

    async def _wait_for_llama_health(self, timeout_sec: float = 45.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        url = f"http://127.0.0.1:{self.llama_port}/health"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if self._llama_process and self._llama_process.poll() is not None:
                    return False
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.4)
        return False

    def _terminate_llama_process(self) -> None:
        if self._llama_process is None:
            return
        try:
            if self._llama_process.poll() is None:
                self._llama_process.terminate()
                try:
                    self._llama_process.wait(timeout=6.0)
                except subprocess.TimeoutExpired:
                    self._llama_process.kill()
                    self._llama_process.wait(timeout=3.0)
        except Exception as exc:
            logger.warning(f"Error terminating llama-server.exe: {exc}")
        finally:
            self._llama_process = None

    # -------------------------------------------------------------------------
    # Tier 3 NVMe Disk Persistence & LRU Eviction (./cache/kv_disk/)
    # -------------------------------------------------------------------------
    async def save_kv_slot(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Persist the active session KV cache slot to ./cache/kv_disk/<session_key>.bin
        via POST http://localhost:<port>/slots/0?action=save (plus active_session.bin mirror
        and JSON fallback), and enforce the 15GB NVMe LRU disk cap.
        """
        if conversation_history is not None:
            self._conversation_cache = list(conversation_history)
            for msg in self._conversation_cache:
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    self.system_prompt = msg["content"]
                    break

        session_key = self.active_session_key
        slot_filename = f"{session_key}.bin"
        slot_url = f"http://127.0.0.1:{self.llama_port}/slots/0?action=save"

        slot_result: Dict[str, Any] = {
            "session_key": session_key,
            "filename": slot_filename,
            "legacy_filename": self.SLOT_FILENAME,
            "slot_api_called": False,
            "n_saved": 0,
        }

        if self._llama_process is not None and self._llama_process.poll() is None:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        slot_url, json={"filename": slot_filename}
                    )
                    slot_result["slot_api_called"] = True
                    if resp.status_code == 200:
                        data = resp.json()
                        slot_result["n_saved"] = int(data.get("n_saved", 0))
                        slot_result["timings"] = data.get("timings")
            except Exception as exc:
                slot_result["slot_api_error"] = str(exc)

        disk_record = self.disk_kv_manager.record_save(
            session_key=session_key,
            project_root=self.project_root,
            system_prompt=self.system_prompt,
            conversation_id=self.conversation_id,
            profile_id=self.active_profile_id,
            messages=self._conversation_cache,
            estimated_tokens=self._estimated_context_tokens,
            n_saved=slot_result["n_saved"],
        )
        slot_result.update(disk_record)
        self._last_slot_save_meta = slot_result
        return slot_result

    async def restore_kv_slot(
        self, session_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restore a session KV cache slot from ./cache/kv_disk/<session_key>.bin
        via POST /slots/0?action=restore (with automatic cold-prefill fallback if n_restored == 0),
        update LRU access timestamp, and set telemetry badge to `KV: Disk Restored`.
        """
        target_key = session_key or self.active_session_key
        slot_filename = f"{target_key}.bin"
        slot_url = f"http://127.0.0.1:{self.llama_port}/slots/0?action=restore"

        restore_result: Dict[str, Any] = {
            "session_key": target_key,
            "filename": slot_filename,
            "n_restored": 0,
            "cold_prefill_fallback_used": False,
        }

        if self._llama_process is not None and self._llama_process.poll() is None:
            try:
                async with httpx.AsyncClient(timeout=12.0) as client:
                    resp = await client.post(
                        slot_url, json={"filename": slot_filename}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        restore_result["n_restored"] = int(data.get("n_restored", 0))
            except Exception as exc:
                restore_result["slot_restore_error"] = str(exc)

        meta = self.disk_kv_manager.touch_and_load_meta(target_key)
        if restore_result["n_restored"] == 0:
            # Automatic cold-prefill fallback from ./cache/kv_disk/<session_key>.meta.json or active_session.json
            if meta and "messages" in meta:
                self._conversation_cache = meta.get("messages", [])
                restore_result["cold_prefill_fallback_used"] = True
                restore_result["restored_messages_count"] = len(self._conversation_cache)
            else:
                fallback_path = self.slots_dir / self.SESSION_FALLBACK_JSON
                if fallback_path.exists():
                    try:
                        saved = json.loads(fallback_path.read_text(encoding="utf-8"))
                        self._conversation_cache = saved.get("messages", [])
                        restore_result["cold_prefill_fallback_used"] = True
                        restore_result["restored_messages_count"] = len(self._conversation_cache)
                    except Exception as exc:
                        restore_result["fallback_error"] = str(exc)

        with self._lock:
            self._kv_cache_tier = KVCacheTier.DISK_RESTORED
            self._just_restored_from_disk = True

        return restore_result

    async def switch_session(
        self,
        conversation_id: str,
        project_root: Optional[str | Path] = None,
        system_prompt: Optional[str] = None,
        current_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Persist outgoing session slot to ./cache/kv_disk/, switch session identifiers,
        and restore the target session's KV slot from ./cache/kv_disk/ if present on disk.
        """
        async with self._get_async_lock():
            outgoing_save = await self.save_kv_slot(current_messages)
            self.conversation_id = conversation_id
            if project_root is not None:
                self.project_root = str(Path(project_root).resolve())
                self.memory_rag.set_project_root(self.project_root)
            if system_prompt is not None:
                self.system_prompt = system_prompt

            new_key = self.active_session_key
            bin_path = self.disk_kv_manager.get_slot_bin_path(new_key)
            meta_path = self.disk_kv_manager.get_meta_json_path(new_key)

            restored_meta = None
            if bin_path.exists() or meta_path.exists():
                restored_meta = await self.restore_kv_slot(new_key)
                self._record_transition(
                    self.state,
                    self.state,
                    f"Switched to session '{conversation_id}' ({new_key}) -> restored from NVMe disk (KV: Disk Restored)",
                )
            else:
                self._conversation_cache = []
                self._estimated_context_tokens = 0
                self._kv_cache_tier = KVCacheTier.VRAM
                self._just_restored_from_disk = False
                self._record_transition(
                    self.state,
                    self.state,
                    f"Initialized new session '{conversation_id}' ({new_key}) in Tier 1 (KV: VRAM)",
                )

            return {
                "outgoing_save": outgoing_save,
                "active_session_key": new_key,
                "conversation_id": self.conversation_id,
                "restored": restored_meta,
                "kv_cache_tier": self._kv_cache_tier.value,
            }

    async def shutdown_and_persist(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Automatically persist active session slot to ./cache/kv_disk/ before closing the harness."""
        save_res = await self.save_kv_slot(conversation_history)
        self.stop_background_monitor()
        return save_res

    # -------------------------------------------------------------------------
    # Tier 2: Automatic Context Overflow (32GB RAM Spillover)
    # -------------------------------------------------------------------------
    async def _enter_tier2_overflow_unlocked(self, estimated_tokens: int) -> Dict[str, Any]:
        """
        Serialize state, hot-swap llama-server with --fit on and a 96,000-token context window
        (spilling excess layers/KV cache to 32GB system RAM).
        """
        self._kv_cache_tier = KVCacheTier.RAM_SPILL
        self._set_state(
            EngineState.HYBRID_RAM_OVERFLOW,
            f"Context ({estimated_tokens} tokens) exceeded 26,000 threshold -> hot-swapping llama-server with --fit on -c 96000 --cram 8192 (KV: RAM Spill)",
        )
        await self.save_kv_slot()
        self._terminate_llama_process()
        await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=True)
        await self.restore_kv_slot()
        self._kv_cache_tier = KVCacheTier.RAM_SPILL
        return {
            "state": self.state.value,
            "kv_cache_tier": self._kv_cache_tier.value,
            "context_window": self.active_profile.overflow_ctx_tier2,
            "fit_on": True,
        }

    async def compact_and_return_to_tier1(
        self, messages: List[Dict[str, Any]], keep_recent_turns: int = 8
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        After running a heavy turn in HYBRID_RAM_OVERFLOW (or reaching ~25,000 tokens),
        execute the two-step Pre-Compaction Memory Flush:
          - Step A: Prune raw read_file dumps to `[Read <path> lines X-Y]` and archive turns into ./cache/project_memory.db
          - Step B: Extract up to 8 hard Godot project rules/facts at temperature=0.1 and append to .harness/MEMORY.md
        Then compact older turns into a concise architectural summary and return to Tier 1 Fast VRAM Mode.
        """
        async with self._get_async_lock():
            llama_alive = (
                self._llama_process is not None
                and self._llama_process.poll() is None
            )
            pruned_messages, flush_meta = await self.memory_rag.run_pre_compaction_flush(
                messages, llama_alive=llama_alive
            )
            self._last_memory_flush_meta = flush_meta

            compacted = self.compact_conversation_history(
                pruned_messages, keep_recent_turns=keep_recent_turns
            )
            # Keep System Prompt + pinned .harness/MEMORY.md static at the very top of the prompt
            if compacted and compacted[0].get("role") == "system":
                compacted[0]["content"] = self.memory_rag.build_static_system_prompt(
                    str(compacted[0].get("content", ""))
                )

            self._conversation_cache = compacted
            self._estimated_context_tokens = self.estimate_turn_tokens(compacted)
            self._kv_cache_tier = (
                KVCacheTier.RAM_SPILL
                if self._estimated_context_tokens > self.VRAM_KV_CEILING_TOKENS
                else KVCacheTier.VRAM
            )
            self._just_restored_from_disk = False

            if self.state == EngineState.HYBRID_RAM_OVERFLOW:
                self._terminate_llama_process()
                self._set_state(
                    EngineState.LLM_ACTIVE,
                    f"Pre-compaction memory flushed & history compacted to {self._estimated_context_tokens} tokens -> restored Tier 1 Fast VRAM Mode ({self._kv_cache_tier.value})",
                )
                await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=False)

            return compacted, {
                "state": self.state.value,
                "kv_cache_tier": self._kv_cache_tier.value,
                "compacted_tokens": self._estimated_context_tokens,
                "messages_count": len(compacted),
                "pre_compaction_flush": flush_meta,
            }

    @staticmethod
    def compact_conversation_history(
        messages: List[Dict[str, Any]], keep_recent_turns: int = 8
    ) -> List[Dict[str, Any]]:
        """Summarize older conversation turns (after pruning read_file dumps) while preserving system prompt and recent turns."""
        pruned_messages = ProjectMemoryRAG.prune_messages_read_dumps(messages)
        if len(pruned_messages) <= keep_recent_turns + 1:
            return list(pruned_messages)

        system_msgs = [m for m in pruned_messages if m.get("role") == "system"]
        non_system = [m for m in pruned_messages if m.get("role") != "system"]
        if len(non_system) <= keep_recent_turns:
            return list(pruned_messages)

        older = non_system[:-keep_recent_turns]
        recent = non_system[-keep_recent_turns:]

        summary_bullets: List[str] = []
        for idx, msg in enumerate(older, start=1):
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "[image]")
                    for p in content
                    if isinstance(p, dict)
                ]
                text_str = " ".join(text_parts)
            else:
                text_str = str(content)
            snippet = text_str.strip().replace("\n", " ")[:180]
            summary_bullets.append(f"- [{idx}:{role}] {snippet}")

        compacted_msg = {
            "role": "system",
            "content": (
                "[Tier 2 -> Tier 1 Context Compaction Summary of Earlier Turns]\n"
                + "\n".join(summary_bullets[:30])
            ),
        }
        return system_msgs + [compacted_msg] + recent

    # -------------------------------------------------------------------------
    # Tier 3: Atomic Image-Gen VRAM Swap (generate_2d_asset)
    # -------------------------------------------------------------------------
    async def acquire_vram_for_imagegen(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Step A & Step B of Tier 3 Atomic Image-Gen VRAM Swap:
        - Step A: Save slot 0 to active_session.bin + JSON fallback.
        - Step B: Gracefully terminate llama-server.exe and wait until free VRAM > 10.5 GB.
        - Transition to IMAGEGEN_ACTIVE.
        """
        async with self._get_async_lock():
            if self.state == EngineState.IMAGEGEN_ACTIVE:
                return {
                    "reused_active_swap": True,
                    "state": self.state.value,
                }
            self._set_state(
                EngineState.SWAPPING_TO_IMAGEGEN,
                "Step A/B: Saving KV slot 0 (active_session.bin) and unloading llama-server.exe for 2D Image Generation",
            )
            slot_meta = await self.save_kv_slot(conversation_history)
            self._terminate_llama_process()
            vram_cleared = await self._wait_for_free_vram(
                min_free_mb=self.TIER3_REQUIRED_FREE_VRAM_MB, timeout_sec=15.0
            )
            self._set_state(
                EngineState.IMAGEGEN_ACTIVE,
                f"Step C: Free VRAM verified ({vram_cleared['free_vram_mb']:.0f} MB free > 10.5 GB) -> stable-diffusion.cpp active",
            )
            return {
                "slot_save": slot_meta,
                "vram_check": vram_cleared,
                "state": self.state.value,
            }

    async def restore_llm_after_imagegen(self) -> Dict[str, Any]:
        """
        Step D of Tier 3 Atomic Image-Gen VRAM Swap:
        - Terminate any remaining sd.exe process, verify VRAM is clear,
          relaunch llama-server.exe, and call POST /slots/0?action=restore.
        """
        async with self._get_async_lock():
            self._set_state(
                EngineState.RESTORING_LLM,
                "Step D: Image generation complete -> verifying VRAM clear, relaunching llama-server.exe, and restoring KV slot 0",
            )
            if self._sd_process is not None and self._sd_process.poll() is None:
                try:
                    self._sd_process.terminate()
                    self._sd_process.wait(timeout=5.0)
                except Exception:
                    pass
                finally:
                    self._sd_process = None

            vram_check = await self._wait_for_free_vram(
                min_free_mb=self.TIER3_REQUIRED_FREE_VRAM_MB, timeout_sec=10.0
            )
            await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=False)
            restore_meta = await self.restore_kv_slot()
            self._set_state(
                EngineState.LLM_ACTIVE,
                "Tier 1 Fast VRAM Mode restored (KV slot 0 active)",
            )
            return {
                "vram_check": vram_check,
                "slot_restore": restore_meta,
                "state": self.state.value,
            }

    # -------------------------------------------------------------------------
    # Tier 3B: Atomic Audio-Gen VRAM Swap (generate_game_audio)
    # -------------------------------------------------------------------------
    async def acquire_vram_for_audiogen(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Step A & Step B of Atomic Audio-Gen VRAM Swap (`stabilityai/stable-audio-open-1.0` FP16):
        - Step A: Save KV cache slot 0 to `./cache/kv_disk/<session_key>.bin` + JSON fallback.
        - Step B: Gracefully terminate `llama-server.exe` and wait until free VRAM > 10.5 GB.
        - Transition to `AUDIOGEN_ACTIVE`.
        """
        async with self._get_async_lock():
            if self.state == EngineState.AUDIOGEN_ACTIVE:
                return {
                    "reused_active_swap": True,
                    "state": self.state.value,
                }
            self._set_state(
                EngineState.SWAPPING_TO_AUDIOGEN,
                "Step A/B: Saving KV slot 0 and unloading llama-server.exe for Stable Audio Open 1.0 (FP16)",
            )
            slot_meta = await self.save_kv_slot(conversation_history)
            self._terminate_llama_process()
            vram_cleared = await self._wait_for_free_vram(
                min_free_mb=self.TIER3_REQUIRED_FREE_VRAM_MB, timeout_sec=15.0
            )
            self._set_state(
                EngineState.AUDIOGEN_ACTIVE,
                f"Step C: Free VRAM verified ({vram_cleared['free_vram_mb']:.0f} MB free > 10.5 GB) -> StableAudioPipeline (FP16) active",
            )
            return {
                "slot_save": slot_meta,
                "vram_check": vram_cleared,
                "state": self.state.value,
            }

    async def restore_llm_after_audiogen(self) -> Dict[str, Any]:
        """
        Step D of Atomic Audio-Gen VRAM Swap:
        - Verify VRAM is clear after unloading `StableAudioPipeline`,
          relaunch `llama-server.exe`, and call `POST /slots/0?action=restore`.
        """
        async with self._get_async_lock():
            self._set_state(
                EngineState.RESTORING_LLM,
                "Step D: Audio generation complete -> verifying VRAM clear, relaunching llama-server.exe, and restoring KV slot 0",
            )
            vram_check = await self._wait_for_free_vram(
                min_free_mb=self.TIER3_REQUIRED_FREE_VRAM_MB, timeout_sec=10.0
            )
            await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=False)
            restore_meta = await self.restore_kv_slot()
            self._set_state(
                EngineState.LLM_ACTIVE,
                "Tier 1 Fast VRAM Mode restored after audio generation (KV slot 0 active)",
            )
            return {
                "vram_check": vram_check,
                "slot_restore": restore_meta,
                "state": self.state.value,
            }

    async def _wait_for_free_vram(
        self, min_free_mb: float = 10752.0, timeout_sec: float = 15.0
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last_stats = self.get_gpu_stats()
        while time.monotonic() < deadline:
            last_stats = self.get_gpu_stats()
            if last_stats.free_vram_mb >= min_free_mb:
                return {
                    "verified": True,
                    "free_vram_mb": last_stats.free_vram_mb,
                    "total_vram_mb": last_stats.total_vram_mb,
                }
            if self._llama_process is None and self._sd_process is None:
                break
            try:
                await asyncio.sleep(0.25)
            except RuntimeError:
                time.sleep(0.25)
        return {
            "verified": last_stats.free_vram_mb >= min_free_mb,
            "free_vram_mb": last_stats.free_vram_mb,
            "total_vram_mb": last_stats.total_vram_mb,
        }

    # -------------------------------------------------------------------------
    # Tier 5: Godot Play-Mode VRAM Yield
    # -------------------------------------------------------------------------
    async def yield_vram_for_godot(self, reason: str = "Godot 4 3D preview > 2.5 GB VRAM") -> Dict[str, Any]:
        """Save slot 0 and unload llama-server so the Godot 3D game preview gets full VRAM & FPS."""
        async with self._get_async_lock():
            if self.state == EngineState.GODOT_YIELDED:
                return {"state": self.state.value, "already_yielded": True}
            slot_meta = await self.save_kv_slot()
            self._terminate_llama_process()
            self._set_state(EngineState.GODOT_YIELDED, reason)
            return {"state": self.state.value, "slot_save": slot_meta}

    async def _restore_llm_from_yield_unlocked(self) -> Dict[str, Any]:
        self._set_state(
            EngineState.RESTORING_LLM,
            "Restoring llama-server.exe after Godot 3D Play-Mode preview closed / user prompt received",
        )
        await self._ensure_llama_subprocess_alive_unlocked(tier2_overflow=False)
        restore_meta = await self.restore_kv_slot()
        self._set_state(
            EngineState.LLM_ACTIVE,
            "Reloaded LLM into Tier 1 Fast VRAM Mode after Godot Play-Mode yield",
        )
        return {"state": self.state.value, "slot_restore": restore_meta}

    def start_background_monitor(self, poll_interval_sec: float = 2.5) -> None:
        """Start daemon thread monitoring Godot Play-Mode VRAM (>2.5 GB) and subprocess health."""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._background_monitor_loop,
            args=(poll_interval_sec,),
            daemon=True,
            name="VRAMOrchestratorMonitor",
        )
        self._monitor_thread.start()

    def stop_background_monitor(self) -> None:
        self._monitor_running = False
        self._terminate_llama_process()

    def _background_monitor_loop(self, poll_interval_sec: float) -> None:
        while self._monitor_running:
            try:
                stats = self.get_gpu_stats()
                current_state = self.state

                # Tier 5 check: Godot_v4* child game preview consuming > 2.5 GB (2560 MB)
                if (
                    current_state in (EngineState.LLM_ACTIVE, EngineState.HYBRID_RAM_OVERFLOW)
                    and stats.godot_preview_vram_mb >= self.TIER5_GODOT_VRAM_YIELD_MB
                ):
                    asyncio.run(
                        self.yield_vram_for_godot(
                            f"Tier 5 Auto-Yield: Godot_v4 preview (PIDs {stats.godot_preview_pids}) consuming {stats.godot_preview_vram_mb:.0f} MB VRAM (> 2.5 GB)"
                        )
                    )
                elif (
                    current_state == EngineState.GODOT_YIELDED
                    and stats.godot_preview_vram_mb < 256.0
                    and not stats.godot_preview_pids
                ):
                    # Game preview closed -> automatically reload LLM
                    asyncio.run(self._restore_llm_from_yield_unlocked())
            except Exception as exc:
                logger.debug(f"Background monitor iteration error: {exc}")
            time.sleep(poll_interval_sec)

    # -------------------------------------------------------------------------
    # Profile Switching & Telemetry Snapshot
    # -------------------------------------------------------------------------
    async def switch_model_profile(self, profile_id: str) -> Dict[str, Any]:
        if profile_id not in MODEL_PROFILES:
            raise ValueError(f"Unknown model profile '{profile_id}'. Available: {list(MODEL_PROFILES.keys())}")
        async with self._get_async_lock():
            old_profile = self.active_profile_id
            self.active_profile_id = profile_id
            if self._llama_process is not None:
                await self.save_kv_slot()
                self._terminate_llama_process()
                await self._ensure_llama_subprocess_alive_unlocked(
                    tier2_overflow=(self.state == EngineState.HYBRID_RAM_OVERFLOW)
                )
            self._record_transition(
                self.state,
                self.state,
                f"Switched LLM profile from {old_profile} to {profile_id} ({self.active_profile.display_name})",
            )
            return self.get_telemetry_snapshot()

    def get_telemetry_snapshot(self, blender_status: str = "Ready") -> Dict[str, Any]:
        stats = self.get_gpu_stats()
        profile = self.active_profile
        state = self.state
        kv_tier = self.kv_cache_tier

        mode_label_map = {
            EngineState.LLM_ACTIVE: f"{profile.display_name}",
            EngineState.HYBRID_RAM_OVERFLOW: f"{profile.id} [96k RAM Overflow]",
            EngineState.SWAPPING_TO_IMAGEGEN: "Swapping -> FLUX 2D",
            EngineState.IMAGEGEN_ACTIVE: f"{IMAGE_MODELS[self.active_image_model_id]['display_name']} [VRAM]",
            EngineState.SWAPPING_TO_AUDIOGEN: "Swapping -> Stable Audio 1.0",
            EngineState.AUDIOGEN_ACTIVE: f"{AUDIO_MODELS[self.active_audio_model_id]['display_name']} [VRAM]",
            EngineState.RESTORING_LLM: "Restoring LLM Slot 0...",
            EngineState.GODOT_YIELDED: "Godot 3D Play-Mode Yielded",
        }
        engine_label = mode_label_map.get(state, profile.display_name)
        used_gb = stats.used_vram_mb / 1024.0
        total_gb = stats.total_vram_mb / 1024.0
        status_bar_text = (
            f"VRAM: {used_gb:.1f}/{total_gb:.0f}GB | "
            f"Engine: {engine_label} | "
            f"{kv_tier.value} | "
            f"Blender: {blender_status}"
        )

        return {
            "state": state.value,
            "kv_cache_tier": kv_tier.value,
            "status_bar_text": status_bar_text,
            "active_profile_id": self.active_profile_id,
            "active_profile": {
                "id": profile.id,
                "display_name": profile.display_name,
                "model_filename": profile.model_filename,
                "mmproj_filename": profile.mmproj_filename,
                "context_window": (
                    profile.overflow_ctx_tier2
                    if state == EngineState.HYBRID_RAM_OVERFLOW
                    else profile.default_ctx_tier1
                ),
            },
            "available_profiles": [
                {"id": k, "display_name": v.display_name, "model_filename": v.model_filename}
                for k, v in MODEL_PROFILES.items()
            ],
            "active_image_model": IMAGE_MODELS[self.active_image_model_id],
            "active_audio_model": AUDIO_MODELS[self.active_audio_model_id],
            "kv_hierarchy": {
                "active_tier": kv_tier.value,
                "tier1_vram_ceiling_tokens": self.VRAM_KV_CEILING_TOKENS,
                "tier2_cram_mb": self.CRAM_POOL_MB,
                "tier2_cache_reuse_tokens": self.CACHE_REUSE_TOKENS,
                "tier3_disk_cache": self.disk_kv_manager.get_disk_cache_stats(
                    active_session_key=self.active_session_key
                ),
            },
            "gpu": {
                "name": stats.gpu_name,
                "total_vram_mb": stats.total_vram_mb,
                "used_vram_mb": stats.used_vram_mb,
                "free_vram_mb": stats.free_vram_mb,
                "used_vram_gb": round(used_gb, 2),
                "total_vram_gb": round(total_gb, 1),
                "util_percent": stats.gpu_util_percent,
                "temperature_c": stats.temperature_c,
                "nvml_available": stats.nvml_available,
            },
            "system_ram": {
                "total_gb": stats.system_ram_total_gb,
                "used_gb": stats.system_ram_used_gb,
            },
            "godot_preview": {
                "vram_mb": stats.godot_preview_vram_mb,
                "pids": stats.godot_preview_pids,
                "yield_threshold_mb": self.TIER5_GODOT_VRAM_YIELD_MB,
            },
            "estimated_context_tokens": self._estimated_context_tokens,
            "pre_compaction_flush_threshold": self.PRE_COMPACTION_FLUSH_THRESHOLD_TOKENS,
            "tier2_overflow_threshold": self.TIER2_OVERFLOW_THRESHOLD_TOKENS,
            "memory_rag": self.memory_rag.get_memory_stats(),
            "last_memory_flush": self._last_memory_flush_meta,
            "llama_cmd_preview": " ".join(
                self.build_llama_server_command(
                    tier2_overflow=(state == EngineState.HYBRID_RAM_OVERFLOW)
                )
            ),
            "last_slot_save": self._last_slot_save_meta,
            "transitions": self._transition_history[-20:],
        }

    # -------------------------------------------------------------------------
    # First-Run Setup & Health Check
    # -------------------------------------------------------------------------
    def run_setup_health_check(
        self, blender_path: Optional[str] = None, godot_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inspect local binaries, GGUF weights, CUDA availability, Blender 4.x+, and Godot 4.x+.
        """
        llama_path = self.get_llama_server_path()
        sd_path = self.get_sd_binary_path()

        models_status: Dict[str, Any] = {}
        for pid, prof in MODEL_PROFILES.items():
            mpath = self.models_dir / prof.model_filename
            mmpath = (
                self.models_dir / prof.mmproj_filename
                if prof.mmproj_filename
                else None
            )
            models_status[pid] = {
                "id": pid,
                "display_name": prof.display_name,
                "model_path": str(mpath),
                "model_exists": mpath.exists(),
                "model_size_mb": round(mpath.stat().st_size / (1024 ** 2), 1) if mpath.exists() else 0,
                "mmproj_path": str(mmpath) if mmpath else None,
                "mmproj_exists": mmpath.exists() if mmpath else False,
                "hf_model_url": prof.hf_model_url,
                "hf_mmproj_url": prof.hf_mmproj_url,
            }

        image_models_status: Dict[str, Any] = {}
        for iid, imod in IMAGE_MODELS.items():
            ipath = self.models_dir / imod["filename"]
            image_models_status[iid] = {
                "id": iid,
                "display_name": imod["display_name"],
                "path": str(ipath),
                "exists": ipath.exists(),
                "size_mb": round(ipath.stat().st_size / (1024 ** 2), 1) if ipath.exists() else 0,
                "hf_url": imod["hf_url"],
            }

        audio_models_status: Dict[str, Any] = {}
        for aid, amod in AUDIO_MODELS.items():
            apath = self.models_dir / amod["local_dir"]
            has_weights = apath.exists() and any(apath.rglob("*.safetensors"))
            audio_models_status[aid] = {
                "id": aid,
                "display_name": amod["display_name"],
                "repo_id": amod["repo_id"],
                "path": str(apath),
                "exists": has_weights,
                "hf_url": amod["hf_url"],
                "gated_note": "Requires Hugging Face license acceptance & HF_TOKEN for stabilityai/stable-audio-open-1.0",
            }

        gpu_stats = self.get_gpu_stats()

        return {
            "binaries": {
                "llama_server": {
                    "id": "llama-server",
                    "path": str(llama_path),
                    "exists": llama_path.exists(),
                    "release_url": PORTABLE_BINARIES["llama-server"]["release_url"],
                },
                "stable_diffusion": {
                    "id": "sd",
                    "path": str(sd_path),
                    "exists": sd_path.exists(),
                    "release_url": PORTABLE_BINARIES["sd"]["release_url"],
                },
                "blender": {
                    "id": "blender",
                    "path": blender_path,
                    "exists": bool(blender_path and Path(blender_path).exists()),
                },
                "godot": {
                    "id": "godot",
                    "path": godot_path,
                    "exists": bool(godot_path and Path(godot_path).exists()),
                },
            },
            "llm_models": models_status,
            "image_models": image_models_status,
            "audio_models": audio_models_status,
            "download_jobs": self.get_download_jobs_status(),
            "directories": {
                "bin_dir": str(self.bin_dir),
                "models_dir": str(self.models_dir),
                "slots_dir": str(self.slots_dir),
            },
            "gpu": {
                "name": gpu_stats.gpu_name,
                "total_vram_mb": gpu_stats.total_vram_mb,
                "free_vram_mb": gpu_stats.free_vram_mb,
                "nvml_available": gpu_stats.nvml_available,
            },
        }

    # -------------------------------------------------------------------------
    # Built-In Streaming Model & Portable CUDA Binary Downloader
    # -------------------------------------------------------------------------
    def get_download_jobs_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._download_jobs.items()}

    def _update_download_job(
        self,
        job_key: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        **updates: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            job = self._download_jobs.setdefault(job_key, {"job_key": job_key})
            job.update(updates)
            job["updated_at"] = time.time()
            snap = dict(job)
        if progress_cb:
            try:
                progress_cb(snap)
            except Exception:
                pass
        return snap

    async def _stream_file_download(
        self,
        job_key: str,
        url: str,
        dest_path: Path,
        hf_token: Optional[str] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Path:
        """
        Stream a large GGUF or binary asset to disk with HTTP Range resume support (.part file).
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0

        headers: Dict[str, str] = {"User-Agent": "Godot4-Local-Harness/1.0"}
        token = hf_token or os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {token}"
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        self._update_download_job(
            job_key,
            progress_cb,
            status="downloading",
            current_file=dest_path.name,
            url=url,
            downloaded_mb=round(existing_bytes / (1024 ** 2), 2),
        )

        start_t = time.time()
        last_emit = 0.0

        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"HTTP {resp.status_code} Unauthorized/Gated for {url}. "
                        "Provide a valid HF_TOKEN (and accept the repo terms on huggingface.co if gated)."
                    )
                if resp.status_code == 416:
                    # Range already satisfied
                    if part_path.exists():
                        part_path.replace(dest_path)
                    return dest_path

                resp.raise_for_status()
                is_resume = resp.status_code == 206
                mode = "ab" if is_resume else "wb"
                if not is_resume:
                    existing_bytes = 0

                content_len = int(resp.headers.get("Content-Length", "0") or 0)
                total_bytes = (existing_bytes + content_len) if content_len > 0 else 0
                downloaded = existing_bytes

                with open(part_path, mode) as fh:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 512):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_emit >= 0.35 or (total_bytes and downloaded >= total_bytes):
                            last_emit = now
                            elapsed = max(0.001, now - start_t)
                            speed_mbps = ((downloaded - existing_bytes) / (1024 ** 2)) / elapsed
                            pct = round((downloaded / total_bytes) * 100.0, 1) if total_bytes > 0 else 0.0
                            self._update_download_job(
                                job_key,
                                progress_cb,
                                status="downloading",
                                current_file=dest_path.name,
                                downloaded_mb=round(downloaded / (1024 ** 2), 2),
                                total_mb=round(total_bytes / (1024 ** 2), 2) if total_bytes else None,
                                percent=pct,
                                speed_mbps=round(speed_mbps, 2),
                            )

        part_path.replace(dest_path)
        return dest_path

    async def _download_github_cuda_binary(
        self,
        job_key: str,
        item_id: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Query GitHub Releases API for llama.cpp or stable-diffusion.cpp, locate the newest release
        that has finished uploading Windows x64 CUDA zip assets, download it, and extract .exe + .dll files into ./bin/.
        """
        import zipfile

        repo = (
            "ggml-org/llama.cpp"
            if item_id in ("llama-server", "llama_server")
            else "leejet/stable-diffusion.cpp"
        )
        api_url = f"https://api.github.com/repos/{repo}/releases?per_page=10"
        self._update_download_job(
            job_key,
            progress_cb,
            status="resolving_release",
            message=f"Querying {repo} releases for Windows CUDA x64 assets...",
        )

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(api_url, headers={"User-Agent": "Godot4-Local-Harness/1.0"})
            resp.raise_for_status()
            releases_list = resp.json()
            if isinstance(releases_list, dict):
                releases_list = [releases_list]

        chosen_assets: List[Dict[str, Any]] = []
        chosen_release: Dict[str, Any] = {}

        for rel in releases_list:
            assets = rel.get("assets", [])
            if not assets:
                continue

            if "llama.cpp" in repo:
                cuda_bins = [
                    a for a in assets
                    if a.get("name", "").lower().endswith(".zip")
                    and "win" in a.get("name", "").lower()
                    and "x64" in a.get("name", "").lower()
                    and "cuda" in a.get("name", "").lower()
                    and not a.get("name", "").lower().startswith("cudart")
                ]
                cudart_bins = [
                    a for a in assets
                    if a.get("name", "").lower().endswith(".zip")
                    and "win" in a.get("name", "").lower()
                    and "x64" in a.get("name", "").lower()
                    and a.get("name", "").lower().startswith("cudart")
                ]
                if cuda_bins:
                    # Prefer CUDA 12.x if multiple CUDA versions are present, otherwise take the first CUDA x64 zip
                    cu12 = [a for a in cuda_bins if "12." in a.get("name", "").lower() or "cu12" in a.get("name", "").lower()]
                    primary_bin = cu12[0] if cu12 else cuda_bins[0]
                    chosen_assets = [primary_bin]
                    if cudart_bins:
                        cu12_rt = [a for a in cudart_bins if "12." in a.get("name", "").lower() or "cu12" in a.get("name", "").lower()]
                        chosen_assets.append(cu12_rt[0] if cu12_rt else cudart_bins[0])
                    chosen_release = rel
                    break

                # Fallback to Vulkan or CPU x64 zip if no CUDA zip in this release
                fallback_bins = [
                    a for a in assets
                    if a.get("name", "").lower().endswith(".zip")
                    and "win" in a.get("name", "").lower()
                    and "x64" in a.get("name", "").lower()
                    and ("vulkan" in a.get("name", "").lower() or "cpu" in a.get("name", "").lower() or "avx2" in a.get("name", "").lower())
                ]
                if fallback_bins:
                    chosen_assets = [fallback_bins[0]]
                    chosen_release = rel
                    break
            else:
                for a in assets:
                    name_l = a.get("name", "").lower()
                    if name_l.endswith(".zip") and "win" in name_l and "x64" in name_l and ("cuda" in name_l or "cudart" in name_l):
                        chosen_assets.append(a)
                if not chosen_assets:
                    for a in assets:
                        name_l = a.get("name", "").lower()
                        if name_l.endswith(".zip") and "win" in name_l and "x64" in name_l:
                            chosen_assets.append(a)
                            break
                if chosen_assets:
                    chosen_release = rel
                    break

        if not chosen_assets:
            raise RuntimeError(f"Could not find Windows x64 .zip asset in recent {repo} releases.")

        extracted_files: List[str] = []
        for asset in chosen_assets[:2]:
            dl_url = asset["browser_download_url"]
            zip_path = self.cache_dir / asset["name"]
            await self._stream_file_download(
                job_key=job_key,
                url=dl_url,
                dest_path=zip_path,
                progress_cb=progress_cb,
            )
            self._update_download_job(
                job_key,
                progress_cb,
                status="extracting",
                current_file=asset["name"],
                message=f"Extracting {asset['name']} to {self.bin_dir}...",
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    fname = Path(member.filename).name
                    if fname.lower().endswith((".exe", ".dll")):
                        target_file = self.bin_dir / fname
                        with zf.open(member) as src, open(target_file, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted_files.append(fname)
            try:
                zip_path.unlink()
            except Exception:
                pass

        # Ensure sd.exe alias exists if sd-cli.exe was extracted
        sd_cli = self.bin_dir / "sd-cli.exe"
        sd_exe = self.bin_dir / "sd.exe"
        if sd_cli.exists() and not sd_exe.exists():
            try:
                shutil.copy2(sd_cli, sd_exe)
                extracted_files.append("sd.exe")
            except Exception:
                pass

        return {
            "repo": repo,
            "tag": chosen_release.get("tag_name"),
            "extracted_files": extracted_files,
            "bin_dir": str(self.bin_dir),
        }

    async def start_download_job(
        self,
        category: str,
        item_id: str,
        hf_token: Optional[str] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an asynchronous download of a model profile, image model, audio model, or binary.
        """
        job_key = f"{category}:{item_id}"
        with self._lock:
            existing = self._download_jobs.get(job_key, {})
            if existing.get("status") in ("downloading", "extracting", "resolving_release"):
                return dict(existing)

        self._update_download_job(
            job_key,
            progress_cb,
            category=category,
            item_id=item_id,
            status="starting",
            percent=0.0,
            downloaded_mb=0.0,
            error=None,
        )

        try:
            if category == "llm" and item_id in MODEL_PROFILES:
                prof = MODEL_PROFILES[item_id]
                downloaded_paths: List[str] = []
                model_dest = self.models_dir / prof.model_filename
                if not (model_dest.exists() and model_dest.stat().st_size > 50 * 1024 * 1024):
                    await self._stream_file_download(
                        job_key=job_key,
                        url=prof.hf_model_url,
                        dest_path=model_dest,
                        hf_token=hf_token,
                        progress_cb=progress_cb,
                    )
                downloaded_paths.append(str(model_dest))

                if prof.mmproj_filename and prof.hf_mmproj_url:
                    mmproj_dest = self.models_dir / prof.mmproj_filename
                    if not (mmproj_dest.exists() and mmproj_dest.stat().st_size > 1024 * 1024):
                        try:
                            await self._stream_file_download(
                                job_key=job_key,
                                url=prof.hf_mmproj_url,
                                dest_path=mmproj_dest,
                                hf_token=hf_token,
                                progress_cb=progress_cb,
                            )
                            downloaded_paths.append(str(mmproj_dest))
                        except Exception as mm_exc:
                            logger.info(
                                f"Optional mmproj weight ({prof.mmproj_filename}) not present at {prof.hf_mmproj_url}: {mm_exc}; continuing with main GGUF."
                            )
                    else:
                        downloaded_paths.append(str(mmproj_dest))

                return self._update_download_job(
                    job_key,
                    progress_cb,
                    status="completed",
                    percent=100.0,
                    files=downloaded_paths,
                    message=f"Ready: {prof.model_filename} in {self.models_dir}",
                )

            if category == "image_model" and item_id in IMAGE_MODELS:
                imod = IMAGE_MODELS[item_id]
                dest = self.models_dir / imod["filename"]
                if not (dest.exists() and dest.stat().st_size > 50 * 1024 * 1024):
                    await self._stream_file_download(
                        job_key=job_key,
                        url=imod["hf_url"],
                        dest_path=dest,
                        hf_token=hf_token,
                        progress_cb=progress_cb,
                    )
                return self._update_download_job(
                    job_key,
                    progress_cb,
                    status="completed",
                    percent=100.0,
                    files=[str(dest)],
                    message=f"Ready: {imod['display_name']} in {dest}",
                )

            if category == "audio_model" and item_id in AUDIO_MODELS:
                amod = AUDIO_MODELS[item_id]
                target_dir = self.models_dir / amod["local_dir"]
                target_dir.mkdir(parents=True, exist_ok=True)
                token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None
                primary_repo = amod["repo_id"]
                fallback_repo = amod.get("fallback_repo_id", "cudabenchmarktest/stable-audio-open-1.0")
                chosen_repo = primary_repo if token else fallback_repo

                self._update_download_job(
                    job_key,
                    progress_cb,
                    status="downloading",
                    current_file=chosen_repo,
                    message=f"Downloading {chosen_repo} snapshot into {target_dir}...",
                )

                def _run_hf_snapshot() -> str:
                    from huggingface_hub import snapshot_download  # type: ignore

                    try:
                        return snapshot_download(
                            repo_id=chosen_repo,
                            local_dir=str(target_dir),
                            token=token if chosen_repo == primary_repo else None,
                            ignore_patterns=["*.ckpt", "*.bin"],
                        )
                    except Exception as first_err:
                        if chosen_repo != fallback_repo:
                            logger.info(f"Gated repo {primary_repo} failed ({first_err}); falling back to ungated mirror {fallback_repo}...")
                            return snapshot_download(
                                repo_id=fallback_repo,
                                local_dir=str(target_dir),
                                token=None,
                                ignore_patterns=["*.ckpt", "*.bin"],
                            )
                        raise

                snap_path = await asyncio.to_thread(_run_hf_snapshot)
                return self._update_download_job(
                    job_key,
                    progress_cb,
                    status="completed",
                    percent=100.0,
                    files=[str(snap_path)],
                    message=f"Downloaded {amod['display_name']} to {target_dir}",
                )

            if category == "binary" and item_id in ("llama-server", "llama_server", "sd", "stable_diffusion"):
                norm_id = "llama-server" if "llama" in item_id else "sd"
                res = await self._download_github_cuda_binary(
                    job_key=job_key,
                    item_id=norm_id,
                    progress_cb=progress_cb,
                )
                return self._update_download_job(
                    job_key,
                    progress_cb,
                    status="completed",
                    percent=100.0,
                    files=res["extracted_files"],
                    message=f"Extracted {len(res['extracted_files'])} CUDA binaries/DLLs into {self.bin_dir}",
                )

            raise ValueError(f"Unknown download item: {category}/{item_id}")

        except Exception as exc:
            logger.warning(f"Download job {job_key} failed: {exc}")
            return self._update_download_job(
                job_key,
                progress_cb,
                status="error",
                error=str(exc),
                message=f"Download failed: {exc}",
            )

