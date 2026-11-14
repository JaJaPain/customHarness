# CustomHarness — Standalone Local-First Godot 4 Agentic Coding & 2D/3D/Audio Asset Harness

A standalone, zero-wrapper local agentic coding, 2D texture/sprite, 3D headless Blender `.glb`, and 2-in-1 SFX/Music generation harness built for **Godot 4** game development on **Windows 11** (Hardware target: **NVIDIA RTX 3060 12GB VRAM, 32GB DDR4 System RAM, Intel i5-10400F**).

Runs directly on portable Windows CUDA binaries (`llama-server.exe` + `sd.exe`), headless **Blender 4.x+**, and **Stable Audio Open 1.0 (FP16)** with zero third-party wrapper apps (no Ollama, LM Studio, or ComfyUI).

---

## Core Architecture

1. **Mutually Exclusive GPU Engines & 3-Tier Lazy KV Cache (`vram_orchestrator.py`)**
   - **Tier 1 (VRAM First — Default):** Keeps 100% of `Ornith-1.5-9B-Q6_K.gguf` + `mmproj-Ornith-1.5-9B-f16.gguf` (`-ngl 99`) and a 32,768-token `q8_0` KV cache inside the 12GB RTX 3060 VRAM.
   - **Tier 2 (Lazy Host-RAM Spillover via `--cram 8192`):** Uses an 8GB host-memory prompt cache pool (`--cram 8192 --cache-reuse 256`) and automatic 96k context expansion (`HYBRID_RAM_OVERFLOW` with `--fit on`) into 32GB DDR4 RAM when context exceeds 26,000 tokens.
   - **Tier 3 (NVMe Disk Persistence & 15GB LRU Eviction):** `DiskKVCacheManager` hashes `project_root + system_prompt + conversation_id` and persists/restores KV slots to `./cache/kv_disk/` (`POST /slots/0?action=save` / `restore`) across image/audio swaps, session switches, and harness shutdown.
   - **Batched Asset VRAM Swaps:** Automatically keeps `llama-server.exe` unloaded across consecutive `generate_2d_asset` or `generate_game_audio` calls in the same turn (`defer_llm_restore=True`) and restores the LLM KV slot once before the vision verification turn.
   - **Godot 3D Play-Mode VRAM Yield (`GODOT_YIELDED`):** Detects `Godot_v4*` preview windows exceeding 2.5GB VRAM and yields the GPU automatically.

2. **Zero-VRAM Pre-Compaction Memory Flush & On-Demand RAG (`memory_rag.py`)**
   - Uses a CPU-only 384-d embedder (`BAAI/bge-small-en-v1.5` on CPU, **0 bytes VRAM**) + SQLite hybrid vector/BM25 store (`./cache/project_memory.db`) scoped per Godot project hash.
   - At ~25,000 tokens, prunes raw `read_file` dumps to `[Read <path> lines X-Y]`, indexes conversation chunks into SQLite, and extracts durable project facts into `.harness/MEMORY.md` (pinned at `CONVERSATION_HISTORY[0]` so KV prefix caching is never invalidated).

3. **2D Image-Gen + Closed-Loop Vision Self-Check (`image_gen_worker.py`)**
   - Runs 4-step 1024×1024 `FLUX.1-schnell-Q5_K_S.gguf` via `stable-diffusion.cpp` (`sd.exe`), with corner-color background flood-fill removal for sprites (`sprite_rgba`), normal/roughness/height + Godot `.tres` `StandardMaterial3D` generation for seamless textures (`generate_pbr_maps=True`), and automatic VLM vision inspection.

4. **Headless Blender 4.x+ 3D Pipeline (`blender_worker.py`)**
   - Wraps `bpy` scripts in the **Godot 4 Export Harness Wrapper** (`1.0m` metric scale, `res://` texture path resolution, modifier application before transform freeze, `-colonly` collision suffix preservation, `+Y`-up binary `.glb` export, and a **3-Angle Visual Verification Contact Sheet**).

5. **2-in-1 Sound Effects & Instrumental Music Generator (`audio_gen_worker.py`)**
   - Powered by `stabilityai/stable-audio-open-1.0` (`StableAudioPipeline` FP16) with `mode="sfx"` (0.5s–10.0s, silence trim, `-1.0 dBFS` normalization) and `mode="music_loop"` (10.0s–45.0s, strictly instrumental tags, equal-power cosine crossfade loop smoothing, `.import` sidecar with `loop=true`, and **Waveform + 64-Band Mel-Spectrogram PNG** vision check).

---

## Quick Start (Windows 11)

1. **Launch the Harness:**
   ```bat
   start_harness.bat
   ```
   Or manually:
   ```powershell
   pip install -r requirements.txt
   python server.py
   ```
   Then open `http://127.0.0.1:8000`.

2. **Download Models & Portable CUDA Binaries (`./models` & `./bin`):**
   - **In the Web UI:** Click **`⚙ Setup & Health`** in the top bar and click **`⬇ Download All Default Models & Binaries`** (or download individual profiles).
   - **From the CLI:**
     ```powershell
     python download_models.py --check
     python download_models.py --all --hf-token hf_YOUR_TOKEN
     ```
