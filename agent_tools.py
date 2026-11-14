"""
Agentic Coding Tools, Godot 4 Workflow Integration, Diff/Rollback Engine,
and <think>...</think> Reasoning + Tool-Call Parser.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from audio_gen_worker import AudioGenWorker
from blender_worker import BlenderWorker
from godot_parser import GodotParser
from image_gen_worker import ImageGenWorker
from memory_rag import ProjectMemoryRAG

logger = logging.getLogger("CustomHarness.AgentTools")


# =============================================================================
# OpenAI-Compatible Function Tool Schemas
# =============================================================================
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories inside a workspace path or Godot res:// path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative, absolute, or res://). Defaults to '.'",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to recursively list subdirectories (up to 200 entries).",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file with line numbers and optional start_line/end_line range. "
                "Automatically includes a compact Godot 4 3D Scene Graph Summary for .tscn/.tres/.gd files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read (supports res:// or workspace-relative paths).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-based start line (inclusive).",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based end line (inclusive).",
                        "default": 250,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a regex or literal pattern across workspace files (.gd, .tscn, .tres, .gdshader, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Regex or literal text pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional directory or file path to scope the search.",
                        "default": ".",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob filter such as '*.gd' or '*.tscn'.",
                        "default": "*",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search is case-sensitive.",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_diff",
            "description": (
                "Edit an existing file by replacing one block of text with another. Copy `search_text` from "
                "read_file output WITHOUT the leading line numbers ('  12 | '). Indentation differences are "
                "tolerated, but the lines themselves must match and must be unique in the file. "
                "A backup is saved and .gd files are syntax-checked automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path (res:// or workspace-relative).",
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Existing lines to find (a few whole lines are best).",
                    },
                    "replace_text": {
                        "type": "string",
                        "description": "New lines that replace search_text.",
                    },
                },
                "required": ["path", "search_text", "replace_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file or overwrite an existing file (automatically saving a .bak backup "
                "snapshot if overwriting and running Godot --check-only validation on .gd scripts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path (res:// or workspace-relative).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_input_action",
            "description": (
                "Create or replace an input action in project.godot with correctly formatted key / mouse "
                "bindings. ALWAYS use this instead of hand-editing the [input] section. An action with no "
                "bindings never fires, so every action used by Input.is_action_pressed()/get_axis() needs this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action name used in scripts, e.g. 'move_left' or 'launch_ball'.",
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Keyboard keys: letters 'a'-'z', digits '0'-'9', 'space', 'enter', 'escape', 'tab', "
                            "'shift', 'ctrl', 'alt', 'left', 'right', 'up', 'down', 'backspace', 'f1'-'f12'."
                        ),
                    },
                    "mouse_buttons": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["left", "right", "middle"]},
                        "description": "Optional mouse buttons that also trigger the action.",
                    },
                },
                "required": ["action", "keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the project folder. The command `godot --headless --quit` is special: "
                "it validates every scene and script AND runs the game headless for a moment, reporting "
                "parse errors and runtime errors. Run it after every change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command line or Godot CLI invocation.",
                    },
                    "check_gdscript_path": {
                        "type": "string",
                        "description": "Optional path to a .gd script to validate with `godot --headless --check-only -s`.",
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Execution timeout in seconds.",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_2d_asset",
            "description": (
                "Trigger Tier 3 Atomic Image-Gen VRAM Swap (snapshots LLM KV slot 0, unloads llama-server, "
                "runs stable-diffusion.cpp FLUX.1-schnell 4-step 1024x1024 generation, optional rembg background "
                "removal and Sobel PBR normal/roughness maps, restores LLM slot 0, and returns a Closed-Loop "
                "Vision Self-Check preview image so you can inspect the asset before writing Godot code)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed visual prompt for the 2D game asset.",
                    },
                    "asset_type": {
                        "type": "string",
                        "enum": ["sprite", "ui_icon", "seamless_texture", "skybox_concept"],
                        "description": "Category of 2D asset to generate.",
                        "default": "sprite",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Godot res:// output path (e.g. 'res://assets/textures/stone_wall.png').",
                        "default": "res://assets/textures/generated_sprite.png",
                    },
                    "transparent_bg": {
                        "type": "boolean",
                        "description": "Remove background via rembg / local alpha-key for sprites & UI icons.",
                        "default": False,
                    },
                    "generate_pbr_maps": {
                        "type": "boolean",
                        "description": "Generate matching _normal.png, _roughness.png, _height.png, and _material.tres.",
                        "default": False,
                    },
                    "width": {"type": "integer", "default": 1024},
                    "height": {"type": "integer", "default": 1024},
                    "steps": {"type": "integer", "default": 4},
                    "seed": {"type": "integer", "default": -1},
                },
                "required": ["prompt", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_blender_script",
            "description": (
                "Run a headless Blender 4.x Python (bpy) script wrapped inside the Godot 4 Export Harness. "
                "Automatically applies all transforms, enforces 1.0m metric units, converts Z-up to glTF +Y up, "
                "preserves Godot collision suffixes (-col, -colonly, -convcol, -noimp), exports binary .glb + "
                "blend_src/.blend, and renders a 3-Angle Visual Verification Contact Sheet (Front -Y, 3/4 Iso, "
                "Wireframe/Top) injected into your vision input."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script_code": {
                        "type": "string",
                        "description": "Python bpy script creating/modeling/rigging the 3D asset.",
                    },
                    "output_glb_path": {
                        "type": "string",
                        "description": "Target Godot res:// path for the exported binary .glb (e.g. 'res://assets/models/crate.glb').",
                    },
                    "render_preview": {
                        "type": "boolean",
                        "description": "Render the 3-Angle Visual Verification Contact Sheet for closed-loop VLM inspection.",
                        "default": True,
                    },
                    "heavy_cycles_bake": {
                        "type": "boolean",
                        "description": "If True, swaps LLM out of VRAM so Blender Cycles can bake on the 12GB GPU.",
                        "default": False,
                    },
                },
                "required": ["script_code", "output_glb_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search archived project session history, debugging steps, error fixes, and pinned "
                "Godot 4 project facts (.harness/MEMORY.md) using Zero-VRAM CPU embeddings + keyword "
                "hybrid search inside ./cache/project_memory.db. Retrieved snippets are injected at "
                "the bottom of the active conversation so llama-server KV prefix cache is never invalidated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language or keyword query to search archived Godot decisions, "
                            "collision layers, node paths, autoload names, or debugging steps."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of ranked memory snippets to retrieve (default: 4).",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_game_audio",
            "description": (
                "Trigger the Atomic Audio-Gen VRAM Swap (snapshots LLM KV slot 0, unloads llama-server, "
                "runs `stabilityai/stable-audio-open-1.0` via `diffusers` `StableAudioPipeline` in FP16, "
                "unloads the audio pipeline, and restores LLM KV slot 0). "
                "Supports two modes:\n"
                "  - `mode='sfx'`: Short, punchy Foley/UI/weapon/environment sound effects (`duration_sec` 0.5s-10.0s), "
                "automatically trimming trailing silence and normalizing peak loudness to `-1.0 dBFS`.\n"
                "  - `mode='music_loop'`: Instrumental background music beds (`duration_sec` 10.0s-45.0s, strictly "
                "instrumental tags appended automatically), with equal-power crossfade loop smoothing (`seamless_loop=True`) "
                "so tracks loop cleanly in Godot without clicks.\n"
                "Exports 44.1 kHz stereo `.wav` or `.ogg` + matching Godot 4 `.import` sidecar (`loop=true` when `seamless_loop=True`), "
                "and returns a compact **Waveform + Mel-Spectrogram PNG** alongside Peak/RMS dBFS metrics for Closed-Loop Vision Self-Check."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the sound effect or instrumental music bed.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sfx", "music_loop"],
                        "description": "Audio generation mode: 'sfx' (0.5s-10.0s) or 'music_loop' (10.0s-45.0s).",
                        "default": "sfx",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Target Godot res:// path (e.g. 'res://audio/sfx/laser_shot.wav' or 'res://audio/music/cyber_loop.ogg').",
                        "default": "res://audio/sfx/generated_sfx.wav",
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "Target duration in seconds (clamped to 0.5-10.0s for 'sfx', 10.0-45.0s for 'music_loop').",
                        "default": 2.5,
                    },
                    "seamless_loop": {
                        "type": "boolean",
                        "description": "Apply equal-power loop crossfade smoothing and set `loop=true` in the Godot `.import` sidecar.",
                        "default": False,
                    },
                    "steps": {"type": "integer", "default": 50},
                    "cfg_scale": {"type": "number", "default": 6.5},
                    "seed": {"type": "integer", "default": -1},
                },
                "required": ["prompt", "mode", "output_path"],
            },
        },
    },
]


# =============================================================================
# <think>...</think> Reasoning Block & Tool-Call JSON Parser
# =============================================================================
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK_RE = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\}|\[.*?\])\s*</tool_call>", re.DOTALL)
FENCED_TOOL_JSON_RE = re.compile(
    r"```(?:json|tool_call)?\s*(\{\s*\"(?:name|tool|function)\"\s*:.*?\}|\[\s*\{\s*\"(?:name|tool|function)\"\s*:.*?\}\s*\])\s*```",
    re.DOTALL,
)


def _normalize_single_tool_dict(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):
        fn = item["function"]
        name = fn.get("name") or item.get("name")
        args = fn.get("arguments") or item.get("arguments") or {}
    else:
        name = item.get("name") or item.get("tool") or item.get("function")
        args = item.get("arguments") or item.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not name or not isinstance(name, str):
        return None
    return {
        "id": item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
        "name": name,
        "arguments": args if isinstance(args, dict) else {},
    }


def parse_thinking_and_tool_calls(
    raw_content: str,
    api_tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Cleanly extract <think>...</think> blocks into a separate `thinking` field
    BEFORE parsing any tool calls so reasoning tokens never corrupt tool-call JSON.
    Supports:
      1. Native OpenAI `tool_calls` array from llama-server (--jinja)
      2. Single or multiple <tool_call>{...}</tool_call> or <tool_call>[...]</tool_call> tags
      3. Fenced JSON tool call objects or arrays outside <think>...</think>
    """
    raw_content = raw_content or ""
    thinking_parts: List[str] = []

    for match in THINK_BLOCK_RE.finditer(raw_content):
        thinking_parts.append(match.group(1).strip())

    cleaned_content = THINK_BLOCK_RE.sub("", raw_content)

    # Check if there's an unclosed <think> block (e.g. mid-stream or truncated)
    unclosed = UNCLOSED_THINK_RE.search(cleaned_content)
    if unclosed:
        thinking_parts.append(unclosed.group(1).strip())
        cleaned_content = UNCLOSED_THINK_RE.sub("", cleaned_content)

    normalized_tool_calls: List[Dict[str, Any]] = []
    if api_tool_calls:
        for tc in api_tool_calls:
            norm = _normalize_single_tool_dict(tc)
            if norm:
                normalized_tool_calls.append(norm)

    # Also parse inline <tool_call>{...}</tool_call> or <tool_call>[...]</tool_call> blocks outside <think>
    for match in TOOL_CALL_TAG_RE.finditer(cleaned_content):
        try:
            payload = json.loads(match.group(1))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                norm = _normalize_single_tool_dict(item)
                if norm:
                    normalized_tool_calls.append(norm)
        except Exception:
            continue

    cleaned_content = TOOL_CALL_TAG_RE.sub("", cleaned_content)

    if not normalized_tool_calls:
        for match in FENCED_TOOL_JSON_RE.finditer(cleaned_content):
            try:
                payload = json.loads(match.group(1))
                items = payload if isinstance(payload, list) else [payload]
                added = False
                for item in items:
                    norm = _normalize_single_tool_dict(item)
                    if norm:
                        normalized_tool_calls.append(norm)
                        added = True
                if added:
                    cleaned_content = cleaned_content.replace(match.group(0), "")
            except Exception:
                continue

    return {
        "thinking": "\n\n".join(p for p in thinking_parts if p),
        "clean_content": cleaned_content.strip(),
        "tool_calls": normalized_tool_calls,
    }


# =============================================================================
# Agent Tool Executor
# =============================================================================
class AgentToolExecutor:
    """
    Executes the 8 agentic coding & asset-creation tools inside the Godot workspace,
    maintains .bak rollback snapshots, validates GDScript via Godot --headless --check-only,
    and manages command approval gating.
    """

    SEARCH_REPLACE_BLOCK_RE = re.compile(
        r"<<<<<<<\s*SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    )

    def __init__(
        self,
        workspace_dir: str | Path,
        image_worker: ImageGenWorker,
        blender_worker: BlenderWorker,
        require_command_approval: bool = False,
        memory_rag: Optional[ProjectMemoryRAG] = None,
        audio_worker: Optional[AudioGenWorker] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.workspace_dir / ".harness_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.image_worker = image_worker
        self.blender_worker = blender_worker
        self.require_command_approval = require_command_approval
        self.memory_rag = memory_rag
        self.audio_worker = audio_worker

        self._diff_history: List[Dict[str, Any]] = []
        self._pending_commands: Dict[str, Dict[str, Any]] = {}
        self._cached_godot_exe: Optional[str] = None

    def set_workspace_dir(self, new_workspace: str | Path) -> Path:
        """Switch the active Godot 4 project workspace directory across all tool workers."""
        resolved = Path(new_workspace).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = resolved
        self.backup_dir = self.workspace_dir / ".harness_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        if self.image_worker is not None:
            self.image_worker.workspace_dir = resolved
        if self.blender_worker is not None:
            self.blender_worker.workspace_dir = resolved
        if self.audio_worker is not None:
            self.audio_worker.workspace_dir = resolved
        return resolved

    # -------------------------------------------------------------------------
    # Path Resolution & Godot Binary Detection
    # -------------------------------------------------------------------------
    def resolve_workspace_path(self, raw_path: str) -> Path:
        cleaned = (raw_path or ".").strip()
        if cleaned.startswith("res://"):
            cleaned = cleaned[len("res://") :]
        cleaned = cleaned.lstrip("/\\")
        if not cleaned or cleaned == ".":
            return self.workspace_dir
        candidate = (self.workspace_dir / cleaned).resolve()
        if not str(candidate).startswith(str(self.workspace_dir)):
            raise PermissionError(f"Path '{raw_path}' escapes workspace boundary '{self.workspace_dir}'.")
        return candidate

    def to_res_path(self, abs_path: Path) -> str:
        try:
            rel = abs_path.resolve().relative_to(self.workspace_dir)
            rel_str = str(rel).replace("\\", "/")
            return "res://" if rel_str == "." else f"res://{rel_str}"
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def detect_godot_executable(self) -> Optional[str]:
        if self._cached_godot_exe and Path(self._cached_godot_exe).exists():
            return self._cached_godot_exe

        env_godot = os.environ.get("GODOT_PATH") or os.environ.get("GODOT4_BIN")
        if env_godot and Path(env_godot).exists():
            self._cached_godot_exe = str(Path(env_godot).resolve())
            return self._cached_godot_exe

        for bin_name in (
            "godot",
            "godot4",
            "Godot_v4.exe",
            "Godot_v4.6.3-stable_win64_console.exe",
            "Godot_v4.6.3-stable_win64.exe",
            "Godot_v4.3-stable_win64_console.exe",
            "Godot_v4.3-stable_win64.exe",
        ):
            which_g = shutil.which(bin_name)
            if which_g:
                self._cached_godot_exe = str(Path(which_g).resolve())
                return self._cached_godot_exe

        search_dirs = [
            Path(r"D:\CodingProjects\spacegame\Godot"),
            Path(r"D:\Godot"),
            Path(r"C:\Godot"),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Godot",
            Path(r"D:\Program Files\Godot"),
            Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local")) / "Godot",
            Path.home() / "Desktop",
            Path.home() / "Downloads",
        ]
        for d in search_dirs:
            try:
                if not d.exists():
                    continue
                for exe in sorted(d.glob("Godot_v4*.exe"), reverse=True):
                    if "console" in exe.name.lower():
                        self._cached_godot_exe = str(exe.resolve())
                        return self._cached_godot_exe
                for exe in sorted(d.glob("Godot_v4*.exe"), reverse=True):
                    self._cached_godot_exe = str(exe.resolve())
                    return self._cached_godot_exe
            except Exception:
                continue

        return None

    # -------------------------------------------------------------------------
    # Unified Tool Dispatcher
    # -------------------------------------------------------------------------
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        approved: bool = False,
        defer_llm_restore: bool = False,
    ) -> Dict[str, Any]:
        """Dispatch a tool call by name and return structured JSON + optional vision attachments."""
        try:
            if tool_name == "list_dir":
                return self.tool_list_dir(
                    path=arguments.get("path", "."),
                    recursive=bool(arguments.get("recursive", False)),
                )
            if tool_name == "read_file":
                return self.tool_read_file(
                    path=arguments["path"],
                    start_line=int(arguments.get("start_line", 1)),
                    end_line=int(arguments.get("end_line", 250)),
                )
            if tool_name == "grep_search":
                return self.tool_grep_search(
                    query=arguments["query"],
                    path=arguments.get("path", "."),
                    file_pattern=arguments.get("file_pattern", "*"),
                    case_sensitive=bool(arguments.get("case_sensitive", False)),
                )
            if tool_name == "apply_diff":
                return await self.tool_apply_diff(
                    path=arguments["path"],
                    search_text=arguments.get("search_text"),
                    replace_text=arguments.get("replace_text"),
                    unified_diff=arguments.get("unified_diff"),
                )
            if tool_name == "write_file":
                return await self.tool_write_file(
                    path=arguments["path"],
                    content=arguments["content"],
                )
            if tool_name == "set_input_action":
                return self.tool_set_input_action(
                    action=str(arguments["action"]),
                    keys=arguments.get("keys") or [],
                    mouse_buttons=arguments.get("mouse_buttons") or [],
                )
            if tool_name == "run_command":
                return await self.tool_run_command(
                    command=arguments["command"],
                    check_gdscript_path=arguments.get("check_gdscript_path"),
                    timeout_sec=int(arguments.get("timeout_sec", 30)),
                    approved=approved,
                )
            if tool_name == "generate_2d_asset":
                return await self.image_worker.generate_2d_asset(
                    prompt=arguments["prompt"],
                    asset_type=arguments.get("asset_type", "sprite"),
                    output_path=arguments.get(
                        "output_path", "res://assets/textures/generated_sprite.png"
                    ),
                    width=int(arguments.get("width", 1024)),
                    height=int(arguments.get("height", 1024)),
                    steps=int(arguments.get("steps", 4)),
                    transparent_bg=bool(arguments.get("transparent_bg", False)),
                    generate_pbr_maps=bool(arguments.get("generate_pbr_maps", False)),
                    seed=int(arguments.get("seed", -1)),
                    conversation_history=conversation_history,
                    defer_llm_restore=defer_llm_restore,
                )
            if tool_name == "run_blender_script":
                return await self.blender_worker.run_blender_script(
                    script_code=arguments["script_code"],
                    output_glb_path=arguments["output_glb_path"],
                    render_preview=bool(arguments.get("render_preview", True)),
                    heavy_cycles_bake=bool(arguments.get("heavy_cycles_bake", False)),
                    conversation_history=conversation_history,
                )
            if tool_name == "recall_memory":
                return self.tool_recall_memory(
                    query=str(arguments.get("query", "")),
                    top_k=int(arguments.get("top_k", 4)),
                )
            if tool_name == "generate_game_audio":
                if self.audio_worker is None:
                    return {
                        "status": "error",
                        "tool": "generate_game_audio",
                        "error": "AudioGenWorker is not configured on AgentToolExecutor.",
                    }
                return await self.audio_worker.generate_game_audio(
                    prompt=arguments["prompt"],
                    mode=arguments.get("mode", "sfx"),
                    output_path=arguments.get(
                        "output_path", "res://audio/sfx/generated_sfx.wav"
                    ),
                    duration_sec=(
                        float(arguments["duration_sec"])
                        if "duration_sec" in arguments and arguments["duration_sec"] is not None
                        else None
                    ),
                    seamless_loop=(
                        bool(arguments["seamless_loop"])
                        if "seamless_loop" in arguments and arguments["seamless_loop"] is not None
                        else None
                    ),
                    steps=int(arguments.get("steps", 50)),
                    cfg_scale=float(arguments.get("cfg_scale", 6.5)),
                    seed=int(arguments.get("seed", -1)),
                    conversation_history=conversation_history,
                    defer_llm_restore=defer_llm_restore,
                )

            return {"status": "error", "error": f"Unknown tool '{tool_name}'"}
        except KeyError as exc:
            return {
                "status": "error",
                "tool": tool_name,
                "error": f"Missing required argument {exc} for tool '{tool_name}'. Call it again with valid JSON arguments.",
            }
        except Exception as exc:
            logger.exception(f"Tool execution failed ({tool_name}): {exc}")
            return {"status": "error", "tool": tool_name, "error": str(exc)}

    # -------------------------------------------------------------------------
    # Tool 1: list_dir
    # -------------------------------------------------------------------------
    def tool_recall_memory(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Execute Zero-VRAM CPU RAG recall over ./cache/project_memory.db + .harness/MEMORY.md.
        Returns snippets designed to be injected strictly at the bottom of the active conversation.
        """
        if self.memory_rag is None:
            return {
                "status": "error",
                "tool": "recall_memory",
                "error": "ProjectMemoryRAG is not configured on AgentToolExecutor.",
            }
        return self.memory_rag.recall_memory(query=query, top_k=top_k)

    def tool_list_dir(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        target = self.resolve_workspace_path(path)
        if not target.exists():
            return {"status": "error", "error": f"Directory not found: {path}"}
        if not target.is_dir():
            return {"status": "error", "error": f"Not a directory: {path}"}

        entries: List[Dict[str, Any]] = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        for item in sorted(iterator, key=lambda p: (not p.is_dir(), p.name.lower())):
            if any(part.startswith(".harness_backups") or part == "__pycache__" for part in item.parts):
                continue
            rel = self.to_res_path(item)
            entries.append(
                {
                    "name": item.name,
                    "res_path": rel,
                    "type": "dir" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                }
            )
            if len(entries) >= 200:
                break

        return {
            "status": "success",
            "directory": self.to_res_path(target),
            "count": len(entries),
            "entries": entries,
        }

    # -------------------------------------------------------------------------
    # Tool 2: read_file (Line-Ranged + Godot Scene Graph Summary)
    # -------------------------------------------------------------------------
    def tool_read_file(
        self, path: str, start_line: int = 1, end_line: int = 250
    ) -> Dict[str, Any]:
        target = self.resolve_workspace_path(path)
        if not target.exists() or not target.is_file():
            return {"status": "error", "error": f"File not found: {path}"}

        content = target.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        total_lines = len(all_lines)

        start_line = max(1, start_line)
        end_line = min(total_lines, max(start_line, end_line))
        max_lines = max(1, end_line - start_line + 1)

        numbered = GodotParser.format_numbered_lines(
            content, start_line=start_line, max_lines=max_lines
        )

        godot_summary = None
        if target.suffix.lower() in (".tscn", ".tres", ".gd", ".gdshader", ".godot") or target.name == "project.godot":
            parsed = GodotParser.parse_content(
                content, filename=target.name, file_path=str(target), include_raw_lines=False
            )
            godot_summary = parsed["summary_text"]

        return {
            "status": "success",
            "path": str(target),
            "res_path": self.to_res_path(target),
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "godot_scene_summary": godot_summary,
            "numbered_content": numbered,
        }

    # -------------------------------------------------------------------------
    # Tool 3: grep_search
    # -------------------------------------------------------------------------
    def tool_grep_search(
        self,
        query: str,
        path: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        target = self.resolve_workspace_path(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags)
        except re.error:
            regex = re.compile(re.escape(query), flags)

        matches: List[Dict[str, Any]] = []
        files = [target] if target.is_file() else sorted(target.rglob(file_pattern))

        for fpath in files:
            if not fpath.is_file():
                continue
            if any(
                part.startswith(".") or part in ("__pycache__", "node_modules", "blend_src")
                for part in fpath.parts
            ):
                continue
            if fpath.suffix.lower() in (".png", ".jpg", ".webp", ".glb", ".blend", ".gguf", ".exe", ".bin"):
                continue

            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "file": self.to_res_path(fpath),
                            "line": line_no,
                            "content": line.strip()[:240],
                        }
                    )
                    if len(matches) >= 100:
                        return {
                            "status": "success",
                            "query": query,
                            "match_count": len(matches),
                            "truncated": True,
                            "matches": matches,
                        }

        return {
            "status": "success",
            "query": query,
            "match_count": len(matches),
            "truncated": False,
            "matches": matches,
        }

    # -------------------------------------------------------------------------
    # Tool 4: apply_diff (with .bak Rollback Snapshot + GDScript Validation)
    # -------------------------------------------------------------------------
    async def tool_apply_diff(
        self,
        path: str,
        search_text: Optional[str] = None,
        replace_text: Optional[str] = None,
        unified_diff: Optional[str] = None,
    ) -> Dict[str, Any]:
        target = self.resolve_workspace_path(path)
        if not target.exists():
            return {"status": "error", "error": f"Target file does not exist: {path}"}

        original_content = target.read_text(encoding="utf-8", errors="replace")
        updated_content = original_content

        # Case 1: Explicit search_text & replace_text
        if search_text is not None and replace_text is not None:
            search_text, replace_text = self._strip_line_number_prefixes(search_text, replace_text)
            exact_hits = updated_content.count(search_text) if search_text else 0
            if exact_hits > 1:
                return {
                    "status": "error",
                    "error": (
                        f"search_text matches {exact_hits} places in {self.to_res_path(target)}. "
                        "Include more surrounding lines so it is unique. Nothing was changed."
                    ),
                }
            if exact_hits == 1:
                updated_content = updated_content.replace(search_text, replace_text, 1)
            else:
                norm_orig = updated_content.replace("\r\n", "\n")
                norm_search = search_text.replace("\r\n", "\n")
                norm_replace = replace_text.replace("\r\n", "\n")
                if norm_search and norm_search in norm_orig:
                    updated_content = norm_orig.replace(norm_search, norm_replace, 1)
                else:
                    # Whitespace-tolerant fallback: small models often get tabs/indentation wrong
                    fuzzy, problem = self._whitespace_tolerant_replace(norm_orig, norm_search, norm_replace)
                    if fuzzy is None:
                        return {
                            "status": "error",
                            "error": (
                                f"search_text was not applied to {self.to_res_path(target)}: {problem} "
                                "Nothing was changed."
                            ),
                        }
                    updated_content = fuzzy

        # Case 2: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE or Unified Diff
        elif unified_diff:
            sr_matches = list(self.SEARCH_REPLACE_BLOCK_RE.finditer(unified_diff))
            if sr_matches:
                norm_orig = updated_content.replace("\r\n", "\n")
                for m in sr_matches:
                    s_block = m.group(1).replace("\r\n", "\n")
                    r_block = m.group(2).replace("\r\n", "\n")
                    if s_block not in norm_orig:
                        return {
                            "status": "error",
                            "error": f"SEARCH block not found in {self.to_res_path(target)}.",
                        }
                    norm_orig = norm_orig.replace(s_block, r_block, 1)
                updated_content = norm_orig
            else:
                patched = self._apply_unified_diff_hunks(original_content, unified_diff)
                if patched is None:
                    return {
                        "status": "error",
                        "error": "Could not apply unified_diff hunks cleanly.",
                    }
                updated_content = patched
        else:
            return {
                "status": "error",
                "error": "Provide either (search_text + replace_text) or unified_diff.",
            }

        # Save .bak snapshot before writing
        bak_path = self._create_backup_snapshot(target, original_content)
        target.write_text(updated_content, encoding="utf-8")

        diff_record = self._record_diff_entry(
            target=target,
            old_text=original_content,
            new_text=updated_content,
            bak_path=bak_path,
            operation="apply_diff",
        )

        # Automatic Godot --check-only validation if target is a .gd script
        gd_validation = None
        if target.suffix.lower() == ".gd":
            gd_validation = await self.validate_gdscript(target)

        return {
            "status": "success",
            "path": str(target),
            "res_path": self.to_res_path(target),
            "backup_path": str(bak_path),
            "diff_id": diff_record["diff_id"],
            "unified_diff": diff_record["unified_diff"],
            "lines_added": diff_record["lines_added"],
            "lines_removed": diff_record["lines_removed"],
            "gdscript_validation": gd_validation,
        }

    _LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s\|\s?")

    @classmethod
    def _strip_line_number_prefixes(cls, search_text: str, replace_text: str) -> Tuple[str, str]:
        """If the model pasted read_file's '  12 | ' prefixes into its edit, remove them."""
        s_lines = [ln for ln in search_text.split("\n") if ln.strip()]
        if s_lines and all(cls._LINE_NUMBER_PREFIX_RE.match(ln) for ln in s_lines):
            def _clean(text: str) -> str:
                return "\n".join(cls._LINE_NUMBER_PREFIX_RE.sub("", ln, count=1) for ln in text.split("\n"))
            return _clean(search_text), _clean(replace_text)
        return search_text, replace_text

    @staticmethod
    def _whitespace_tolerant_replace(
        original: str, search: str, replace: str
    ) -> Tuple[Optional[str], str]:
        """
        Match `search` against `original` line-by-line ignoring leading/trailing whitespace.
        On exactly one match, substitute `replace`, shifting its indentation to the file's real indentation.
        Returns (new_text, "") or (None, explanation).
        """
        o_lines = original.split("\n")
        s_lines = search.strip("\n").split("\n")
        key = [ln.strip() for ln in s_lines]
        if not any(key):
            return None, "search_text is empty."
        n = len(key)
        hits = [
            i for i in range(len(o_lines) - n + 1)
            if [ln.strip() for ln in o_lines[i : i + n]] == key
        ]
        if len(hits) > 1:
            return None, (
                f"the block matches {len(hits)} places (lines {', '.join(str(h + 1) for h in hits[:5])}). "
                "Include more surrounding lines so it is unique."
            )
        if not hits:
            first = next(k for k in key if k)
            near = [i for i, ln in enumerate(o_lines) if ln.strip() == first]
            if near:
                i = near[0]
                actual = "\n".join(o_lines[i : i + n])
                return None, (
                    f"the first line was found at line {i + 1} but the following lines differ. "
                    f"The file actually contains:\n{actual}\nCopy these lines exactly."
                )
            return None, "none of these lines exist in the file. Call read_file to see the current content."

        i = hits[0]

        def _indent(ln: str) -> str:
            return ln[: len(ln) - len(ln.lstrip())]

        k = next(j for j, ln in enumerate(key) if ln)
        actual_indent = _indent(o_lines[i + k])
        search_indent = _indent(s_lines[k])
        new_lines: List[str] = []
        if replace.strip():
            for ln in replace.strip("\n").split("\n"):
                if not ln.strip():
                    new_lines.append("")
                elif search_indent and ln.startswith(search_indent):
                    new_lines.append(actual_indent + ln[len(search_indent):])
                elif not search_indent:
                    new_lines.append(actual_indent + ln)
                else:
                    new_lines.append(ln)
        return "\n".join(o_lines[:i] + new_lines + o_lines[i + n :]), ""

    # -------------------------------------------------------------------------
    # Tool: set_input_action (writes correctly formatted bindings into project.godot)
    # -------------------------------------------------------------------------
    _KEY_SPECIAL = 4194304
    _KEYCODES: Dict[str, int] = {
        "space": 32, "escape": _KEY_SPECIAL + 1, "esc": _KEY_SPECIAL + 1, "tab": _KEY_SPECIAL + 2,
        "backspace": _KEY_SPECIAL + 4, "enter": _KEY_SPECIAL + 5, "return": _KEY_SPECIAL + 5,
        "left": _KEY_SPECIAL + 15, "up": _KEY_SPECIAL + 16, "right": _KEY_SPECIAL + 17, "down": _KEY_SPECIAL + 18,
        "shift": _KEY_SPECIAL + 21, "ctrl": _KEY_SPECIAL + 22, "control": _KEY_SPECIAL + 22, "alt": _KEY_SPECIAL + 24,
        **{f"f{i}": 4194304 + 27 + i for i in range(1, 13)},
    }
    _MOUSE_BUTTONS: Dict[str, int] = {"left": 1, "right": 2, "middle": 3}

    @classmethod
    def _resolve_keycode(cls, key_name: str) -> Optional[int]:
        k = str(key_name).strip().lower().replace("key_", "").replace("arrow", "").replace("_", "").strip()
        if k in cls._KEYCODES:
            return cls._KEYCODES[k]
        if len(k) == 1 and (k.isalpha() or k.isdigit()):
            return ord(k.upper())
        return None

    def tool_set_input_action(
        self, action: str, keys: List[str], mouse_buttons: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        action = action.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", action or ""):
            return {"status": "error", "error": f"Invalid action name '{action}'. Use letters, digits and underscores."}
        project_file = self.workspace_dir / "project.godot"
        if not project_file.exists():
            return {"status": "error", "error": "project.godot not found in the project folder."}

        events: List[str] = []
        unknown: List[str] = []
        bound: List[str] = []
        for key_name in keys or []:
            code = self._resolve_keycode(key_name)
            if code is None:
                unknown.append(str(key_name))
                continue
            events.append(
                'Object(InputEventKey,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,'
                '"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"pressed":false,'
                f'"keycode":0,"physical_keycode":{code},"key_label":0,"unicode":0,"location":0,"echo":false,"script":null)'
            )
            bound.append(str(key_name))
        for btn in mouse_buttons or []:
            idx = self._MOUSE_BUTTONS.get(str(btn).strip().lower())
            if idx is None:
                unknown.append(f"mouse:{btn}")
                continue
            events.append(
                'Object(InputEventMouseButton,"resource_local_to_scene":false,"resource_name":"","device":-1,"window_id":0,'
                '"alt_pressed":false,"shift_pressed":false,"ctrl_pressed":false,"meta_pressed":false,"button_mask":0,'
                '"position":Vector2(0, 0),"global_position":Vector2(0, 0),"factor":1.0,'
                f'"button_index":{idx},"canceled":false,"pressed":false,"double_click":false,"script":null)'
            )
            bound.append(f"mouse:{btn}")
        if not events:
            return {
                "status": "error",
                "error": f"No valid keys or mouse buttons given (unrecognised: {unknown}). Nothing was changed.",
            }

        block = f'{action}={{\n"deadzone": 0.5,\n"events": [' + "\n, ".join(events) + "\n]\n}\n"
        original = project_file.read_text(encoding="utf-8", errors="replace")
        text = original
        if not text.endswith("\n"):
            text += "\n"

        section = re.search(r"^\[input\][ \t]*\n", text, flags=re.MULTILINE)
        if section is None:
            text = text.rstrip("\n") + "\n\n[input]\n\n" + block
        else:
            body_start = section.end()
            next_section = re.search(r"^\[[^\]\n]+\][ \t]*$", text[body_start:], flags=re.MULTILINE)
            body_end = body_start + next_section.start() if next_section else len(text)
            body = text[body_start:body_end]
            # An action block runs from `name={` to the first line that is just `}`
            existing = re.search(
                rf"^{re.escape(action)}\s*=\s*\{{.*?^\}}[ \t]*\n?", body, flags=re.MULTILINE | re.DOTALL
            )
            if existing:
                body = body[: existing.start()] + block + body[existing.end():]
            else:
                body = body.rstrip("\n") + "\n" + block + ("\n" if next_section else "")
                if not body.startswith("\n"):
                    body = "\n" + body
            text = text[:body_start] + body + text[body_end:]

        bak_path = self._create_backup_snapshot(project_file, original)
        project_file.write_text(text, encoding="utf-8")
        diff_record = self._record_diff_entry(
            target=project_file, old_text=original, new_text=text, bak_path=bak_path, operation="set_input_action"
        )
        result: Dict[str, Any] = {
            "status": "success",
            "action": action,
            "bound": bound,
            "res_path": "res://project.godot",
            "diff_id": diff_record["diff_id"],
            "note": f'Use it in scripts as Input.is_action_pressed("{action}") / is_action_just_pressed("{action}").',
        }
        if unknown:
            result["ignored_unrecognised"] = unknown
        return result

    # -------------------------------------------------------------------------
    # Tool 5: write_file (with .bak Rollback Snapshot + GDScript Validation)
    # -------------------------------------------------------------------------
    async def tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        target = self.resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        old_content = ""
        bak_path = None
        if target.exists():
            old_content = target.read_text(encoding="utf-8", errors="replace")
            bak_path = self._create_backup_snapshot(target, old_content)

        target.write_text(content, encoding="utf-8")

        diff_record = self._record_diff_entry(
            target=target,
            old_text=old_content,
            new_text=content,
            bak_path=bak_path,
            operation="write_file",
        )

        gd_validation = None
        scene_validation = None
        if target.suffix.lower() == ".gd":
            gd_validation = await self.validate_gdscript(target)
        elif target.suffix.lower() in (".tscn", ".tres"):
            scene_validation = self.validate_scene_or_resource_file(target)

        return {
            "status": "success",
            "path": str(target),
            "res_path": self.to_res_path(target),
            "backup_path": str(bak_path) if bak_path else None,
            "diff_id": diff_record["diff_id"],
            "unified_diff": diff_record["unified_diff"],
            "lines_added": diff_record["lines_added"],
            "lines_removed": diff_record["lines_removed"],
            "gdscript_validation": gd_validation,
            "scene_validation": scene_validation,
        }

    # -------------------------------------------------------------------------
    # Tool 6: run_command (with Approval Toggle + Godot --headless Validation)
    # -------------------------------------------------------------------------
    async def tool_run_command(
        self,
        command: str,
        check_gdscript_path: Optional[str] = None,
        timeout_sec: int = 30,
        approved: bool = False,
    ) -> Dict[str, Any]:
        if check_gdscript_path:
            script_target = self.resolve_workspace_path(check_gdscript_path)
            return await self.validate_gdscript(script_target)

        if self.require_command_approval and not approved:
            cmd_id = f"cmd_{uuid.uuid4().hex[:8]}"
            self._pending_commands[cmd_id] = {
                "cmd_id": cmd_id,
                "command": command,
                "timeout_sec": timeout_sec,
                "created_at": time.strftime("%H:%M:%S"),
            }
            return {
                "status": "approval_required",
                "cmd_id": cmd_id,
                "command": command,
                "message": (
                    f"Command execution requires user approval in the UI: `{command}` "
                    f"(approval_id={cmd_id})."
                ),
            }

        # Intercept `godot --headless ...` (`--quit`, `--check-only`, `-s ...`) commands
        # so they automatically resolve the local Godot 4 executable AND run full project-wide
        # scene/resource/GDScript verification even when `godot` is not on the system PATH.
        cmd_stripped = command.strip()
        cmd_lower = cmd_stripped.lower()
        if re.match(r"^(?:godot4?|\"[^\"]*godot[^\"]*\")\b", cmd_lower) and "--headless" in cmd_lower:
            return await self.validate_godot_project(raw_command=cmd_stripped, timeout_sec=timeout_sec)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.workspace_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout_sec)
            )
            stdout_str = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
            stderr_str = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
            return {
                "status": "success" if proc.returncode == 0 else "non_zero_exit",
                "command": command,
                "exit_code": proc.returncode,
                "stdout": stdout_str[:4000],
                "stderr": stderr_str[:4000],
            }
        except RuntimeError:
            # Fallback when driven outside a running ProactorEventLoop (e.g. synchronous test runner)
            proc_sync = subprocess.run(
                command,
                cwd=str(self.workspace_dir),
                shell=True,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
            )
            return {
                "status": "success" if proc_sync.returncode == 0 else "non_zero_exit",
                "command": command,
                "exit_code": proc_sync.returncode,
                "stdout": (proc_sync.stdout or "")[:4000],
                "stderr": (proc_sync.stderr or "")[:4000],
            }
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "command": command,
                "error": f"Command timed out after {timeout_sec}s.",
            }
        except Exception as exc:
            return {
                "status": "error",
                "command": command,
                "error": str(exc),
            }

    async def approve_and_run_pending_command(self, cmd_id: str) -> Dict[str, Any]:
        pending = self._pending_commands.pop(cmd_id, None)
        if not pending:
            return {"status": "error", "error": f"No pending command found for id '{cmd_id}'."}
        return await self.tool_run_command(
            command=pending["command"],
            timeout_sec=pending.get("timeout_sec", 30),
            approved=True,
        )

    # -------------------------------------------------------------------------
    # Godot 4 Headless --check-only GDScript Validator
    # -------------------------------------------------------------------------
    async def validate_gdscript(self, script_path: Path) -> Dict[str, Any]:
        """
        Validate GDScript 2.0 syntax & types:
          1. If Godot 4 binary is detected on host, runs:
             `godot --headless --path <workspace> --check-only -s <res_path>`
          2. Also runs a fast built-in GDScript 2.0 static linter (checking Godot 3 vs Godot 4
             pitfalls like `KinematicBody`, `yield()`, `export var`, `onready var`, `.connect()` syntax,
             and bracket/indentation balance).
        """
        res_script = self.to_res_path(script_path)
        content = script_path.read_text(encoding="utf-8", errors="replace")
        static_diagnostics = self._static_lint_gdscript4(content, script_path.name)

        godot_exe = self.detect_godot_executable()
        if godot_exe and (self.workspace_dir / "project.godot").exists():
            cmd = [
                godot_exe,
                "--headless",
                "--path",
                str(self.workspace_dir),
                "--check-only",
                "-s",
                res_script,
            ]
            try:
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.workspace_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=creationflags,
                )
                out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                output = out_b.decode("utf-8", errors="replace") if out_b else ""
                valid = proc.returncode == 0 and "SCRIPT ERROR:" not in output
                return {
                    "validator": "godot --headless --check-only",
                    "valid": valid,
                    "exit_code": proc.returncode,
                    "output": output.strip()[:1200],
                    "static_diagnostics": static_diagnostics,
                }
            except Exception as exc:
                logger.debug(f"Godot CLI check fallback: {exc}")

        return {
            "validator": "built-in-gdscript4-static-analyzer",
            "valid": len([d for d in static_diagnostics if d["severity"] == "error"]) == 0,
            "static_diagnostics": static_diagnostics,
        }

    def validate_scene_or_resource_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a Godot 4 `.tscn` or `.tres` file:
          - Verify all `[ext_resource ... path="res://..." id="..."]` target files exist on disk.
          - Verify all `ExtResource("id")` and `SubResource("id")` references resolve to declared IDs.
        """
        res_file = self.to_res_path(file_path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        parsed = GodotParser.parse_content(
            content, filename=file_path.name, file_path=str(file_path), include_raw_lines=False
        )
        errors: List[Dict[str, Any]] = []

        declared_ext_ids = set(parsed.get("ext_resources", {}).keys())
        declared_sub_ids = set(parsed.get("sub_resources", {}).keys())

        for ext_id, ext_info in parsed.get("ext_resources", {}).items():
            ext_res_path = ext_info.get("path", "")
            if ext_res_path.startswith("res://"):
                try:
                    resolved = self.resolve_workspace_path(ext_res_path)
                    if not resolved.exists():
                        errors.append(
                            {
                                "file": res_file,
                                "severity": "error",
                                "message": f"Missing ExtResource file on disk: '{ext_res_path}' (id='{ext_id}')",
                            }
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "file": res_file,
                            "severity": "error",
                            "message": f"Invalid ExtResource path '{ext_res_path}': {exc}",
                        }
                    )

        for ref_id in re.findall(r'ExtResource\(\s*"([^"]+)"\s*\)', content):
            if ref_id not in declared_ext_ids:
                errors.append(
                    {
                        "file": res_file,
                        "severity": "error",
                        "message": f"Undeclared ExtResource(\"{ref_id}\") referenced in {res_file}",
                    }
                )

        for ref_id in re.findall(r'SubResource\(\s*"([^"]+)"\s*\)', content):
            if ref_id not in declared_sub_ids:
                errors.append(
                    {
                        "file": res_file,
                        "severity": "error",
                        "message": f"Undeclared SubResource(\"{ref_id}\") referenced in {res_file}",
                    }
                )

        return {
            "file": res_file,
            "valid": len(errors) == 0,
            "errors": errors,
            "scene_summary": parsed.get("summary_text"),
        }

    async def validate_godot_project(
        self, raw_command: str = "godot --headless --quit", timeout_sec: int = 30
    ) -> Dict[str, Any]:
        """
        Comprehensive project-wide Godot 4 validation for `godot --headless --quit` / `--check-only`:
          1. Validates `project.godot` and confirms `run/main_scene` exists if specified.
          2. Validates all `.tscn` and `.tres` files in the workspace (confirming every `ExtResource`
             `res://` path exists on disk and all `ExtResource`/`SubResource` IDs resolve).
          3. Validates all `.gd` scripts in the workspace (GDScript 2.0 linter + verifying all
             `preload("res://...")` / `load("res://...")` paths exist on disk).
          4. If Godot 4 binary is installed on the host, also executes the live headless Godot CLI check.
        """
        diagnostics: List[Dict[str, Any]] = []
        checked_scripts: List[str] = []
        checked_scenes: List[str] = []

        proj_file = self.workspace_dir / "project.godot"
        if not proj_file.exists():
            diagnostics.append(
                {
                    "file": "res://project.godot",
                    "severity": "error",
                    "message": "project.godot not found in workspace root.",
                }
            )
        else:
            proj_text = proj_file.read_text(encoding="utf-8", errors="replace")
            m_scene = re.search(r'run/main_scene\s*=\s*"([^"]+)"', proj_text)
            if m_scene:
                main_scene_res = m_scene.group(1)
                try:
                    if not self.resolve_workspace_path(main_scene_res).exists():
                        diagnostics.append(
                            {
                                "file": "res://project.godot",
                                "severity": "error",
                                "message": f"run/main_scene '{main_scene_res}' does not exist on disk.",
                            }
                        )
                except Exception as exc:
                    diagnostics.append(
                        {
                            "file": "res://project.godot",
                            "severity": "error",
                            "message": str(exc),
                        }
                    )

        for fpath in sorted(self.workspace_dir.rglob("*")):
            if not fpath.is_file():
                continue
            if any(part.startswith(".") or part == "blend_src" for part in fpath.parts):
                continue
            suffix = fpath.suffix.lower()
            if suffix in (".tscn", ".tres"):
                checked_scenes.append(self.to_res_path(fpath))
                s_val = self.validate_scene_or_resource_file(fpath)
                diagnostics.extend(s_val["errors"])
            elif suffix == ".gd":
                res_gd = self.to_res_path(fpath)
                checked_scripts.append(res_gd)
                gd_text = fpath.read_text(encoding="utf-8", errors="replace")
                diagnostics.extend(self._static_lint_gdscript4(gd_text, res_gd))
                for line_no, line in enumerate(gd_text.splitlines(), start=1):
                    code_only = line.split("#", 1)[0]
                    for res_ref in re.findall(r'(?:preload|load)\(\s*"(res://[^"]+)"\s*\)', code_only):
                        try:
                            if not self.resolve_workspace_path(res_ref).exists():
                                diagnostics.append(
                                    {
                                        "file": res_gd,
                                        "line": line_no,
                                        "severity": "error",
                                        "message": f"Referenced resource '{res_ref}' does not exist on disk.",
                                    }
                                )
                        except Exception as exc:
                            diagnostics.append(
                                {
                                    "file": res_gd,
                                    "line": line_no,
                                    "severity": "error",
                                    "message": str(exc),
                                }
                            )

        godot_exe = self.detect_godot_executable()
        cli_output = ""
        cli_exit = 0
        if godot_exe and proj_file.exists():
            try:
                proc_sync = subprocess.run(
                    # Run the main scene headless for ~90 frames so runtime errors in _ready/_process surface too
                    [godot_exe, "--headless", "--path", str(self.workspace_dir), "--quit-after", "90"],
                    cwd=str(self.workspace_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=float(max(timeout_sec, 45)),
                )
                raw_lines = ((proc_sync.stdout or "") + "\n" + (proc_sync.stderr or "")).splitlines()
                # Engine shutdown noise that is not a project problem
                benign = ("resources still in use at exit", "ObjectDB instances leaked", "at: cleanup (", "at: clear (")
                kept_lines = [ln for ln in raw_lines if ln.strip() and not any(b in ln for b in benign)]
                cli_output = "\n".join(kept_lines).strip()
                seen_msgs: set = set()
                for li, err_line in enumerate(kept_lines):
                    if "ERROR:" not in err_line:
                        continue
                    msg = err_line.strip()
                    nxt = kept_lines[li + 1].strip() if li + 1 < len(kept_lines) else ""
                    if nxt.startswith("at:"):
                        msg += " " + nxt
                    if msg in seen_msgs:
                        continue
                    seen_msgs.add(msg)
                    if len(seen_msgs) <= 15:
                        diagnostics.append({"file": "godot-cli", "severity": "error", "message": msg})
                cli_exit = proc_sync.returncode if seen_msgs else 0
            except subprocess.TimeoutExpired:
                cli_output = "Godot headless run did not exit in time (possible infinite loop in a script)."
            except Exception as exc:
                logger.debug(f"Godot headless project check CLI fallback: {exc}")

        valid = len(diagnostics) == 0 and cli_exit == 0
        summary_lines = [
            f"Godot 4 Headless Project Validation ({'PASSED - 0 errors' if valid else f'FAILED - {len(diagnostics)} error(s)'})",
            f"Checked {len(checked_scenes)} scene/resource files and {len(checked_scripts)} GDScript files.",
        ]
        for d in diagnostics:
            loc = f"{d.get('file')}:{d['line']}" if d.get("line") else d.get("file")
            summary_lines.append(f"  ERROR [{loc}]: {d.get('message')}")
        if cli_output and not valid:
            summary_lines.append("--- Godot output ---")
            summary_lines.append(cli_output[:1500])

        return {
            "status": "success" if valid else "non_zero_exit",
            "command": raw_command,
            "exit_code": 0 if valid else 1,
            "valid": valid,
            "checked_scenes": checked_scenes,
            "checked_scripts": checked_scripts,
            "diagnostics": diagnostics,
            "stdout": "\n".join(summary_lines),
            "stderr": "",
        }

    @staticmethod
    def _static_lint_gdscript4(content: str, filename: str) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        deprecated_g3_patterns = [
            (r"(?<!@)\bexport\s+var\b", "Use `@export var` in Godot 4 instead of Godot 3 `export var`."),
            (r"(?<!@)\bonready\s+var\b", "Use `@onready var` in Godot 4 instead of Godot 3 `onready var`."),
            (r"\bKinematicBody(2D|3D)?\b", "Use `CharacterBody2D` / `CharacterBody3D` in Godot 4."),
            (r"\bSpatial\b", "Use `Node3D` in Godot 4 instead of Godot 3 `Spatial`."),
            (r"\byield\s*\(", "Use `await` in Godot 4 instead of Godot 3 `yield()`."),
            (r"\.instance\s*\(\s*\)", "Use `.instantiate()` in Godot 4 instead of `.instance()`."),
        ]
        for idx, line in enumerate(content.splitlines(), start=1):
            code_part = line.split("#", 1)[0]
            for pat, msg in deprecated_g3_patterns:
                if re.search(pat, code_part):
                    diagnostics.append(
                        {
                            "file": filename,
                            "line": idx,
                            "severity": "error",
                            "message": msg,
                        }
                    )
        return diagnostics

    # -------------------------------------------------------------------------
    # Diff Generation & .bak Rollback Snapshots
    # -------------------------------------------------------------------------
    def _create_backup_snapshot(self, target: Path, original_content: str) -> Path:
        ts = int(time.time() * 1000)
        bak_name = f"{target.name}.{ts}.bak"
        bak_path = self.backup_dir / bak_name
        bak_path.write_text(original_content, encoding="utf-8")
        # Also write immediate sibling <filename>.bak for convenience
        sibling_bak = target.with_name(target.name + ".bak")
        sibling_bak.write_text(original_content, encoding="utf-8")
        return bak_path

    def _record_diff_entry(
        self,
        target: Path,
        old_text: str,
        new_text: str,
        bak_path: Optional[Path],
        operation: str,
    ) -> Dict[str, Any]:
        res_path = self.to_res_path(target)
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{res_path}",
                tofile=f"b/{res_path}",
                lineterm="",
            )
        )
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        diff_id = f"diff_{uuid.uuid4().hex[:8]}"
        entry = {
            "diff_id": diff_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "operation": operation,
            "path": str(target),
            "res_path": res_path,
            "backup_path": str(bak_path) if bak_path else None,
            "unified_diff": "\n".join(diff_lines),
            "lines_added": added,
            "lines_removed": removed,
            "rolled_back": False,
        }
        self._diff_history.append(entry)
        return entry

    def rollback_diff(self, diff_id: str) -> Dict[str, Any]:
        """Restore a file from its .bak snapshot recorded in _diff_history."""
        target_entry = None
        for entry in reversed(self._diff_history):
            if entry["diff_id"] == diff_id:
                target_entry = entry
                break

        if not target_entry:
            return {"status": "error", "error": f"Diff ID '{diff_id}' not found."}

        target_path = Path(target_entry["path"])
        bak_path = Path(target_entry["backup_path"]) if target_entry.get("backup_path") else None

        if bak_path and bak_path.exists():
            restored_text = bak_path.read_text(encoding="utf-8", errors="replace")
            target_path.write_text(restored_text, encoding="utf-8")
            target_entry["rolled_back"] = True
            return {
                "status": "success",
                "diff_id": diff_id,
                "res_path": target_entry["res_path"],
                "message": f"Restored {target_entry['res_path']} from backup {bak_path.name}.",
            }
        elif target_entry["operation"] == "write_file" and not bak_path:
            if target_path.exists():
                target_path.unlink()
            target_entry["rolled_back"] = True
            return {
                "status": "success",
                "diff_id": diff_id,
                "res_path": target_entry["res_path"],
                "message": f"Removed newly created file {target_entry['res_path']}.",
            }

        return {"status": "error", "error": "Backup .bak file no longer exists on disk."}

    def get_diff_history(self) -> List[Dict[str, Any]]:
        return list(reversed(self._diff_history[-30:]))

    @staticmethod
    def _apply_unified_diff_hunks(original_text: str, patch_text: str) -> Optional[str]:
        """Apply standard unified diff hunks (@@ -start,count +start,count @@) to original_text."""
        orig_lines = original_text.replace("\r\n", "\n").splitlines()
        patch_lines = patch_text.replace("\r\n", "\n").splitlines()

        hunk_re = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@")
        result_lines: List[str] = []
        orig_idx = 0
        i = 0

        found_hunk = False
        while i < len(patch_lines):
            line = patch_lines[i]
            m = hunk_re.match(line)
            if not m:
                i += 1
                continue
            found_hunk = True
            start_old = max(1, int(m.group(1))) - 1
            while orig_idx < start_old and orig_idx < len(orig_lines):
                result_lines.append(orig_lines[orig_idx])
                orig_idx += 1

            i += 1
            while i < len(patch_lines) and not hunk_re.match(patch_lines[i]):
                hline = patch_lines[i]
                if hline.startswith(("---", "+++")):
                    i += 1
                    continue
                if hline.startswith("+"):
                    result_lines.append(hline[1:])
                elif hline.startswith("-"):
                    orig_idx += 1
                elif hline.startswith(" ") or hline == "":
                    if orig_idx < len(orig_lines):
                        result_lines.append(orig_lines[orig_idx])
                        orig_idx += 1
                i += 1

        if not found_hunk:
            return None

        while orig_idx < len(orig_lines):
            result_lines.append(orig_lines[orig_idx])
            orig_idx += 1

        return "\n".join(result_lines) + ("\n" if original_text.endswith("\n") else "")
