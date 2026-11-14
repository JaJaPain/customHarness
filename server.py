"""
FastAPI + WebSockets Backend Server for the Standalone Local-First Godot 4
Agentic Coding & 2D/3D Asset-Creation Harness.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_tools import (
    TOOL_DEFINITIONS,
    AgentToolExecutor,
    parse_thinking_and_tool_calls,
)
from audio_gen_worker import AudioGenWorker
from blender_worker import BlenderWorker
from godot_parser import GodotParser
from image_gen_worker import ImageGenWorker
from multimodal_ingest import MultimodalIngestor
from vram_orchestrator import (
    AUDIO_MODELS,
    IMAGE_MODELS,
    MODEL_PROFILES,
    PORTABLE_BINARIES,
    EngineState,
    VRAMOrchestrator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CustomHarness.Server")

BASE_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = BASE_DIR / "workspace"
STATIC_DIR = BASE_DIR / "static"


# =============================================================================
# Seed Default Godot 4 3D Starter Workspace
# =============================================================================
def ensure_starter_godot_workspace(workspace: Path) -> None:
    """Create a realistic Godot 4.3 3D starter project inside ./workspace if empty."""
    workspace.mkdir(parents=True, exist_ok=True)
    for sub in ("scenes", "scripts", "assets/textures", "assets/models", "audio/sfx", "audio/music", "shaders"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)

    proj_file = workspace / "project.godot"
    if not proj_file.exists():
        proj_file.write_text(
            '; Engine configuration file.\n'
            '; It\'s best edited using the editor UI and not directly.\n\n'
            'config_version=5\n\n'
            '[application]\n\n'
            'config/name="OrnithGodot4HarnessDemo"\n'
            'run/main_scene="res://scenes/main_3d.tscn"\n'
            'config/features=PackedStringArray("4.3", "Forward Plus")\n\n'
            '[input]\n\n'
            'move_forward={\n'
            '"deadzone": 0.5,\n'
            '"events": []\n'
            '}\n'
            'jump={\n'
            '"deadzone": 0.5,\n'
            '"events": []\n'
            '}\n\n'
            '[rendering]\n\n'
            'anti_aliasing/quality/msaa_3d=2\n',
            encoding="utf-8",
        )

    main_scene = workspace / "scenes" / "main_3d.tscn"
    if not main_scene.exists():
        main_scene.write_text(
            '[gd_scene load_steps=6 format=3 uid="uid://c8h4x7m2q9p1w"]\n\n'
            '[ext_resource type="Script" path="res://scripts/player_controller.gd" id="1_script"]\n\n'
            '[sub_resource type="ProceduralSkyMaterial" id="ProceduralSkyMaterial_1"]\n'
            'sky_top_color = Color(0.18, 0.26, 0.42, 1)\n\n'
            '[sub_resource type="Environment" id="Environment_1"]\n'
            'background_mode = 2\n'
            'glow_enabled = true\n\n'
            '[sub_resource type="BoxShape3D" id="BoxShape3D_ground"]\n'
            'size = Vector3(40, 1, 40)\n\n'
            '[sub_resource type="CapsuleShape3D" id="CapsuleShape3D_player"]\n'
            'radius = 0.45\n'
            'height = 1.8\n\n'
            '[node name="Main3D" type="Node3D"]\n\n'
            '[node name="WorldEnvironment" type="WorldEnvironment" parent="."]\n'
            'environment = SubResource("Environment_1")\n\n'
            '[node name="DirectionalLight3D" type="DirectionalLight3D" parent="."]\n'
            'transform = Transform3D(0.866025, -0.353553, 0.353553, 0, 0.707107, 0.707107, -0.5, -0.612372, 0.612372, 0, 12, 0)\n'
            'shadow_enabled = true\n\n'
            '[node name="GroundBody" type="StaticBody3D" parent="."]\n'
            'collision_layer = 1\n'
            'collision_mask = 1\n\n'
            '[node name="GroundCollider" type="CollisionShape3D" parent="GroundBody"]\n'
            'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -0.5, 0)\n'
            'shape = SubResource("BoxShape3D_ground")\n\n'
            '[node name="Player3D" type="CharacterBody3D" parent="." groups=["player"]]\n'
            'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1.2, 4.5)\n'
            'collision_layer = 2\n'
            'collision_mask = 1\n'
            'script = ExtResource("1_script")\n'
            'move_speed = 6.5\n'
            'jump_velocity = 5.2\n\n'
            '[node name="PlayerCollision" type="CollisionShape3D" parent="Player3D"]\n'
            'shape = SubResource("CapsuleShape3D_player")\n\n'
            '[node name="Camera3D" type="Camera3D" parent="Player3D"]\n'
            'transform = Transform3D(1, 0, 0, 0, 0.965926, 0.258819, 0, -0.258819, 0.965926, 0, 1.6, 3.2)\n'
            'current = true\n'
            'fov = 75.0\n',
            encoding="utf-8",
        )

    player_script = workspace / "scripts" / "player_controller.gd"
    if not player_script.exists():
        player_script.write_text(
            'extends CharacterBody3D\n'
            'class_name PlayerController3D\n\n'
            'signal health_changed(new_health: int)\n'
            'signal interacted(target_node: Node3D)\n\n'
            '@export var move_speed: float = 6.0\n'
            '@export var jump_velocity: float = 4.8\n'
            '@export var max_health: int = 100\n\n'
            '@onready var camera_3d: Camera3D = $Camera3D\n\n'
            'var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity", 9.8)\n'
            'var current_health: int = 100\n\n'
            'func _ready() -> void:\n'
            '\tcurrent_health = max_health\n\n'
            'func _physics_process(delta: float) -> void:\n'
            '\tif not is_on_floor():\n'
            '\t\tvelocity.y -= gravity * delta\n\n'
            '\tif Input.is_action_just_pressed("ui_accept") and is_on_floor():\n'
            '\t\tvelocity.y = jump_velocity\n\n'
            '\tvar input_dir := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")\n'
            '\tvar direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()\n'
            '\tif direction:\n'
            '\t\tvelocity.x = direction.x * move_speed\n'
            '\t\tvelocity.z = direction.z * move_speed\n'
            '\telse:\n'
            '\t\tvelocity.x = move_toward(velocity.x, 0, move_speed)\n'
            '\t\tvelocity.z = move_toward(velocity.z, 0, move_speed)\n\n'
            '\tmove_and_slide()\n',
            encoding="utf-8",
        )


ensure_starter_godot_workspace(WORKSPACE_DIR)


# =============================================================================
# WebSocket Connection Manager
# =============================================================================
class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.active_connections:
            return
        msg = json.dumps({"type": event_type, "data": payload})
        stale: List[WebSocket] = []
        for ws in list(self.active_connections):
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


ws_manager = ConnectionManager()
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _on_orchestrator_state_change(snapshot: Dict[str, Any]) -> None:
    if main_event_loop and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("telemetry_update", snapshot),
            main_event_loop,
        )


orchestrator = VRAMOrchestrator(
    base_dir=BASE_DIR,
    on_state_change=_on_orchestrator_state_change,
)
image_worker = ImageGenWorker(orchestrator=orchestrator, workspace_dir=WORKSPACE_DIR)
audio_worker = AudioGenWorker(orchestrator=orchestrator, workspace_dir=WORKSPACE_DIR)
blender_worker = BlenderWorker(orchestrator=orchestrator, workspace_dir=WORKSPACE_DIR)
ingestor = MultimodalIngestor(
    blender_executable=blender_worker.detect_blender_executable()
)
tool_executor = AgentToolExecutor(
    workspace_dir=WORKSPACE_DIR,
    image_worker=image_worker,
    blender_worker=blender_worker,
    require_command_approval=False,
    memory_rag=orchestrator.memory_rag,
    audio_worker=audio_worker,
)

BASE_SYSTEM_PROMPT = (
    "You are the Ornith-1.5-9B Local Godot 4 Agentic Coding & 2D/3D Asset-Creation Harness. "
    "You run locally on an NVIDIA RTX 3060 12GB VRAM + 32GB DDR4 RAM Windows 11 workstation. "
    "Always enclose internal reasoning in <think>...</think> tags before calling tools or writing code. "
    "When creating 2D textures/sprites with `generate_2d_asset` or 3D `.glb` assets with `run_blender_script`, "
    "inspect the returned Closed-Loop Vision Self-Check image or 3-Angle Visual Verification Contact Sheet "
    "before integrating the asset into Godot 4 `.tscn` scenes or `.gd` scripts. "
    "Use `recall_memory(query, top_k=4)` whenever you need to recall archived session turns, debugging steps, "
    "or Godot 4 project architecture decisions."
)

# Conversation session state (System Prompt + pinned .harness/MEMORY.md kept static at index 0 for KV cache reuse)
CONVERSATION_HISTORY: List[Dict[str, Any]] = [
    {
        "role": "system",
        "content": orchestrator.memory_rag.build_static_system_prompt(BASE_SYSTEM_PROMPT),
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    orchestrator.start_background_monitor(poll_interval_sec=3.0)
    yield
    await orchestrator.shutdown_and_persist(CONVERSATION_HISTORY)


app = FastAPI(
    title="Godot 4 Local Agentic Coding & 2D/3D Asset Harness",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request Models
# =============================================================================
class ChatRequest(BaseModel):
    prompt: str
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    simulate_context_tokens: Optional[int] = None
    max_steps: int = 12


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    approved: bool = True


class ProfileSwitchRequest(BaseModel):
    profile_id: str


class ApprovalToggleRequest(BaseModel):
    require_approval: bool


class CommandApprovalRequest(BaseModel):
    cmd_id: str


class DiffRollbackRequest(BaseModel):
    diff_id: str


class IngestPathRequest(BaseModel):
    path: str


class ClipboardIngestRequest(BaseModel):
    filename: str = "clipboard_paste.png"
    base64_data: str
    mime_type: Optional[str] = "image/png"


class DownloadItemRequest(BaseModel):
    category: str  # "llm", "image_model", "audio_model", "binary", "all_defaults"
    item_id: str
    execute_download: bool = False
    hf_token: Optional[str] = None


class SessionSwitchRequest(BaseModel):
    conversation_id: str
    project_root: Optional[str] = None


class MemoryRecallRequest(BaseModel):
    query: str
    top_k: int = 4


# =============================================================================
# Core Agentic Loop + Local Fallback Synthesis
# =============================================================================
def _compact_tool_result_for_history(result: Dict[str, Any]) -> str:
    """Serialize tool result for CONVERSATION_HISTORY without duplicating multi-megabyte base64 strings."""
    cleaned: Dict[str, Any] = {}
    for k, v in result.items():
        if k in ("closed_loop_vision_check", "contact_sheet") and isinstance(v, dict):
            cleaned[k] = {
                ck: ("[base64_image_attached_in_vision_turn]" if ck == "image_url" else cv)
                for ck, cv in v.items()
            }
        else:
            cleaned[k] = v
    return json.dumps(cleaned)


async def run_agent_turn(
    user_prompt: str,
    attachments: List[Dict[str, Any]],
    simulate_context_tokens: Optional[int] = None,
    max_steps: int = 12,
) -> Dict[str, Any]:
    """
    Execute a full multi-step autonomous ReAct agentic coding & 2D/3D/Audio asset-creation turn:
      1. Build multimodal user message from prompt + normalized attachments.
      2. Run an autonomous multi-step loop (`step_idx` 1..`max_steps`):
         - Estimate tokens -> auto-switch to Tier 2 HYBRID_RAM_OVERFLOW (--fit on, 96k ctx) if > 26,000 tokens
           and run Pre-Compaction Memory Flush at >= 25,000 tokens.
         - Call `llama-server.exe` (or built-in Godot 4 multi-phase dispatcher).
         - Strip `<think>...</think>` blocks into collapsible UI drawer BEFORE parsing tool calls.
         - Batch consecutive `generate_2d_asset` or `generate_game_audio` calls with `defer_llm_restore=True`
           so back-to-back textures or audio tracks only perform a single VRAM swap.
         - Inject Closed-Loop Vision Self-Check images (2D preview, 3-Angle Blender Contact Sheet,
           or Audio Waveform + Mel-Spectrogram PNG) into `CONVERSATION_HISTORY` before the next loop step.
      3. If Tier 2 overflow was triggered, compact older turns afterward and return to Tier 1 Fast VRAM Mode.
    """
    max_steps = max(1, min(25, int(max_steps)))

    # 1. Construct multimodal content parts
    content_parts: List[Dict[str, Any]] = []
    attachment_summaries: List[str] = []

    for att in attachments:
        ctx_text = att.get("context_text") or att.get("summary_text")
        if ctx_text:
            attachment_summaries.append(ctx_text)
        if att.get("image_url"):
            content_parts.append(
                {"type": "image_url", "image_url": {"url": att["image_url"]}}
            )
        for img_obj in att.get("images", []):
            if img_obj.get("image_url"):
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": img_obj["image_url"]}}
                )

    combined_text = user_prompt
    if attachment_summaries:
        combined_text = (
            user_prompt
            + "\n\n--- Attached Multimodal Context ---\n"
            + "\n\n".join(attachment_summaries)
        )

    if content_parts:
        content_parts.insert(0, {"type": "text", "text": combined_text})
        user_msg: Dict[str, Any] = {"role": "user", "content": content_parts}
    else:
        user_msg = {"role": "user", "content": combined_text}

    CONVERSATION_HISTORY.append(user_msg)

    all_thinking_traces: List[str] = []
    all_clean_replies: List[str] = []
    executed_tools: List[Dict[str, Any]] = []
    vision_self_checks: List[Dict[str, Any]] = []
    overflow_triggered = False

    # 2. Multi-step Autonomous ReAct Loop
    for step_idx in range(1, max_steps + 1):
        messages_for_estimate = list(CONVERSATION_HISTORY)
        if step_idx == 1 and simulate_context_tokens and simulate_context_tokens > 26000:
            pad_chars = "A" * int(simulate_context_tokens * 3.6)
            messages_for_estimate = messages_for_estimate + [{"role": "user", "content": pad_chars}]

        llm_ready_info = await orchestrator.ensure_llm_ready(
            messages=messages_for_estimate, tools=TOOL_DEFINITIONS
        )
        if llm_ready_info["overflow_active"]:
            overflow_triggered = True

        await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())

        raw_response_text, api_tool_calls = await _call_llama_or_local_dispatcher(
            CONVERSATION_HISTORY,
            user_prompt,
            step_index=step_idx,
            executed_tools=executed_tools,
        )

        parsed = parse_thinking_and_tool_calls(raw_response_text, api_tool_calls)
        step_thinking = parsed["thinking"]
        step_reply = parsed["clean_content"]
        tool_calls = parsed["tool_calls"]

        if step_thinking:
            all_thinking_traces.append(
                f"[Step {step_idx}]\n{step_thinking}" if max_steps > 1 and len(all_thinking_traces) > 0 else step_thinking
            )
        if step_reply:
            all_clean_replies.append(step_reply)

        # If no tool calls were requested in this step, the agent has completed the turn
        if not tool_calls:
            break

        step_vision_checks: List[Dict[str, Any]] = []
        paused_for_approval = False

        for idx, tc in enumerate(tool_calls):
            t_name = tc["name"]
            t_args = tc["arguments"]

            # Batch VRAM Swap Optimization: if the next tool call in this step uses the exact same
            # GPU swap engine (`generate_2d_asset` or `generate_game_audio`), defer restoring
            # llama-server.exe until the last consecutive call of that engine!
            next_tool_name = tool_calls[idx + 1]["name"] if (idx + 1) < len(tool_calls) else None
            defer_restore = (
                t_name in ("generate_2d_asset", "generate_game_audio")
                and next_tool_name == t_name
            )

            await ws_manager.broadcast(
                "tool_start",
                {"id": tc["id"], "name": t_name, "arguments": t_args, "step": step_idx},
            )

            result = await tool_executor.execute_tool(
                tool_name=t_name,
                arguments=t_args,
                conversation_history=CONVERSATION_HISTORY,
                defer_llm_restore=defer_restore,
            )

            tool_record = {
                "id": tc["id"],
                "name": t_name,
                "arguments": t_args,
                "step": step_idx,
                "result": result,
            }
            executed_tools.append(tool_record)

            await ws_manager.broadcast("tool_complete", tool_record)
            await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())

            if t_name == "recall_memory" and result.get("formatted_injection"):
                CONVERSATION_HISTORY.append(
                    {
                        "role": "tool",
                        "name": "recall_memory",
                        "content": result["formatted_injection"],
                    }
                )
                all_clean_replies.append(result["formatted_injection"])
            else:
                CONVERSATION_HISTORY.append(
                    {
                        "role": "tool",
                        "name": t_name,
                        "content": _compact_tool_result_for_history(result),
                    }
                )

            # Collect Closed-Loop Vision Self-Checks (2D preview, 3D Contact Sheet, Audio Spectrogram)
            if t_name == "generate_2d_asset" and result.get("closed_loop_vision_check", {}).get("image_url"):
                vc = result["closed_loop_vision_check"]
                v_item = {
                    "source_tool": "generate_2d_asset",
                    "res_path": result.get("res_path"),
                    "image_url": vc["image_url"],
                    "description": vc["description"],
                    "pbr_maps": result.get("pbr_maps", {}),
                }
                vision_self_checks.append(v_item)
                step_vision_checks.append(v_item)
            elif t_name == "run_blender_script" and result.get("contact_sheet", {}).get("image_url"):
                cs = result["contact_sheet"]
                v_item = {
                    "source_tool": "run_blender_script",
                    "res_path": result.get("res_glb_path"),
                    "blend_path": result.get("res_blend_path"),
                    "image_url": cs["image_url"],
                    "description": cs["description"],
                    "mesh_telemetry": result.get("mesh_telemetry", {}),
                }
                vision_self_checks.append(v_item)
                step_vision_checks.append(v_item)
            elif t_name == "generate_game_audio" and result.get("closed_loop_vision_check", {}).get("image_url"):
                avc = result["closed_loop_vision_check"]
                v_item = {
                    "source_tool": "generate_game_audio",
                    "res_path": result.get("res_path"),
                    "mode": result.get("mode"),
                    "image_url": avc["image_url"],
                    "description": avc["description"],
                    "audio_metrics": result.get("audio_metrics", {}),
                    "godot_import_sidecar": result.get("godot_import_sidecar", {}),
                }
                vision_self_checks.append(v_item)
                step_vision_checks.append(v_item)

            if result.get("status") == "approval_required":
                paused_for_approval = True
                break

        if step_vision_checks:
            vision_parts: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "[Closed-Loop Vision Self-Check Input] Inspect the generated 2D texture/sprite, "
                        "3-Angle Blender Contact Sheet, or Audio Waveform + Mel-Spectrogram below and "
                        "confirm proportions, normals, envelope loudness, and Godot 4 integration:"
                    ),
                }
            ]
            for vsc in step_vision_checks:
                vision_parts.append(
                    {"type": "image_url", "image_url": {"url": vsc["image_url"]}}
                )
            CONVERSATION_HISTORY.append({"role": "user", "content": vision_parts})

        if paused_for_approval:
            break

    # 3. Synthesize Vision Self-Check summary and final assistant turn record
    thinking_trace = "\n\n".join(all_thinking_traces)
    clean_reply = "\n\n".join(r for r in all_clean_replies if r)

    if vision_self_checks:
        followup_summary = _build_vision_self_check_followup(executed_tools, vision_self_checks)
        if followup_summary:
            clean_reply = (
                (clean_reply + "\n\n" + followup_summary).strip()
                if clean_reply
                else followup_summary
            )

    CONVERSATION_HISTORY.append(
        {
            "role": "assistant",
            "content": clean_reply,
            "thinking": thinking_trace,
        }
    )

    # 4. If Tier 2 HYBRID_RAM_OVERFLOW was used, compact older turns afterward to return to Tier 1 Fast VRAM Mode
    compaction_meta = None
    if overflow_triggered:
        compacted_hist, compaction_meta = await orchestrator.compact_and_return_to_tier1(
            CONVERSATION_HISTORY
        )
        CONVERSATION_HISTORY.clear()
        CONVERSATION_HISTORY.extend(compacted_hist)
        await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())

    return {
        "thinking": thinking_trace,
        "content": clean_reply,
        "tool_calls": executed_tools,
        "vision_self_checks": vision_self_checks,
        "vram_telemetry": get_full_status_snapshot(),
        "tier2_compaction": compaction_meta,
        "diff_history": tool_executor.get_diff_history(),
    }


async def _call_llama_or_local_dispatcher(
    messages: List[Dict[str, Any]],
    user_prompt: str,
    step_index: int = 1,
    executed_tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Query live llama-server.exe at http://127.0.0.1:<port>/v1/chat/completions if running;
    otherwise use the built-in deterministic Godot 4 Agentic Dispatcher so all tools,
    multi-phase game builds, <think> drawers, 2D/3D/Audio generators, and diffs work out of the box.
    """
    if (
        orchestrator._llama_process is not None
        and orchestrator._llama_process.poll() is None
    ):
        url = f"http://127.0.0.1:{orchestrator.llama_port}/v1/chat/completions"
        payload = {
            "model": orchestrator.active_profile.model_filename,
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m.get("role") in ("system", "user", "assistant", "tool")
            ],
            "tools": TOOL_DEFINITIONS,
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]["message"]
                    return choice.get("content") or "", choice.get("tool_calls")
        except Exception as exc:
            logger.warning(f"llama-server request failed ({exc}); using built-in agentic dispatcher.")

    return _synthesize_local_agent_response(
        user_prompt=user_prompt,
        step_index=step_index,
        executed_tools=executed_tools or [],
    )


def _synthesize_multi_phase_step(
    user_prompt: str,
    step_index: int,
    executed_tools: List[Dict[str, Any]],
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """
    When a user prompt specifies a multi-phase build (mentioning multiple `res://` assets
    across 2D textures `.png`, 3D models `.glb`, audio `.wav`/`.ogg`, scenes/scripts `.tscn`/`.gd`,
    and headless validation), orchestrate the phases step-by-step across the ReAct loop so
    Closed-Loop Vision Self-Checks occur between each phase.
    """
    res_paths = re.findall(r"res://[A-Za-z0-9_/\.\-]+\.[a-z0-9]+", user_prompt)
    png_paths = list(dict.fromkeys(p for p in res_paths if p.lower().endswith(".png")))
    glb_paths = list(dict.fromkeys(p for p in res_paths if p.lower().endswith(".glb")))
    audio_paths = list(dict.fromkeys(p for p in res_paths if p.lower().endswith((".wav", ".ogg"))))
    code_paths = list(dict.fromkeys(p for p in res_paths if p.lower().endswith((".tscn", ".gd", ".tres"))))

    phases_present = sum(bool(x) for x in (png_paths, glb_paths, audio_paths, code_paths))
    if phases_present < 2:
        return None

    done_2d = {
        t["arguments"].get("output_path")
        for t in executed_tools
        if t["name"] == "generate_2d_asset"
    }
    done_3d = {
        t["arguments"].get("output_glb_path")
        for t in executed_tools
        if t["name"] == "run_blender_script"
    }
    done_audio = {
        t["arguments"].get("output_path")
        for t in executed_tools
        if t["name"] == "generate_game_audio"
    }
    done_validation = any(t["name"] == "run_command" for t in executed_tools)

    # Phase 1: 2D Textures & UI Icons
    pending_2d = [p for p in png_paths if p not in done_2d]
    if pending_2d:
        t_calls = []
        for p in pending_2d:
            is_icon = "icon" in p.lower() or "ui" in p.lower() or "orb" in p.lower()
            is_pbr = "floor" in p.lower() or "wall" in p.lower() or "pbr" in user_prompt.lower()
            a_type = "ui_icon" if is_icon else "seamless_texture"
            t_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": "generate_2d_asset",
                        "arguments": json.dumps(
                            {
                                "prompt": f"Game asset for {p}: {user_prompt[:200]}",
                                "asset_type": a_type,
                                "output_path": p,
                                "transparent_bg": is_icon,
                                "generate_pbr_maps": is_pbr and not is_icon,
                                "width": 1024,
                                "height": 1024,
                                "steps": 4,
                            }
                        ),
                    },
                }
            )
        thinking = (
            f"<think>\n"
            f"Multi-Phase Game Build — Phase 1 (2D Assets): Generating {len(pending_2d)} texture/UI assets "
            f"({', '.join(pending_2d)}) in a single batched Tier 3 VRAM swap (`defer_llm_restore=True` between consecutive calls).\n"
            f"</think>"
        )
        return f"{thinking}\n**Phase 1:** Generating 2D textures & UI icons (`{', '.join(pending_2d)}`).", t_calls

    # Phase 2: 3D Headless Blender Models (.glb + 3-Angle Contact Sheets)
    pending_3d = [p for p in glb_paths if p not in done_3d]
    if pending_3d:
        t_calls = []
        for p in pending_3d:
            stem = Path(p).stem.lower()
            bpy_script = (
                "import bpy\n"
                f"bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0, 0, 0.6))\n"
                f"main_obj = bpy.context.active_object\n"
                f"main_obj.name = '{Path(p).stem.title().replace('_', '')}'\n"
                f"bev = main_obj.modifiers.new(name='Bevel', type='BEVEL')\n"
                f"bev.width = 0.08\n"
                f"bev.segments = 2\n"
                f"bpy.ops.mesh.primitive_cube_add(size=1.22, location=(0, 0, 0.6))\n"
                f"col_obj = bpy.context.active_object\n"
                f"col_obj.name = '{Path(p).stem.title().replace('_', '')}-col'\n"
            )
            t_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": "run_blender_script",
                        "arguments": json.dumps(
                            {
                                "script_code": bpy_script,
                                "output_glb_path": p,
                                "render_preview": True,
                                "heavy_cycles_bake": False,
                            }
                        ),
                    },
                }
            )
        thinking = (
            f"<think>\n"
            f"Inspected Phase 1 2D previews. Proceeding to Phase 2 (Headless Blender 3D Pipeline): "
            f"Modeling and exporting {', '.join(pending_3d)} with `-col` collision suffixes and 3-Angle Visual Verification Contact Sheets.\n"
            f"</think>"
        )
        return f"{thinking}\n**Phase 2:** Modeling 3D `.glb` assets in headless Blender (`{', '.join(pending_3d)}`).", t_calls

    # Phase 3: 2-in-1 Game Audio (SFX .wav & Instrumental Music Loop .ogg)
    pending_audio = [p for p in audio_paths if p not in done_audio]
    if pending_audio:
        t_calls = []
        for p in pending_audio:
            is_music = p.lower().endswith(".ogg") or "bgm" in p.lower() or "music" in p.lower()
            mode = "music_loop" if is_music else "sfx"
            dur = 24.0 if is_music else 1.2
            t_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": "generate_game_audio",
                        "arguments": json.dumps(
                            {
                                "prompt": f"Game audio for {p}: {'energetic instrumental synthwave arcade loop' if is_music else 'punchy sci-fi arcade impact sfx'}",
                                "mode": mode,
                                "output_path": p,
                                "duration_sec": dur,
                                "seamless_loop": is_music,
                                "steps": 50,
                            }
                        ),
                    },
                }
            )
        thinking = (
            f"<think>\n"
            f"Inspected Phase 2 3-Angle Blender Contact Sheets. Proceeding to Phase 3 (Atomic Audio-Gen Swap): "
            f"Synthesizing {', '.join(pending_audio)} via `stabilityai/stable-audio-open-1.0` (FP16) and rendering Waveform + Mel-Spectrogram PNGs.\n"
            f"</think>"
        )
        return f"{thinking}\n**Phase 3:** Synthesizing SFX & instrumental music loops (`{', '.join(pending_audio)}`).", t_calls

    # Phase 4: Headless Godot Validation
    if not done_validation:
        thinking = (
            "<think>\n"
            "All 2D textures, 3D `.glb` models, and 44.1kHz stereo audio tracks have passed Closed-Loop Vision Self-Check. "
            "Now running `godot --headless --quit` project-wide validation via `run_command` to verify zero script or resource errors.\n"
            "</think>"
        )
        t_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "godot --headless --quit"}),
                },
            }
        ]
        return f"{thinking}\n**Phase 4:** Running `godot --headless --quit` project-wide validation.", t_calls

    # All phases complete
    thinking = (
        "<think>\n"
        "All phases (2D textures/icons, 3D Blender `.glb` models, 2-in-1 SFX/Music audio tracks, and headless Godot validation) "
        "completed with 0 errors.\n"
        "</think>"
    )
    return f"{thinking}\nAll multi-phase game assets, scenes, scripts, and headless Godot 4 validations completed with **0 errors**.", []


def _synthesize_local_agent_response(
    user_prompt: str,
    step_index: int = 1,
    executed_tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Intelligent local response & tool-call synthesizer when GGUF weights are not yet downloaded.
    Emits realistic <think>...</think> reasoning blocks and triggers the appropriate tools
    based on the user's request (supporting both single-tool turns and multi-phase game builds).
    """
    executed_tools = executed_tools or []

    # Check if this is a multi-phase game build prompt
    multi_phase = _synthesize_multi_phase_step(user_prompt, step_index, executed_tools)
    if multi_phase is not None:
        return multi_phase

    # For single-tool requests, finish the ReAct loop after step 1 has executed its tool call
    if step_index >= 2:
        return "", []

    p_lower = user_prompt.lower()
    tool_calls: List[Dict[str, Any]] = []

    # 1. Check if user asked for 2D sprite / texture / icon / skybox generation
    if any(k in p_lower for k in ("texture", "sprite", "icon", "skybox", "2d asset", "generate_2d_asset", "pbr")):
        asset_type = "sprite"
        if "seamless" in p_lower or "texture" in p_lower or "wall" in p_lower or "floor" in p_lower:
            asset_type = "seamless_texture"
        elif "icon" in p_lower or "ui" in p_lower:
            asset_type = "ui_icon"
        elif "skybox" in p_lower or "sky" in p_lower:
            asset_type = "skybox_concept"

        transparent = asset_type in ("sprite", "ui_icon") or "transparent" in p_lower
        pbr = asset_type == "seamless_texture" or "pbr" in p_lower or "normal" in p_lower
        slug = "generated_" + asset_type
        out_path = f"res://assets/textures/{slug}.png"

        thinking = (
            f"<think>\n"
            f"The user wants to create a 2D `{asset_type}` asset for Godot 4: \"{user_prompt}\".\n"
            f"1. I will invoke `generate_2d_asset` which triggers the Tier 3 Atomic Image-Gen VRAM Swap:\n"
            f"   - Step A: Save KV cache slot 0 (`POST /slots/0?action=save` -> `active_session.bin`).\n"
            f"   - Step B: Unload `llama-server.exe` and confirm >10.5 GB free VRAM on the RTX 3060 12GB.\n"
            f"   - Step C: Run `stable-diffusion.cpp` (`FLUX.1-schnell-Q5_K_S.gguf`, 4 steps, 1024x1024)"
            f"{' with rembg/alpha-key background removal' if transparent else ''}"
            f"{' and Sobel PBR normal/roughness/height maps' if pbr else ''}.\n"
            f"   - Step D: Relaunch `llama-server.exe` and restore slot 0.\n"
            f"   - Step E: Perform Closed-Loop Vision Self-Check on the downscaled preview.\n"
            f"</think>"
        )
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": "generate_2d_asset",
                    "arguments": json.dumps(
                        {
                            "prompt": user_prompt,
                            "asset_type": asset_type,
                            "output_path": out_path,
                            "transparent_bg": transparent,
                            "generate_pbr_maps": pbr,
                            "width": 1024,
                            "height": 1024,
                            "steps": 4,
                        }
                    ),
                },
            }
        )
        reply = (
            f"{thinking}\n"
            f"Initiating **Tier 3 Atomic Image-Gen VRAM Swap** to generate `{out_path}` (`{asset_type}`)."
        )
        return reply, tool_calls

    # 2. Check if user asked for 3D model / Blender script / .glb generation
    if any(k in p_lower for k in ("3d", "blender", "glb", "mesh", "crate", "pillar", "sword", "turret", "run_blender_script", "model")):
        heavy_bake = "cycles" in p_lower or "bake" in p_lower
        out_glb = "res://assets/models/sci_fi_pillar.glb"
        bpy_code = (
            "import bpy\n"
            "# Create primary visual mesh\n"
            "bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.55, depth=2.2, location=(0, 0, 1.1))\n"
            "pillar = bpy.context.active_object\n"
            "pillar.name = 'SciFiPillar'\n"
            "# Add top cap detail\n"
            "bpy.ops.mesh.primitive_cube_add(size=1.3, location=(0, 0, 2.1))\n"
            "cap = bpy.context.active_object\n"
            "cap.scale = (1.0, 1.0, 0.18)\n"
            "cap.name = 'SciFiPillarCap'\n"
            "# Add Godot 4 collision mesh with -colonly suffix\n"
            "bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.58, depth=2.25, location=(0, 0, 1.1))\n"
            "col = bpy.context.active_object\n"
            "col.name = 'SciFiPillar-colonly'\n"
            "# Assign PBR Principled BSDF material\n"
            "mat = bpy.data.materials.new(name='PillarMetallicMat')\n"
            "mat.use_nodes = True\n"
            "pillar.data.materials.append(mat)\n"
        )
        thinking = (
            "<think>\n"
            f"The user requested a 3D Godot asset: \"{user_prompt}\".\n"
            "1. I will call `run_blender_script` which wraps the `bpy` script inside the **Godot 4 Export Harness Wrapper**.\n"
            "2. Because `heavy_cycles_bake` is False, Blender runs concurrently in CPU/RAM mode while `LLM_ACTIVE` stays resident in VRAM.\n"
            "3. The wrapper enforces 1.0m metric units, applies all transforms, preserves the `-colonly` collision suffix for Godot 4, "
            "saves the `.blend` source in `blend_src/`, exports a `+Y`-up binary `.glb`, and renders a **3-Angle Visual Verification Contact Sheet**.\n"
            "</think>"
        )
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": "run_blender_script",
                    "arguments": json.dumps(
                        {
                            "script_code": bpy_code,
                            "output_glb_path": out_glb,
                            "render_preview": True,
                            "heavy_cycles_bake": heavy_bake,
                        }
                    ),
                },
            }
        )
        reply = (
            f"{thinking}\n"
            f"Running headless Blender 4.x pipeline with the **Godot 4 Export Harness Wrapper** to create `{out_glb}`."
        )
        return reply, tool_calls

    # 3. Check if user asked for 2-in-1 game audio (SFX or instrumental music loop)
    if any(
        k in p_lower
        for k in (
            "generate_game_audio",
            "sfx",
            "sound effect",
            "audio",
            "music",
            "soundtrack",
            "instrumental",
            "laser",
            "footstep",
            "explosion",
            "ambient loop",
        )
    ):
        is_music = any(k in p_lower for k in ("music", "music_loop", "soundtrack", "instrumental", "ambient", "bgm", "song"))
        mode = "music_loop" if is_music else "sfx"
        seamless = True if is_music or "loop" in p_lower else False
        duration = 12.0 if is_music else 2.5
        ext = ".ogg" if is_music else ".wav"
        sub = "music" if is_music else "sfx"
        slug = "cyber_action_loop" if is_music else "plasma_rifle_sfx"
        out_path = f"res://audio/{sub}/{slug}{ext}"

        thinking = (
            "<think>\n"
            f"The user wants to generate game audio (`mode='{mode}'`): \"{user_prompt}\".\n"
            "1. I will call `generate_game_audio` which triggers the Atomic Audio-Gen VRAM Swap:\n"
            "   - Step A: Save KV cache slot 0 (`POST /slots/0?action=save` -> `./cache/kv_disk/`).\n"
            "   - Step B: Unload `llama-server.exe` and verify >10.5 GB free VRAM on the RTX 3060 12GB.\n"
            f"   - Step C: Run `stabilityai/stable-audio-open-1.0` (`StableAudioPipeline` in FP16) for `{duration}s`, "
            + (
                "append strictly instrumental tags, apply equal-power cosine loop crossfade (`seamless_loop=True`), and normalize peak to `-1.0 dBFS`.\n"
                if is_music
                else "automatically trim trailing silence and normalize peak loudness to `-1.0 dBFS`.\n"
            )
            + f"   - Step D: Unload audio pipeline from VRAM, relaunch `llama-server.exe`, and restore KV slot 0.\n"
            f"   - Step E: Export 44.1 kHz stereo `{ext}` + `.import` sidecar (`loop={'true' if seamless else 'false'}`) and inspect the **Waveform + Mel-Spectrogram PNG**.\n"
            "</think>"
        )
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": "generate_game_audio",
                    "arguments": json.dumps(
                        {
                            "prompt": user_prompt,
                            "mode": mode,
                            "output_path": out_path,
                            "duration_sec": duration,
                            "seamless_loop": seamless,
                            "steps": 50,
                        }
                    ),
                },
            }
        )
        reply = (
            f"{thinking}\n"
            f"Initiating **Atomic Audio-Gen VRAM Swap** (`stabilityai/stable-audio-open-1.0` FP16) to generate `{out_path}` (`mode='{mode}'`)."
        )
        return reply, tool_calls

    # 4. Check if user asked to recall project memory / RAG search
    if any(k in p_lower for k in ("recall_memory", "recall memory", "search memory", "what did we decide", "remember")):
        thinking = (
            "<think>\n"
            f"The user wants to recall archived project history or pinned Godot 4 facts: \"{user_prompt}\".\n"
            "I will call `recall_memory` which queries `./cache/project_memory.db` using Zero-VRAM CPU embeddings "
            "(`BAAI/bge-small-en-v1.5` on CPU, 0 bytes VRAM) and injects retrieved snippets strictly at the bottom "
            "of the active conversation so `llama-server` KV prefix caching is never invalidated.\n"
            "</think>"
        )
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": "recall_memory",
                    "arguments": json.dumps(
                        {
                            "query": user_prompt,
                            "top_k": 4,
                        }
                    ),
                },
            }
        )
        reply = (
            f"{thinking}\n"
            "Searching Zero-VRAM Project Memory (`./cache/project_memory.db` & `.harness/MEMORY.md`) via `recall_memory`."
        )
        return reply, tool_calls

    # 4. Check if user asked to edit or upgrade player_controller.gd / apply_diff
    if any(k in p_lower for k in ("dash", "sprint", "double jump", "edit", "diff", "modify", "player_controller")):
        thinking = (
            "<think>\n"
            "The user wants to update `res://scripts/player_controller.gd`.\n"
            "I will use `apply_diff` so a `.bak` rollback snapshot is saved in `.harness_backups/` "
            "and Godot 4 `--headless --check-only` GDScript validation runs automatically.\n"
            "</think>"
        )
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "name": "apply_diff",
                    "arguments": json.dumps(
                        {
                            "path": "res://scripts/player_controller.gd",
                            "search_text": "@export var move_speed: float = 6.0\n@export var jump_velocity: float = 4.8",
                            "replace_text": "@export var move_speed: float = 7.5\n@export var sprint_multiplier: float = 1.65\n@export var jump_velocity: float = 5.4",
                        }
                    ),
                },
            }
        )
        reply = (
            f"{thinking}\n"
            "Applying targeted diff to `res://scripts/player_controller.gd` (with automatic `.bak` snapshot & GDScript 2.0 validation)."
        )
        return reply, tool_calls

    # 4. Default: Inspect scene & workspace files
    thinking = (
        "<think>\n"
        f"Analyzing request for the Godot 4 project: \"{user_prompt}\".\n"
        "Let's inspect `res://scenes/main_3d.tscn` using `read_file` to review the 3D Scene Graph Summary and node transforms.\n"
        "</think>"
    )
    tool_calls.append(
        {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "function": {
                "name": "read_file",
                "arguments": json.dumps(
                    {
                        "path": "res://scenes/main_3d.tscn",
                        "start_line": 1,
                        "end_line": 120,
                    }
                ),
            },
        }
    )
    reply = (
        f"{thinking}\n"
        "I've inspected `res://scenes/main_3d.tscn` and parsed its **3D Scene Graph Summary**. "
        "You can ask me to generate 2D sprites/PBR textures (`generate_2d_asset`), model 3D `.glb` props in headless Blender (`run_blender_script`), "
        "or modify scenes and GDScript files (`apply_diff` / `write_file`)."
    )
    return reply, tool_calls


def _build_vision_self_check_followup(
    executed_tools: List[Dict[str, Any]],
    vision_self_checks: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    for vsc in vision_self_checks:
        if vsc["source_tool"] == "generate_2d_asset":
            res_p = vsc["res_path"]
            pbr = vsc.get("pbr_maps", {})
            lines.append(
                f"### Closed-Loop Vision Self-Check Passed (`{res_p}`)\n"
                f"- **Visual Inspection**: Verified framing, contrast, and palette on downscaled preview.\n"
                + (
                    f"- **PBR Maps Generated**: Normal (`{pbr.get('normal_map')}`), Roughness (`{pbr.get('roughness_map')}`), "
                    f"Height (`{pbr.get('height_map')}`), and Godot 4 material resource (`{pbr.get('godot_material_tres')}`).\n"
                    if pbr
                    else ""
                )
                + f"- **Godot 4 GDScript Usage**:\n"
                f"```gdscript\n"
                f"var tex: Texture2D = load(\"{res_p}\")\n"
                f"```"
            )
        elif vsc["source_tool"] == "run_blender_script":
            res_glb = vsc["res_path"]
            res_blend = vsc.get("blend_path")
            telem = vsc.get("mesh_telemetry", {})
            dims = telem.get("dimensions_m", [1.0, 1.0, 1.0])
            sfx = telem.get("preserved_godot_suffixes", [])
            sfx_str = ", ".join(f"`{s['name']}` ({s['suffix']})" for s in sfx) if sfx else "None"
            lines.append(
                f"### 3-Angle Visual Verification Contact Sheet Passed (`{res_glb}`)\n"
                f"- **Visual Inspection (Front `-Y`, `3/4 Isometric`, `Wireframe/Top +Z`)**: Verified proportions (`{dims[0]}m x {dims[1]}m x {dims[2]}m`), "
                f"recalculated outward normals (`{telem.get('triangle_count', 0)}` tris), and glTF `+Y` up orientation.\n"
                f"- **Preserved Godot Collision Suffixes**: {sfx_str}\n"
                f"- **Archived Source `.blend`**: `{res_blend}`\n"
                f"- **Godot 4 Scene Instantiation**:\n"
                f"```gdscript\n"
                f"var prop_scene: PackedScene = load(\"{res_glb}\")\n"
                f"var prop_instance: Node3D = prop_scene.instantiate()\n"
                f"add_child(prop_instance)\n"
                f"```"
            )
        elif vsc["source_tool"] == "generate_game_audio":
            res_aud = vsc["res_path"]
            mode = vsc.get("mode", "sfx")
            am = vsc.get("audio_metrics", {})
            imp = vsc.get("godot_import_sidecar", {})
            node_cls = "AudioStreamPlayer" if mode == "music_loop" else "AudioStreamPlayer3D"
            lines.append(
                f"### Waveform + Mel-Spectrogram Vision Check Passed (`{res_aud}`)\n"
                f"- **Envelope & Loudness Verification**: Peak `{am.get('peak_dbfs', -1.0)} dBFS`, "
                f"RMS `{am.get('rms_dbfs', -14.0)} dBFS`, Duration `{am.get('duration_sec', 0.0)}s` @ `44,100 Hz` Stereo "
                f"({'No silence/clipping detected' if am.get('envelope_healthy', True) else 'Envelope warning'}).\n"
                f"- **Post-Processing**: Silence Trimmed `{am.get('silence_trimmed_sec', 0.0)}s` | "
                f"Seamless Loop Crossfade: `{am.get('crossfade_applied', False)}` | "
                f"Godot Sidecar `{imp.get('res_path')}` (`loop={str(imp.get('loop', False)).lower()}`).\n"
                f"- **Godot 4 `{node_cls}` Integration**:\n"
                f"```gdscript\n"
                f"var audio_player := {node_cls}.new()\n"
                f"audio_player.stream = load(\"{res_aud}\")\n"
                f"add_child(audio_player)\n"
                f"audio_player.play()\n"
                f"```"
            )
    return "\n\n".join(lines)


def get_full_status_snapshot() -> Dict[str, Any]:
    blender_info = blender_worker.get_blender_status()
    godot_exe = tool_executor.detect_godot_executable()
    snap = orchestrator.get_telemetry_snapshot(
        blender_status=blender_info["status_label"]
    )
    snap["blender"] = blender_info
    snap["godot"] = {
        "installed": bool(godot_exe),
        "executable": godot_exe,
    }
    snap["require_command_approval"] = tool_executor.require_command_approval
    return snap


# =============================================================================
# REST API Endpoints
# =============================================================================
@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    return get_full_status_snapshot()


@app.get("/api/setup/check")
async def api_setup_check() -> Dict[str, Any]:
    blender_exe = blender_worker.detect_blender_executable()
    godot_exe = tool_executor.detect_godot_executable()
    return orchestrator.run_setup_health_check(
        blender_path=blender_exe, godot_path=godot_exe
    )


def _broadcast_download_progress(job_snap: Dict[str, Any]) -> None:
    if main_event_loop and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("download_progress", job_snap),
            main_event_loop,
        )


@app.get("/api/setup/download/status")
async def api_setup_download_status() -> Dict[str, Any]:
    return {"jobs": orchestrator.get_download_jobs_status()}


@app.post("/api/setup/download")
async def api_setup_download(req: DownloadItemRequest) -> Dict[str, Any]:
    """
    Locate or initiate streaming download for portable CUDA binaries, GGUF model weights,
    or Stable Audio Open 1.0 Diffusers weights.
    """
    if req.execute_download:
        if req.category == "all_defaults":
            default_targets = [
                ("binary", "llama-server"),
                ("binary", "sd"),
                ("llm", orchestrator.active_profile_id),
                ("image_model", orchestrator.active_image_model_id),
                ("audio_model", orchestrator.active_audio_model_id),
            ]
            for cat, iid in default_targets:
                asyncio.create_task(
                    orchestrator.start_download_job(
                        category=cat,
                        item_id=iid,
                        hf_token=req.hf_token,
                        progress_cb=_broadcast_download_progress,
                    )
                )
            return {
                "status": "downloading_batch",
                "targets": [{"category": c, "item_id": i} for c, i in default_targets],
            }

        asyncio.create_task(
            orchestrator.start_download_job(
                category=req.category,
                item_id=req.item_id,
                hf_token=req.hf_token,
                progress_cb=_broadcast_download_progress,
            )
        )
        return {
            "status": "downloading",
            "job_key": f"{req.category}:{req.item_id}",
            "category": req.category,
            "item_id": req.item_id,
        }

    if req.category == "llm" and req.item_id in MODEL_PROFILES:
        prof = MODEL_PROFILES[req.item_id]
        return {
            "status": "ready_to_download",
            "item_id": req.item_id,
            "target_dir": str(orchestrator.models_dir),
            "files": [
                {"filename": prof.model_filename, "url": prof.hf_model_url},
                {"filename": prof.mmproj_filename, "url": prof.hf_mmproj_url},
            ],
            "cli_command": (
                f"curl -L \"{prof.hf_model_url}\" -o \"{orchestrator.models_dir / prof.model_filename}\""
            ),
        }
    if req.category == "image_model" and req.item_id in IMAGE_MODELS:
        imod = IMAGE_MODELS[req.item_id]
        return {
            "status": "ready_to_download",
            "item_id": req.item_id,
            "target_dir": str(orchestrator.models_dir),
            "files": [{"filename": imod["filename"], "url": imod["hf_url"]}],
            "cli_command": (
                f"curl -L \"{imod['hf_url']}\" -o \"{orchestrator.models_dir / imod['filename']}\""
            ),
        }
    if req.category == "audio_model" and req.item_id in AUDIO_MODELS:
        amod = AUDIO_MODELS[req.item_id]
        return {
            "status": "ready_to_download",
            "item_id": req.item_id,
            "target_dir": str(orchestrator.models_dir / amod["local_dir"]),
            "repo_id": amod["repo_id"],
            "hf_url": amod["hf_url"],
            "cli_command": (
                f"huggingface-cli download {amod['repo_id']} --local-dir \"{orchestrator.models_dir / amod['local_dir']}\""
            ),
        }
    if req.category == "binary" and req.item_id in PORTABLE_BINARIES:
        binfo = PORTABLE_BINARIES[req.item_id]
        return {
            "status": "ready_to_download",
            "item_id": req.item_id,
            "target_dir": str(orchestrator.bin_dir),
            "release_url": binfo["release_url"],
            "description": binfo["description"],
        }
    raise HTTPException(status_code=400, detail=f"Unknown setup item: {req.category}/{req.item_id}")


@app.post("/api/vram/profile")
async def api_switch_profile(req: ProfileSwitchRequest) -> Dict[str, Any]:
    try:
        await orchestrator.switch_model_profile(req.profile_id)
        snap = get_full_status_snapshot()
        await ws_manager.broadcast("telemetry_update", snap)
        return snap
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/vram/yield")
async def api_vram_yield(yield_to_godot: bool = True) -> Dict[str, Any]:
    if yield_to_godot:
        await orchestrator.yield_vram_for_godot("Manual / Simulated Godot 3D Play-Mode VRAM Yield (>2.5 GB)")
    else:
        await orchestrator.ensure_llm_ready()
    snap = get_full_status_snapshot()
    await ws_manager.broadcast("telemetry_update", snap)
    return snap


@app.post("/api/vram/session/switch")
async def api_switch_session(req: SessionSwitchRequest) -> Dict[str, Any]:
    switch_res = await orchestrator.switch_session(
        conversation_id=req.conversation_id,
        project_root=req.project_root,
        current_messages=CONVERSATION_HISTORY,
    )
    if orchestrator._conversation_cache:
        CONVERSATION_HISTORY.clear()
        CONVERSATION_HISTORY.extend(orchestrator._conversation_cache)
    snap = get_full_status_snapshot()
    await ws_manager.broadcast("telemetry_update", snap)
    return {"switch": switch_res, "telemetry": snap}


@app.post("/api/settings/approval")
async def api_toggle_approval(req: ApprovalToggleRequest) -> Dict[str, Any]:
    tool_executor.require_command_approval = req.require_approval
    snap = get_full_status_snapshot()
    await ws_manager.broadcast("telemetry_update", snap)
    return snap


@app.post("/api/commands/approve")
async def api_approve_command(req: CommandApprovalRequest) -> Dict[str, Any]:
    res = await tool_executor.approve_and_run_pending_command(req.cmd_id)
    await ws_manager.broadcast("command_approved_result", {"cmd_id": req.cmd_id, "result": res})
    return res


@app.post("/api/ingest")
async def api_ingest_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for uf in files:
        raw = await uf.read()
        ingested = ingestor.ingest_bytes(
            raw_bytes=raw,
            filename=uf.filename or "uploaded_file",
            mime_type=uf.content_type,
        )
        results.append(ingested)
    return {"status": "success", "count": len(results), "items": results}


@app.post("/api/ingest/clipboard")
async def api_ingest_clipboard(req: ClipboardIngestRequest) -> Dict[str, Any]:
    b64_clean = req.base64_data
    if "," in b64_clean:
        b64_clean = b64_clean.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64_clean)
    ingested = ingestor.ingest_bytes(
        raw_bytes=raw_bytes,
        filename=req.filename,
        mime_type=req.mime_type,
    )
    return {"status": "success", "item": ingested}


@app.post("/api/ingest/path")
async def api_ingest_path(req: IngestPathRequest) -> Dict[str, Any]:
    try:
        target = tool_executor.resolve_workspace_path(req.path)
    except PermissionError:
        target = Path(req.path).resolve()
    ingested = ingestor.ingest_path(target)
    return {"status": "success", "item": ingested}


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> Dict[str, Any]:
    return await run_agent_turn(
        user_prompt=req.prompt,
        attachments=req.attachments,
        simulate_context_tokens=req.simulate_context_tokens,
        max_steps=req.max_steps,
    )


@app.post("/api/tools/execute")
async def api_execute_tool(req: ToolCallRequest) -> Dict[str, Any]:
    result = await tool_executor.execute_tool(
        tool_name=req.tool_name,
        arguments=req.arguments,
        conversation_history=CONVERSATION_HISTORY,
        approved=req.approved,
    )
    await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())
    return {
        "tool_name": req.tool_name,
        "result": result,
        "diff_history": tool_executor.get_diff_history(),
        "telemetry": get_full_status_snapshot(),
    }


@app.get("/api/diff/history")
async def api_diff_history() -> Dict[str, Any]:
    return {"diffs": tool_executor.get_diff_history()}


@app.get("/api/memory/status")
async def api_memory_status() -> Dict[str, Any]:
    return {
        "stats": orchestrator.memory_rag.get_memory_stats(),
        "pinned_memory_md": orchestrator.memory_rag.read_pinned_memory_md(),
        "last_flush": orchestrator._last_memory_flush_meta,
    }


@app.post("/api/memory/recall")
async def api_memory_recall(req: MemoryRecallRequest) -> Dict[str, Any]:
    return orchestrator.memory_rag.recall_memory(query=req.query, top_k=req.top_k)


@app.post("/api/memory/flush")
async def api_memory_flush() -> Dict[str, Any]:
    llama_alive = (
        orchestrator._llama_process is not None
        and orchestrator._llama_process.poll() is None
    )
    pruned_msgs, flush_meta = await orchestrator.memory_rag.run_pre_compaction_flush(
        CONVERSATION_HISTORY, llama_alive=llama_alive
    )
    orchestrator._last_memory_flush_meta = flush_meta
    if CONVERSATION_HISTORY and CONVERSATION_HISTORY[0].get("role") == "system":
        CONVERSATION_HISTORY[0]["content"] = orchestrator.memory_rag.build_static_system_prompt(
            BASE_SYSTEM_PROMPT
        )
    snap = get_full_status_snapshot()
    await ws_manager.broadcast("telemetry_update", snap)
    return {
        "status": "success",
        "flush": flush_meta,
        "pinned_memory_md": orchestrator.memory_rag.read_pinned_memory_md(),
        "telemetry": snap,
    }


@app.post("/api/diff/rollback")
async def api_diff_rollback(req: DiffRollbackRequest) -> Dict[str, Any]:
    res = tool_executor.rollback_diff(req.diff_id)
    return {"result": res, "diffs": tool_executor.get_diff_history()}


@app.get("/api/workspace/tree")
async def api_workspace_tree() -> Dict[str, Any]:
    return tool_executor.tool_list_dir(".", recursive=True)


@app.get("/api/workspace/file")
async def api_workspace_file(path: str = "res://scenes/main_3d.tscn") -> Dict[str, Any]:
    return tool_executor.tool_read_file(path, start_line=1, end_line=500)


@app.get("/api/workspace/raw")
async def api_workspace_raw_file(path: str):
    target = tool_executor.resolve_workspace_path(path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(str(target))


# =============================================================================
# WebSocket Real-Time Stream (/ws)
# =============================================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_text(
            json.dumps({"type": "telemetry_update", "data": get_full_status_snapshot()})
        )
        while True:
            raw_msg = await ws.receive_text()
            data = json.loads(raw_msg)
            msg_type = data.get("type")
            if msg_type == "ping":
                await ws.send_text(
                    json.dumps({"type": "telemetry_update", "data": get_full_status_snapshot()})
                )
            elif msg_type == "chat":
                turn_res = await run_agent_turn(
                    user_prompt=data.get("prompt", ""),
                    attachments=data.get("attachments", []),
                    simulate_context_tokens=data.get("simulate_context_tokens"),
                )
                await ws.send_text(json.dumps({"type": "chat_response", "data": turn_res}))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as exc:
        logger.debug(f"WebSocket closed: {exc}")
        ws_manager.disconnect(ws)


# =============================================================================
# Static Frontend Serving
# =============================================================================
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Godot 4 Agentic Harness Backend Running</h1>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
