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
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from speech_worker import SpeechToText, SpeechUnavailableError
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
DEFAULT_WORKSPACE_DIR = BASE_DIR / "workspace"
# A HARNESS_WORKSPACE_DIR override (tests / throwaway runs) must not rewrite the user's saved project list
WORKSPACE_CONFIG_FILE = BASE_DIR / "cache" / (
    "workspace_config.test.json" if os.environ.get("HARNESS_WORKSPACE_DIR") else "workspace_config.json"
)
MAX_RECENT_PROJECTS = 20
STATIC_DIR = BASE_DIR / "static"
speech_to_text = SpeechToText(BASE_DIR / "models" / "whisper")


def _load_persisted_workspace_dir() -> Path:
    """Load the user's last selected project folder from ./cache/workspace_config.json if available."""
    env_override = os.environ.get("HARNESS_WORKSPACE_DIR")
    if env_override:
        p = Path(env_override).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    try:
        if WORKSPACE_CONFIG_FILE.exists():
            data = json.loads(WORKSPACE_CONFIG_FILE.read_text(encoding="utf-8"))
            saved = data.get("workspace_dir")
            if saved:
                p = Path(saved).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p
    except Exception as exc:
        logger.warning(f"Could not read persisted workspace config: {exc}")
    return DEFAULT_WORKSPACE_DIR


def _read_workspace_config() -> Dict[str, Any]:
    try:
        if WORKSPACE_CONFIG_FILE.exists():
            data = json.loads(WORKSPACE_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.warning(f"Could not read workspace config: {exc}")
    return {}


def _recent_project_paths(active_dir: Optional[Path] = None) -> List[str]:
    """Previously opened project folders, most recent first (the active one is always included)."""
    data = _read_workspace_config()
    saved = [str(x) for x in data.get("recent_projects", []) if isinstance(x, str) and x.strip()]
    if "recent_projects" not in data:
        # First run with this feature: seed with what we know has been opened before
        saved = [str(x) for x in (data.get("workspace_dir"), str(DEFAULT_WORKSPACE_DIR)) if x]
    if active_dir is not None:
        saved.insert(0, str(active_dir.resolve()))
    seen: set = set()
    ordered: List[str] = []
    for raw in saved:
        key = os.path.normcase(os.path.normpath(raw))
        if key not in seen:
            seen.add(key)
            ordered.append(os.path.normpath(raw))
    return ordered[:MAX_RECENT_PROJECTS]


def _write_workspace_config(active_dir: Path, recent: List[str]) -> None:
    try:
        WORKSPACE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        WORKSPACE_CONFIG_FILE.write_text(
            json.dumps(
                {"workspace_dir": str(active_dir.resolve()), "recent_projects": recent},
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"Could not save workspace config: {exc}")


def _save_persisted_workspace_dir(target_dir: Path, previous_dir: Optional[Path] = None) -> None:
    """Persist the selected project folder and move it to the top of the recent-projects list."""
    recent = _recent_project_paths(active_dir=target_dir)
    if previous_dir is not None:
        # Make sure the project we are leaving stays in the list, whatever way it was opened
        prev = os.path.normpath(str(previous_dir.resolve()))
        if os.path.normcase(prev) not in {os.path.normcase(x) for x in recent}:
            recent.insert(1, prev)
    _write_workspace_config(target_dir, recent[:MAX_RECENT_PROJECTS])


WORKSPACE_DIR = _load_persisted_workspace_dir()


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


ensure_starter_godot_workspace(DEFAULT_WORKSPACE_DIR)
if not (WORKSPACE_DIR / "project.godot").exists():
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    for _sub in ("scenes", "scripts", "assets/textures", "assets/models", "assets/audio", "assets/ui"):
        (WORKSPACE_DIR / _sub).mkdir(parents=True, exist_ok=True)


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
    snapshot["workspace_dir"] = str(WORKSPACE_DIR)
    if main_event_loop and main_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("telemetry_update", snapshot),
            main_event_loop,
        )


orchestrator = VRAMOrchestrator(
    base_dir=BASE_DIR,
    project_root=WORKSPACE_DIR,
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

BASE_SYSTEM_PROMPT = """You are a local Godot 4 game-development agent running inside the user's Godot project folder. You work by calling tools, then reporting what you actually did.

HOW TO WORK
1. Understand the request. If the user reports a bug or shows a screenshot, the problem is in THIS project - investigate it; never answer with a generic capability summary.
2. Look before you change: use list_dir / grep_search / read_file to inspect the relevant scenes (.tscn), scripts (.gd) and project.godot. Each user message ends with an up-to-date list of the project files.
3. Make the change with apply_diff (small targeted edits; search_text must match the file exactly) or write_file (new files / full rewrites). Every scene you write is loaded in Godot straight away: if the result says loads_in_godot is false, fix that scene before doing anything else.
4. Verify: after editing, call run_command with `godot --headless --quit` to validate the whole project, read any errors, and fix them. Do not say something works unless a tool result shows it.
5. Finish with a short plain-language summary: what was wrong, what you changed (file paths), and anything the user should test by hand.

RULES
- All paths are Godot `res://` paths relative to the project root.
- Target Godot 4.x / GDScript 2.0 (`@export`, `@onready`, `CharacterBody3D`, `signal.connect(callable)`, `await`). Never use Godot 3 syntax.
- Every input action a script uses (Input.is_action_pressed etc.) must be bound to keys with the set_input_action tool - never hand-edit the [input] section of project.godot. An action with empty events never fires. Scripts must also be attached to nodes in the scene, or nothing happens at runtime.
- For any job with more than two steps, call update_task_list first with all the steps, and call it again as each one is finished. Work through the list in order.
- Do not guess Godot APIs or file formats: call godot_docs("ClassName") to check a class, property or method, and godot_docs("tscn") before writing your first scene file. A .tscn file is data - never write code such as `.new()` in it.
- When Godot reports an error with a file and line number, that line is the problem: read it and fix it. Do not blame line endings, caches or re-imports, and never wait or sleep for a re-import.
- Keep working with tools until the task is done; do not stop after only reading a file. If you are blocked, say exactly what is blocking you.
- Earlier turns of this conversation are real: when the user says "it", "the game", or "that", they mean the project and work discussed above.

ASSET TOOLS
- generate_2d_asset (textures, sprites, icons), run_blender_script (3D .glb models) and generate_game_audio (sfx / music loops) each return a preview image (picture, 3-angle contact sheet, or waveform + spectrogram). Look at the preview and regenerate if it is wrong before wiring the asset into scenes or scripts.
- recall_memory searches archived earlier sessions and pinned project facts when you need a detail that is no longer in the conversation."""


def _fresh_history() -> List[Dict[str, Any]]:
    """System prompt + pinned .harness/MEMORY.md kept static at index 0 for KV prefix-cache reuse."""
    return [
        {
            "role": "system",
            "content": orchestrator.memory_rag.build_static_system_prompt(BASE_SYSTEM_PROMPT),
        }
    ]


# Conversation state for the active project (persisted to <project>/.harness/chat_history.json)
CONVERSATION_HISTORY: List[Dict[str, Any]] = _fresh_history()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    orchestrator.start_background_monitor(poll_interval_sec=3.0)
    if os.environ.get("HARNESS_OPEN_BROWSER") == "1":
        t = threading.Timer(0.8, lambda: webbrowser.open("http://127.0.0.1:8000"))
        t.daemon = True
        t.start()
    warmup_task = None
    if not _scripted_demo_enabled() and orchestrator.is_llama_binary_and_model_present():
        # Load the model in the background right away so the first prompt does not wait for it
        warmup_task = asyncio.create_task(orchestrator.ensure_llm_ready())
    yield
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
    _persist_history()
    await orchestrator.shutdown_and_persist(_strip_images_for_disk(CONVERSATION_HISTORY))


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
# Tool-call rounds allowed in one turn before the agent stops and waits for "continue".
# High by default so a whole project can be built unattended; override with HARNESS_MAX_STEPS.
MAX_STEPS_CAP = 1000
try:
    DEFAULT_MAX_STEPS = max(1, min(MAX_STEPS_CAP, int(os.environ.get("HARNESS_MAX_STEPS", "300"))))
except ValueError:
    DEFAULT_MAX_STEPS = 300
# Stop a turn when the model issues the exact same tool calls this many rounds in a row.
STUCK_REPEAT_LIMIT = 5


class ChatRequest(BaseModel):
    prompt: str
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    simulate_context_tokens: Optional[int] = None
    max_steps: int = DEFAULT_MAX_STEPS


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


class ConversationRequest(BaseModel):
    id: str


class WorkspaceSelectRequest(BaseModel):
    folder_path: str
    init_starter_if_empty: bool = False


class WorkspaceBrowseRequest(BaseModel):
    folder_path: Optional[str] = None


class WorkspaceMkdirRequest(BaseModel):
    parent_path: str
    new_folder_name: str
    select_immediately: bool = True


# =============================================================================
# Conversations (each Godot project folder keeps its own list of saved chats)
#   <project>/.harness/conversations/<id>.json   one file per conversation
#   <project>/.harness/conversations/active.txt  id of the conversation that is open
# =============================================================================
TURN_LOCK = asyncio.Lock()
# Set by POST /api/chat/stop; the running turn checks it between tokens, tool calls and steps.
STOP_REQUESTED = threading.Event()
_last_project_context_sig: Optional[str] = None
ACTIVE_CONVERSATION: Dict[str, Any] = {"id": "", "title": "New conversation", "created_at": ""}


def _conversations_dir(workspace: Optional[Path] = None) -> Path:
    return (workspace or tool_executor.workspace_dir) / ".harness" / "conversations"


def _new_conversation_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _strip_images_for_disk(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace base64 image parts with a short text marker so the saved chat stays small."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = [
                {"type": "text", "text": "[image shown earlier - not kept across restarts]"}
                if isinstance(part, dict) and part.get("type") == "image_url"
                else part
                for part in content
            ]
            m = {**m, "content": parts}
        out.append(m)
    return out


def _derive_title(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and not m.get("synthetic"):
            text = m.get("display_text")
            if text is None:
                content = m.get("content")
                text = content if isinstance(content, str) else ""
            text = " ".join(str(text).split())
            if text:
                return text[:60] + ("..." if len(text) > 60 else "")
    return "New conversation"


def _persist_history(workspace: Optional[Path] = None) -> None:
    """Save the open conversation so a restart, project switch or chat switch never wipes the agent's context."""
    try:
        messages = [m for m in CONVERSATION_HISTORY if m.get("role") != "system"]
        if not messages or not ACTIVE_CONVERSATION["id"]:
            return  # nothing said yet: do not litter the list with empty chats
        conv_dir = _conversations_dir(workspace)
        conv_dir.mkdir(parents=True, exist_ok=True)
        if ACTIVE_CONVERSATION["title"] == "New conversation":
            ACTIVE_CONVERSATION["title"] = _derive_title(messages)
        payload = {
            "id": ACTIVE_CONVERSATION["id"],
            "title": ACTIVE_CONVERSATION["title"],
            "created_at": ACTIVE_CONVERSATION["created_at"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": _strip_images_for_disk(messages),
        }
        path = conv_dir / f"{ACTIVE_CONVERSATION['id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        (conv_dir / "active.txt").write_text(ACTIVE_CONVERSATION["id"], encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Could not persist conversation: {exc}")


def _list_conversations() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    conv_dir = _conversations_dir()
    if conv_dir.exists():
        for f in conv_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                items.append(
                    {
                        "id": data.get("id") or f.stem,
                        "title": data.get("title") or "Untitled",
                        "updated_at": data.get("updated_at") or "",
                        "message_count": sum(
                            1 for m in data.get("messages", [])
                            if m.get("role") == "user" and not m.get("synthetic")
                        ),
                    }
                )
            except Exception:
                continue
    items.sort(key=lambda c: c["updated_at"], reverse=True)
    return items


def _start_new_conversation() -> None:
    """Open a blank conversation (it is only written to disk once something is said)."""
    global _last_project_context_sig
    _last_project_context_sig = None
    tool_executor.task_list = []
    ACTIVE_CONVERSATION.update(
        id=_new_conversation_id(),
        title="New conversation",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    orchestrator.conversation_id = ACTIVE_CONVERSATION["id"]
    CONVERSATION_HISTORY.clear()
    CONVERSATION_HISTORY.extend(_fresh_history())


def _open_conversation(conv_id: str) -> bool:
    """Load a saved conversation into CONVERSATION_HISTORY. Returns False if it does not exist."""
    global _last_project_context_sig
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", conv_id or ""):
        return False
    path = _conversations_dir() / f"{conv_id}.json"
    if not path.exists():
        return False
    tool_executor.task_list = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read conversation {path}: {exc}")
        return False
    restored = [
        m for m in data.get("messages", [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool", "system")
    ]
    _last_project_context_sig = None
    ACTIVE_CONVERSATION.update(
        id=conv_id,
        title=data.get("title") or "Untitled",
        created_at=data.get("created_at") or "",
    )
    orchestrator.conversation_id = conv_id
    CONVERSATION_HISTORY.clear()
    CONVERSATION_HISTORY.extend(_fresh_history())
    CONVERSATION_HISTORY.extend(restored)
    try:
        (_conversations_dir() / "active.txt").write_text(conv_id, encoding="utf-8")
    except Exception:
        pass
    logger.info(f"Opened conversation '{ACTIVE_CONVERSATION['title']}' ({len(restored)} messages)")
    return True


def _load_history_from_disk() -> None:
    """On startup / project switch: reopen the conversation that was last open for this project."""
    conv_dir = _conversations_dir()
    last_id = ""
    try:
        last_id = (conv_dir / "active.txt").read_text(encoding="utf-8").strip()
    except Exception:
        pass
    if last_id and _open_conversation(last_id):
        return
    existing = _list_conversations()
    if existing and _open_conversation(existing[0]["id"]):
        return
    _start_new_conversation()


def _history_for_ui() -> List[Dict[str, Any]]:
    """Flatten stored messages into simple user / assistant cards for the chat panel."""
    cards: List[Dict[str, Any]] = []
    for m in CONVERSATION_HISTORY:
        role = m.get("role")
        if role == "user" and not m.get("synthetic"):
            text = m.get("display_text")
            if text is None:
                content = m.get("content")
                if isinstance(content, list):
                    text = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                    )
                else:
                    text = str(content or "")
            cards.append({"role": "user", "content": text})
        elif role == "assistant":
            text = str(m.get("content") or "")
            tool_names = [
                (tc.get("function") or {}).get("name", "") for tc in (m.get("tool_calls") or [])
            ]
            if cards and cards[-1]["role"] == "assistant":
                # Merge the steps of one agent turn into a single card
                if text:
                    cards[-1]["content"] = (cards[-1]["content"] + "\n\n" + text).strip()
                cards[-1]["tools"].extend(tool_names)
            else:
                cards.append({"role": "assistant", "content": text, "tools": tool_names})
    return cards


_load_history_from_disk()


# =============================================================================
# Project Context (keeps a small model oriented in the user's project)
# =============================================================================
_CODE_SUFFIXES = {".gd", ".tscn", ".tres", ".gdshader", ".godot", ".cfg", ".json", ".txt", ".md", ".cs"}
_ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".glb", ".gltf", ".obj", ".wav", ".ogg", ".mp3", ".ttf", ".otf"}


def _build_project_context() -> str:
    """Compact, current listing of the active project so the model knows what exists without guessing."""
    root = tool_executor.workspace_dir
    code_files: List[str] = []
    asset_files: List[str] = []
    try:
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part.startswith(".") or part in ("blend_src", "__pycache__") for part in rel.parts):
                continue
            suffix = p.suffix.lower()
            res = "res://" + str(rel).replace("\\", "/")
            if suffix in _CODE_SUFFIXES:
                code_files.append(res)
            elif suffix in _ASSET_SUFFIXES:
                asset_files.append(res)
    except Exception as exc:
        logger.debug(f"Project context scan failed: {exc}")

    main_scene = None
    project_file = root / "project.godot"
    if project_file.exists():
        try:
            m = re.search(
                r'^run/main_scene\s*=\s*"([^"]+)"',
                project_file.read_text(encoding="utf-8", errors="replace"),
                flags=re.MULTILINE,
            )
            main_scene = m.group(1) if m else None
        except Exception:
            pass

    def _cap(items: List[str], limit: int) -> str:
        if not items:
            return "(none)"
        shown = ", ".join(items[:limit])
        return shown + (f", ... and {len(items) - limit} more" if len(items) > limit else "")

    return (
        "--- Project Context (auto-generated, current state of the project on disk) ---\n"
        f"Project folder: {root}\n"
        f"Main scene: {main_scene or '(not set in project.godot)'}\n"
        f"Scenes, scripts & config: {_cap(code_files, 80)}\n"
        f"Assets: {_cap(asset_files, 40)}"
    )


# =============================================================================
# LLM Transport (llama-server.exe OpenAI-compatible API)
# =============================================================================
class LLMUnavailableError(RuntimeError):
    """The local model could not be reached; the message is shown to the user verbatim."""


class LLMContextOverflowError(RuntimeError):
    """The prompt no longer fits the loaded context window."""


def _scripted_demo_enabled() -> bool:
    return os.environ.get("HARNESS_SCRIPTED_DEMO") == "1"


def _messages_for_api(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert stored history into a strictly valid OpenAI chat sequence:
      - only API fields are sent (no UI-only keys such as `thinking` / `display_text`),
      - assistant `tool_calls` are preserved and every call is answered by a `tool` message
        carrying the matching `tool_call_id`,
      - orphaned tool results (e.g. after compaction) are dropped,
      - late system messages (compaction summaries) are sent as user messages, since most
        chat templates only allow a system message at position 0.
    """
    staged: List[Dict[str, Any]] = []
    open_ids: set = set()
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if content is None:
            content = ""
        if role == "system":
            staged.append({"role": "user" if staged else "system", "content": content})
            open_ids = set()
        elif role == "user":
            staged.append({"role": "user", "content": content})
            open_ids = set()
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if not tool_calls and not content:
                continue
            msg: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            open_ids = {tc.get("id") for tc in tool_calls}
            staged.append(msg)
        elif role == "tool":
            tid = m.get("tool_call_id")
            if tid in open_ids:
                staged.append(
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }
                )

    fixed: List[Dict[str, Any]] = []
    i = 0
    while i < len(staged):
        m = staged[i]
        fixed.append(m)
        i += 1
        if m["role"] == "assistant" and m.get("tool_calls"):
            answered: set = set()
            while i < len(staged) and staged[i]["role"] == "tool":
                fixed.append(staged[i])
                answered.add(staged[i]["tool_call_id"])
                i += 1
            for tc in m["tool_calls"]:
                if tc.get("id") not in answered:
                    fixed.append(
                        {"role": "tool", "tool_call_id": tc.get("id"), "content": "[tool was not executed]"}
                    )
    return fixed


async def _call_llm(
    messages: List[Dict[str, Any]],
    user_prompt: str,
    step_index: int,
    executed_tools: List[Dict[str, Any]],
) -> Tuple[str, str, Optional[List[Dict[str, Any]]], str]:
    """
    Ask the local model for the next step. Returns (content, reasoning, tool_calls, finish_reason).
    Raises LLMUnavailableError with a human-readable reason instead of ever inventing a reply.
    """
    if _scripted_demo_enabled():
        from scripted_demo import _synthesize_local_agent_response

        text, calls = _synthesize_local_agent_response(
            user_prompt=user_prompt, step_index=step_index, executed_tools=executed_tools
        )
        return text, "", calls, "stop"

    if not orchestrator.is_llama_alive():
        raise LLMUnavailableError(orchestrator.describe_llm_unavailable())

    url = f"http://127.0.0.1:{orchestrator.llama_port}/v1/chat/completions"
    api_messages = _messages_for_api(messages)
    if tool_executor.task_list:
        # Repeated at the end of every request so the model keeps its place; not saved in the history,
        # so it costs a few tokens per step instead of piling up.
        api_messages.append(
            {
                "role": "user",
                "content": (
                    "[Automatic reminder - your task list. Do the first unchecked step next; "
                    "call update_task_list when a step is finished.]\n" + tool_executor.render_task_list()
                ),
            }
        )
    payload = {
        "model": orchestrator.active_profile.model_filename,
        "messages": api_messages,
        "tools": TOOL_DEFINITIONS,
        "temperature": 0.2,
        "max_tokens": 8192,
    }
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    raw_estimate = orchestrator.estimate_turn_tokens(api_messages, TOOL_DEFINITIONS)
    result = await _stream_chat_completion(url, payload, step_index)
    _record_real_prompt_tokens(raw_estimate, _LAST_STREAM_USAGE.get("prompt_tokens", 0))
    return result


_STREAM_FLUSH_INTERVAL_SEC = 0.08
_LAST_STREAM_USAGE: Dict[str, int] = {}

# Repetition watchdog for streamed thinking / answer text (never applied to tool-call arguments,
# where repeated lines are normal file content).
_REPEAT_CHECK_EVERY_CHARS = 300
_REPEAT_MIN_TEXT = 800
_REPEAT_BLOCK_COPIES = 4      # the same block back-to-back this many times
_REPEAT_LINE_COPIES = 6       # or the same line this many times in the last 40 lines
# Automatic "stop reasoning and act" nudges allowed in a row before the turn gives up.
MAX_NUDGES_IN_A_ROW = 3


def _find_repetition_start(text: str) -> Optional[int]:
    """
    If the end of `text` is the model looping (a block repeated back-to-back, or one line repeated
    many times), return the index to cut at so one copy of the repeated part is kept. Else None.
    """
    if len(text) < _REPEAT_MIN_TEXT:
        return None
    tail = text[-3200:]
    offset = len(text) - len(tail)

    # 1. A block of 24..800 chars repeated back-to-back at the very end
    max_period = len(tail) // _REPEAT_BLOCK_COPIES
    for period in range(24, min(800, max_period) + 1):
        block = tail[-period:]
        if not block.strip():
            continue
        if tail[-period * _REPEAT_BLOCK_COPIES :] == block * _REPEAT_BLOCK_COPIES:
            start = len(tail) - period * _REPEAT_BLOCK_COPIES
            while start - period >= 0 and tail[start - period : start] == block:
                start -= period
            return offset + start + period

    # 2. The same substantial line many times in the recent lines (loops that vary slightly in between)
    lines = tail.split("\n")[1:-1]  # drop the possibly partial first and last lines
    recent = [ln.strip() for ln in lines[-40:]]
    counts: Dict[str, int] = {}
    for ln in recent:
        if len(ln) >= 30:
            counts[ln] = counts.get(ln, 0) + 1
    for ln, n in counts.items():
        if n >= _REPEAT_LINE_COPIES:
            first = tail.find(ln)
            return offset + first + len(ln)
    return None


async def _stream_chat_completion(
    url: str, payload: Dict[str, Any], step_index: int
) -> Tuple[str, str, Optional[List[Dict[str, Any]]], str]:
    """
    POST a streaming chat completion and forward tokens to the UI as they arrive
    (`llm_delta` WebSocket events: answer text, reasoning text, and tool-call writing progress).
    Returns the fully assembled (content, reasoning, tool_calls, finish_reason).
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    _LAST_STREAM_USAGE.clear()
    calls: Dict[int, Dict[str, str]] = {}
    finish_reason = "stop"
    pending = {"content": "", "reasoning": ""}
    tool_progress: Optional[Dict[str, Any]] = None
    last_flush = time.monotonic()
    text_len = {"content": 0, "reasoning": 0}
    checked_len = {"content": 0, "reasoning": 0}
    repetition_cut: Optional[Tuple[str, int]] = None
    stopped_by_user = False

    async def flush(force: bool = False) -> None:
        nonlocal last_flush, tool_progress
        now = time.monotonic()
        if not force and now - last_flush < _STREAM_FLUSH_INTERVAL_SEC:
            return
        if not (pending["content"] or pending["reasoning"] or tool_progress):
            return
        await ws_manager.broadcast(
            "llm_delta",
            {
                "step": step_index,
                "content": pending["content"],
                "reasoning": pending["reasoning"],
                "tool_progress": tool_progress,
            },
        )
        pending["content"] = ""
        pending["reasoning"] = ""
        tool_progress = None
        last_flush = now

    try:
        # Local generation of a long file can take minutes on a 12GB GPU - never time out early.
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=10.0)) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:800]
                    if "context" in body.lower() and ("exceed" in body.lower() or "too large" in body.lower()):
                        raise LLMContextOverflowError(body)
                    raise LLMUnavailableError(
                        f"llama-server rejected the request (HTTP {resp.status_code}): {body}"
                    )
                line_iter = resp.aiter_lines().__aiter__()
                next_line: Optional["asyncio.Future[str]"] = None
                while True:
                    # Wait for the next line in short slices so a Stop request is noticed even while
                    # the model is still reading the prompt and has not produced a token yet.
                    if next_line is None:
                        next_line = asyncio.ensure_future(line_iter.__anext__())
                    done, _pending = await asyncio.wait({next_line}, timeout=0.4)
                    if STOP_REQUESTED.is_set():
                        next_line.cancel()
                        stopped_by_user = True
                        break  # leaving the stream closes the connection, which stops llama-server
                    if not done:
                        continue
                    finished, next_line = next_line, None
                    try:
                        line = finished.result()
                    except StopAsyncIteration:
                        break
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    if chunk.get("error"):
                        err_text = json.dumps(chunk["error"])[:800]
                        if "context" in err_text.lower():
                            raise LLMContextOverflowError(err_text)
                        raise LLMUnavailableError(f"llama-server reported an error while generating: {err_text}")
                    if isinstance(chunk.get("usage"), dict):
                        _LAST_STREAM_USAGE.update(
                            {k: int(v) for k, v in chunk["usage"].items() if isinstance(v, (int, float))}
                        )
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                        pending["content"] += delta["content"]
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                        pending["reasoning"] += delta["reasoning_content"]
                    for tc_delta in delta.get("tool_calls") or []:
                        slot = calls.setdefault(int(tc_delta.get("index", 0)), {"id": "", "name": "", "args": ""})
                        if tc_delta.get("id"):
                            slot["id"] = tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                        tool_progress = {"name": slot["name"], "chars": len(slot["args"]), "tail": slot["args"][-700:]}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    await flush()

                    # Repetition watchdog: stop generating as soon as the text is clearly looping
                    for kind, parts, key in (
                        ("reasoning", reasoning_parts, "reasoning_content"),
                        ("content", content_parts, "content"),
                    ):
                        if not delta.get(key):
                            continue
                        text_len[kind] += len(delta[key])
                        if text_len[kind] - checked_len[kind] < _REPEAT_CHECK_EVERY_CHARS:
                            continue
                        checked_len[kind] = text_len[kind]
                        cut_at = _find_repetition_start("".join(parts))
                        if cut_at is not None:
                            repetition_cut = (kind, cut_at)
                            break
                    if repetition_cut:
                        break  # leaving the stream closes the connection, which stops llama-server
        await flush(force=True)
        if stopped_by_user:
            calls.clear()  # a half-written tool call must not run
            finish_reason = "stopped"
        if repetition_cut:
            kind, cut_at = repetition_cut
            logger.warning(f"[AgentLoop] Step {step_index}: model was repeating itself in its {kind}; stopped it.")
            if kind == "reasoning":
                reasoning_parts = ["".join(reasoning_parts)[:cut_at]]
            else:
                content_parts = ["".join(content_parts)[:cut_at]]
            calls.clear()  # a half-written tool call cannot be trusted
            finish_reason = "repetition"
            await ws_manager.broadcast(
                "llm_status", {"message": f"Step {step_index}: model was repeating itself - nudging it to move on..."}
            )
    except (LLMUnavailableError, LLMContextOverflowError):
        raise
    except Exception as exc:
        raise LLMUnavailableError(
            f"The request to the local model (llama-server on port {orchestrator.llama_port}) failed: "
            f"{type(exc).__name__}: {exc}. {orchestrator.describe_llm_unavailable()}"
        ) from exc

    tool_calls = [
        {
            "id": c["id"] or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": c["args"] or "{}"},
        }
        for _, c in sorted(calls.items())
        if c["name"]
    ]
    return "".join(content_parts), "".join(reasoning_parts), (tool_calls or None), finish_reason


# =============================================================================
# Context management: keep long unattended turns inside the fast 32k context
# =============================================================================
# Estimated prompt tokens at which the history is condensed before the next model call.
# 32,768 context - 8,192 reserved for the reply, minus a margin because code tokenises
# worse than the chars/3.6 estimate assumes.
CONTEXT_SOFT_LIMIT_TOKENS = 20000
_CONTEXT_KEEP_RECENT_MESSAGES = 8
_CONTEXT_TAIL_BUDGET_TOKENS = 6000
_BULKY_ARG_KEYS = ("content", "script", "replace_text", "search_text", "bpy_script", "code")
_FILE_TOOLS = ("write_file", "apply_diff", "set_input_action", "generate_2d_asset", "generate_game_audio", "run_blender_script")
_SUMMARY_MARK = "[Context summary - earlier messages were condensed to save memory"


# The character-based estimate is corrected by the real prompt size the model server reports.
_TOKEN_CALIBRATION = {"ratio": 1.0, "last_real_prompt_tokens": 0}


def _estimate_history_tokens(history: List[Dict[str, Any]]) -> int:
    return int(orchestrator.estimate_turn_tokens(history, TOOL_DEFINITIONS) * _TOKEN_CALIBRATION["ratio"])


def _record_real_prompt_tokens(estimated: int, real: int) -> None:
    if estimated <= 0 or real <= 0:
        return
    ratio = max(0.7, min(4.0, real / estimated))
    # move most of the way to the new measurement, keep a little smoothing
    _TOKEN_CALIBRATION["ratio"] = round(0.3 * _TOKEN_CALIBRATION["ratio"] + 0.7 * ratio, 3)
    _TOKEN_CALIBRATION["last_real_prompt_tokens"] = real


def _message_text(m: Dict[str, Any]) -> str:
    content = m.get("content")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def _trim_old_tool_output(history: List[Dict[str, Any]], keep_recent: int = _CONTEXT_KEEP_RECENT_MESSAGES) -> int:
    """
    Stage 1: shrink bulky old material in place - long tool results, file bodies inside old
    write_file / apply_diff / Blender calls, and old preview images. Returns how many messages changed.
    """
    changed = 0
    for m in history[: max(0, len(history) - keep_recent)]:
        role = m.get("role")
        if role == "tool":
            content = m.get("content")
            if isinstance(content, str) and len(content) > 600:
                m["content"] = (
                    content[:300]
                    + "\n[... older tool output trimmed to save context; run the tool again if you need it]"
                )
                changed += 1
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args_text = fn.get("arguments") or ""
                if len(args_text) <= 800:
                    continue
                try:
                    args = json.loads(args_text)
                except Exception:
                    continue
                if not isinstance(args, dict):
                    continue
                touched = False
                for key in _BULKY_ARG_KEYS:
                    val = args.get(key)
                    if isinstance(val, str) and len(val) > 300:
                        args[key] = f"[{len(val)} characters omitted from history; read the file to see the current version]"
                        touched = True
                if touched:
                    fn["arguments"] = json.dumps(args)
                    changed += 1
        elif role == "user" and m.get("synthetic") and isinstance(m.get("content"), list):
            if any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]):
                m["content"] = _message_text(m) + " [preview images removed from history]"
                changed += 1
    return changed


def _build_task_preserving_history(history: List[Dict[str, Any]], current_request: str) -> List[Dict[str, Any]]:
    """
    Stage 2: replace older messages with one summary that keeps what the model needs to stay on task:
    the user's current request in full, earlier requests, files created or changed, the latest
    validation result and its own most recent steps - followed by the untouched recent messages.
    """
    system_msgs = [m for m in history[:1] if m.get("role") == "system"]
    body = history[len(system_msgs):]

    # Recent tail: newest messages within a token budget, starting on a clean boundary (not a tool result)
    tail_start = len(body)
    used = 0
    for i in range(len(body) - 1, -1, -1):
        cost = orchestrator.estimate_turn_tokens([body[i]])
        if used + cost > _CONTEXT_TAIL_BUDGET_TOKENS and len(body) - i > 4:
            break
        used += cost
        tail_start = i
    while tail_start < len(body) and body[tail_start].get("role") == "tool":
        tail_start += 1
    older, recent = body[:tail_start], body[tail_start:]
    if not older:
        return list(history)

    previous_summary = ""
    earlier_requests: List[str] = []
    files: List[str] = []
    progress: List[str] = []
    last_validation = ""
    call_names: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for idx, m in enumerate(body):
        in_older = idx < tail_start
        role = m.get("role")
        text = _message_text(m)
        if role == "system" and text.startswith(_SUMMARY_MARK):
            previous_summary = text
        elif role == "user" and not m.get("synthetic") and in_older:
            shown = str(m.get("display_text") or text).strip()
            if shown and shown.strip() != current_request.strip():
                earlier_requests.append(shown[:300].replace("\n", " "))
        elif role == "assistant":
            if in_older and text.strip():
                progress.append(text.strip().replace("\n", " ")[:220])
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                args = args if isinstance(args, dict) else {}
                call_names[str(tc.get("id"))] = (str(fn.get("name")), args)
                if fn.get("name") in _FILE_TOOLS:
                    path = args.get("path") or args.get("output_path") or args.get("output_glb_path") or args.get("action")
                    if path and str(path) not in files:
                        files.append(str(path))
        elif role == "tool":
            name, args = call_names.get(str(m.get("tool_call_id")), ("", {}))
            if name == "run_command" and "godot" in str(args.get("command", "")).lower():
                try:
                    parsed = json.loads(text)
                    last_validation = str(parsed.get("stdout") or parsed.get("error") or text)
                except Exception:
                    last_validation = text

    # Carry forward what an earlier summary already recorded (merged, not nested)
    if previous_summary:
        section = ""
        old_files: List[str] = []
        old_requests: List[str] = []
        old_validation: List[str] = []
        for line in previous_summary.splitlines():
            if line.startswith("## "):
                section = line
                continue
            if section.startswith("## Files created") and line.startswith("- "):
                old_files.append(line[2:].strip())
            elif section.startswith("## Earlier requests") and line.startswith("- "):
                old_requests.append(line[2:].strip())
            elif section.startswith("## Latest project validation"):
                old_validation.append(line)
        files = [f for f in old_files if f not in files] + files
        earlier_requests = old_requests + earlier_requests
        if not last_validation.strip():
            last_validation = "\n".join(old_validation)

    parts = [
        _SUMMARY_MARK + ". Everything below is still true.]",
        "## What the user asked for (the task you are working on)",
        (current_request or "(see the most recent user message)").strip()[:9000],
    ]
    if earlier_requests:
        parts += ["## Earlier requests in this conversation"] + [f"- {r}" for r in earlier_requests[-6:]]
    if files:
        parts += ["## Files created or changed so far (they exist on disk - do not recreate them blindly)"]
        parts += [f"- {f}" for f in files[-80:]]
    if tool_executor.task_list:
        parts += ["## Your task list ([x] = done)", tool_executor.render_task_list()]
    parts += ["## Latest project validation result", (last_validation.strip()[:1800] or "(the project has not been validated yet)")]
    if progress:
        parts += ["## Your most recent steps before this summary"] + [f"- {s_}" for s_ in progress[-12:]]
    parts.append(
        "Continue the task from here. Do not start over. If you need a file's current contents, read it with "
        "read_file; use list_dir to see what exists. Stay on the user's task - do not investigate unrelated things."
    )
    summary_msg = {"role": "system", "content": "\n".join(parts)}
    return system_msgs + [summary_msg] + recent


async def _condense_context(current_request: str, reason: str, force_full: bool = False) -> Dict[str, Any]:
    """
    Keep the conversation inside the fast context window. Stage 1 trims old bulky tool output;
    if that is not enough (or force_full), stage 2 summarises older messages while keeping the task.
    Mutates CONVERSATION_HISTORY. Returns what was done.
    """
    before = _estimate_history_tokens(CONVERSATION_HISTORY)
    trimmed = _trim_old_tool_output(CONVERSATION_HISTORY)
    after_trim = _estimate_history_tokens(CONVERSATION_HISTORY)
    report: Dict[str, Any] = {"before": before, "after": after_trim, "trimmed_messages": trimmed, "summarised": False}
    needs_full = force_full or after_trim > int(CONTEXT_SOFT_LIMIT_TOKENS * 0.8)
    if not needs_full and orchestrator.state != EngineState.HYBRID_RAM_OVERFLOW:
        logger.info(f"[Context] {reason}: trimmed old tool output, {before} -> {after_trim} est. tokens")
        return report

    await ws_manager.broadcast(
        "llm_status", {"message": "Condensing the earlier conversation so the model stays fast and on task..."}
    )
    try:
        # Archive the full detail into project memory (searchable with recall_memory) before dropping it
        _, flush_meta = await orchestrator.memory_rag.run_pre_compaction_flush(
            CONVERSATION_HISTORY, llama_alive=orchestrator.is_llama_alive()
        )
        orchestrator._last_memory_flush_meta = flush_meta
    except Exception as exc:
        logger.warning(f"[Context] memory flush before condensing failed (continuing): {exc}")

    condensed = _build_task_preserving_history(CONVERSATION_HISTORY, current_request)
    condensed, meta = await orchestrator.compact_and_return_to_tier1(condensed, already_compacted=True)
    CONVERSATION_HISTORY.clear()
    CONVERSATION_HISTORY.extend(condensed)
    report.update({"after": _estimate_history_tokens(CONVERSATION_HISTORY), "summarised": True, "meta": meta})
    logger.info(f"[Context] {reason}: condensed history {before} -> {report['after']} est. tokens")
    await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())
    return report


# =============================================================================
# Core Agentic Loop
# =============================================================================
_MAX_TOOL_RESULT_CHARS = 14000


def _format_tool_result_for_model(tool_name: str, result: Dict[str, Any]) -> str:
    """Render a tool result the way a small model reads best: plain text for files, trimmed JSON otherwise."""
    if tool_name == "godot_docs" and result.get("status") == "success":
        return str(result.get("text") or "")[:_MAX_TOOL_RESULT_CHARS]
    if tool_name == "read_file" and result.get("status") == "success":
        header = (
            f"[read_file {result.get('res_path')} lines {result.get('start_line')}-{result.get('end_line')} "
            f"of {result.get('total_lines')}]"
        )
        parts = [header]
        res_path = str(result.get("res_path") or "")
        if result.get("godot_scene_summary") and not res_path.endswith(".gd"):
            parts.append(str(result["godot_scene_summary"]))
        parts.append(str(result.get("numbered_content") or ""))
        text = "\n".join(parts)
    else:
        cleaned: Dict[str, Any] = {}
        for k, v in result.items():
            if k in ("path", "backup_path", "formatted_injection"):
                continue
            if k in ("closed_loop_vision_check", "contact_sheet") and isinstance(v, dict):
                cleaned[k] = {
                    ck: ("[preview image attached in the next message]" if ck == "image_url" else cv)
                    for ck, cv in v.items()
                }
            elif k == "unified_diff" and isinstance(v, str) and len(v) > 1500:
                cleaned[k] = v[:1500] + "\n... [diff truncated]"
            else:
                cleaned[k] = v
        if tool_name == "recall_memory" and result.get("formatted_injection"):
            text = str(result["formatted_injection"])
        else:
            text = json.dumps(cleaned, ensure_ascii=False, default=str)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        text = text[:_MAX_TOOL_RESULT_CHARS] + "\n... [output truncated - request a narrower range]"
    return text


async def run_agent_turn(
    user_prompt: str,
    attachments: List[Dict[str, Any]],
    simulate_context_tokens: Optional[int] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """Run one user turn (serialized: one turn at a time) and persist the conversation afterwards."""
    async with TURN_LOCK:
        orchestrator.agent_turn_active = True
        try:
            result = await _run_agent_turn_locked(
                user_prompt, attachments, simulate_context_tokens, max_steps
            )
        except Exception as exc:
            logger.exception(f"Agent turn crashed: {exc}")
            result = {
                "thinking": "",
                "content": f"**The harness hit an internal error and stopped this turn:** `{type(exc).__name__}: {exc}`",
                "tool_calls": [],
                "vision_self_checks": [],
                "vram_telemetry": get_full_status_snapshot(),
                "tier2_compaction": None,
                "diff_history": tool_executor.get_diff_history(),
                "error": str(exc),
            }
        finally:
            orchestrator.agent_turn_active = False
        _persist_history()
        return result


async def _run_agent_turn_locked(
    user_prompt: str,
    attachments: List[Dict[str, Any]],
    simulate_context_tokens: Optional[int],
    max_steps: int,
) -> Dict[str, Any]:
    """
    Multi-step tool-calling loop against the real local model:
      1. Build the multimodal user message (prompt + attachments + current project file listing).
      2. Loop: ensure the model is loaded -> ask for the next step -> run requested tools ->
         feed results (and generated-asset preview images) back -> repeat until the model
         answers without tool calls.
      3. If the turn ran in Tier 2 RAM overflow, compact older turns and return to Tier 1.
    """
    global _last_project_context_sig
    max_steps = max(1, min(MAX_STEPS_CAP, int(max_steps)))

    # 1. Construct multimodal content parts
    content_parts: List[Dict[str, Any]] = []
    attachment_summaries: List[str] = []
    seen_image_urls: set = set()
    for att in attachments:
        ctx_text = att.get("context_text") or att.get("summary_text")
        if ctx_text:
            attachment_summaries.append(ctx_text)
        # The ingestor reports a single image both as `image_url` and inside `images`: send it once
        urls = [att.get("image_url")] + [img_obj.get("image_url") for img_obj in att.get("images", [])]
        for url in urls:
            if url and url not in seen_image_urls:
                seen_image_urls.add(url)
                content_parts.append({"type": "image_url", "image_url": {"url": url}})

    combined_text = user_prompt
    if attachment_summaries:
        combined_text += "\n\n--- Attached Multimodal Context ---\n" + "\n\n".join(attachment_summaries)

    project_context = _build_project_context()
    if project_context != _last_project_context_sig:
        combined_text += "\n\n" + project_context
        _last_project_context_sig = project_context

    user_msg: Dict[str, Any] = {"role": "user", "display_text": user_prompt}
    if content_parts:
        user_msg["content"] = [{"type": "text", "text": combined_text}] + content_parts
    else:
        user_msg["content"] = combined_text
    CONVERSATION_HISTORY.append(user_msg)
    history_len_before_reply = len(CONVERSATION_HISTORY)

    all_thinking_traces: List[str] = []
    all_clean_replies: List[str] = []
    executed_tools: List[Dict[str, Any]] = []
    vision_self_checks: List[Dict[str, Any]] = []
    overflow_triggered = False
    overflow_retries = 0
    condensed_this_turn = False
    condense_floor = 0
    nudges_in_a_row = 0
    error_text: Optional[str] = None
    hit_step_limit = False
    last_call_signature: Optional[str] = None
    repeat_count = 0
    stuck_in_loop = False

    stopped_note = "_(Stopped by you. The work done so far is kept - tell me how to continue.)_"
    STOP_REQUESTED.clear()

    # 2. Multi-step tool loop
    for step_idx in range(1, max_steps + 1):
        if STOP_REQUESTED.is_set():
            all_clean_replies.append(stopped_note)
            break
        logger.info(f"[AgentLoop] Step {step_idx}/{max_steps}")
        await ws_manager.broadcast("step_start", {"step": step_idx, "max_steps": max_steps})

        # Keep the conversation inside the fast context: condense before it overflows, as often as needed
        # (After a full condensing the trigger moves up to what was achieved plus headroom, so a prompt
        #  whose fixed part is large cannot cause condensing on every single step.)
        if not _scripted_demo_enabled() and _estimate_history_tokens(CONVERSATION_HISTORY) > max(
            CONTEXT_SOFT_LIMIT_TOKENS, condense_floor + 5000
        ):
            condense_report = await _condense_context(user_prompt, f"step {step_idx} soft limit")
            if condense_report["summarised"]:
                history_len_before_reply = -1
                condensed_this_turn = True
                condense_floor = condense_report["after"]

        messages_for_estimate = list(CONVERSATION_HISTORY)
        if step_idx == 1 and simulate_context_tokens and simulate_context_tokens > 26000:
            messages_for_estimate.append({"role": "user", "content": "A" * int(simulate_context_tokens * 3.6)})

        if not _scripted_demo_enabled() and not orchestrator.is_llama_alive():
            await ws_manager.broadcast(
                "llm_status",
                {"message": f"Loading {orchestrator.active_profile.display_name} into VRAM (can take a minute)..."},
            )
        llm_ready_info = await orchestrator.ensure_llm_ready(
            messages=messages_for_estimate, tools=TOOL_DEFINITIONS
        )
        if llm_ready_info["overflow_active"]:
            overflow_triggered = True
        await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())
        await ws_manager.broadcast("llm_status", {"message": f"Step {step_idx}: model is thinking..."})

        try:
            raw_text, reasoning_text, api_tool_calls, finish_reason = await _call_llm(
                CONVERSATION_HISTORY, user_prompt, step_idx, executed_tools
            )
        except LLMContextOverflowError as exc:
            # The server refused the request as too long (the estimate was off): condense and retry.
            # Only give up if it still does not fit right after condensing, twice in a row.
            overflow_retries += 1
            if overflow_retries > 2:
                error_text = f"The conversation no longer fits the model's context window even after condensing it: {exc}"
                break
            logger.warning("Context overflow reported by llama-server -> condensing history and retrying.")
            await _condense_context(user_prompt, "server reported overflow", force_full=True)
            history_len_before_reply = -1
            condensed_this_turn = True
            continue
        except LLMUnavailableError as exc:
            error_text = str(exc)
            break

        overflow_retries = 0
        parsed = parse_thinking_and_tool_calls(raw_text, api_tool_calls)
        step_thinking = "\n\n".join(t for t in (reasoning_text.strip(), parsed["thinking"]) if t)
        step_reply = parsed["clean_content"]
        tool_calls = parsed["tool_calls"]

        if step_thinking:
            all_thinking_traces.append(f"[Step {step_idx}]\n{step_thinking}")
        if step_reply:
            all_clean_replies.append(step_reply)
        if finish_reason == "stopped" or STOP_REQUESTED.is_set():
            # Keep whatever the model had written, run none of its tool calls, end the turn
            if step_reply or step_thinking:
                stopped_msg: Dict[str, Any] = {"role": "assistant", "content": step_reply}
                if step_thinking:
                    stopped_msg["thinking"] = step_thinking
                CONVERSATION_HISTORY.append(stopped_msg)
            all_clean_replies.append(stopped_note)
            break
        cut_off = finish_reason in ("length", "repetition") and not tool_calls
        out_of_nudges = cut_off and nudges_in_a_row >= MAX_NUDGES_IN_A_ROW
        if out_of_nudges:
            reason = "kept repeating itself" if finish_reason == "repetition" else "kept running into the length limit"
            all_clean_replies.append(
                f"_(The model {reason} even after {MAX_NUDGES_IN_A_ROW} automatic nudges. "
                "Say \"continue\" to let it try again.)_"
            )
        if not cut_off:
            nudges_in_a_row = 0

        # Record the assistant step exactly as the model produced it, so the tool results
        # that follow are correctly paired with their calls on the next request.
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": step_reply}
        if step_thinking:
            assistant_msg["thinking"] = step_thinking
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in tool_calls
            ]
        CONVERSATION_HISTORY.append(assistant_msg)

        if cut_off and not out_of_nudges:
            # The model looped or ran out of tokens mid-thought: nudge it to act, the turn carries on
            nudges_in_a_row += 1
            if finish_reason == "repetition":
                notice = (
                    "[Automatic notice] You were repeating the same lines over and over, so your output was "
                    "stopped. Do not restate your plan or re-check what you already know. Decide the single "
                    "next action now and make that tool call immediately (or give the final answer)."
                )
            else:
                notice = (
                    "[Automatic notice] Your previous output hit the length limit before you called a tool "
                    "or finished. Continue now: keep your reasoning to a few sentences and make the next "
                    "tool call (or give the final answer) immediately."
                )
            CONVERSATION_HISTORY.append({"role": "user", "synthetic": True, "content": notice})
            continue

        # No tool calls -> the model has given its final answer for this turn
        if not tool_calls:
            break

        # Unattended-run guard: the same tool calls with the same arguments, round after round
        call_signature = json.dumps([[tc["name"], tc["arguments"]] for tc in tool_calls], sort_keys=True, default=str)
        repeat_count = repeat_count + 1 if call_signature == last_call_signature else 1
        last_call_signature = call_signature
        if repeat_count >= STUCK_REPEAT_LIMIT:
            stuck_in_loop = True

        step_vision_checks: List[Dict[str, Any]] = []
        paused_for_approval = False

        stopped_mid_tools = False
        for idx, tc in enumerate(tool_calls):
            if STOP_REQUESTED.is_set():
                stopped_mid_tools = True  # remaining calls are reported to the model as not executed
                break
            t_name = tc["name"]
            t_args = tc["arguments"]

            # Batch VRAM swap: when the next call uses the same GPU engine, keep the LLM
            # unloaded until the last consecutive call of that engine.
            next_tool_name = tool_calls[idx + 1]["name"] if (idx + 1) < len(tool_calls) else None
            # (Image -> audio back-to-back also skips the pointless LLM reload in between.)
            gpu_engines = ("generate_2d_asset", "generate_game_audio")
            defer_restore = t_name in gpu_engines and next_tool_name in gpu_engines

            target_hint = (
                t_args.get("output_path")
                or t_args.get("output_glb_path")
                or t_args.get("path")
                or t_args.get("command")
                or t_args.get("query")
                or ""
            )
            logger.info(f"[AgentLoop] Step {step_idx} -> {t_name} ({target_hint})")
            await ws_manager.broadcast(
                "tool_start",
                {
                    "id": tc["id"],
                    "name": t_name,
                    "arguments": t_args,
                    "target_hint": str(target_hint),
                    "step": step_idx,
                    "tool_index": idx + 1,
                    "tool_count": len(tool_calls),
                },
            )

            t_start = time.time()
            result = await tool_executor.execute_tool(
                tool_name=t_name,
                arguments=t_args,
                conversation_history=CONVERSATION_HISTORY,
                defer_llm_restore=defer_restore,
            )
            elapsed_s = round(time.time() - t_start, 2)

            tool_record = {
                "id": tc["id"],
                "name": t_name,
                "arguments": t_args,
                "target_hint": str(target_hint),
                "step": step_idx,
                "elapsed_sec": elapsed_s,
                "result": result,
            }
            executed_tools.append(tool_record)
            await ws_manager.broadcast("tool_complete", tool_record)
            await ws_manager.broadcast("telemetry_update", get_full_status_snapshot())

            CONVERSATION_HISTORY.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": t_name,
                    "content": _format_tool_result_for_model(t_name, result),
                }
            )

            # Collect preview images of generated assets for the model to inspect next step
            if t_name == "generate_2d_asset" and result.get("closed_loop_vision_check", {}).get("image_url"):
                vc = result["closed_loop_vision_check"]
                v_item = {
                    "source_tool": t_name,
                    "res_path": result.get("res_path"),
                    "image_url": vc["image_url"],
                    "description": vc.get("description"),
                    "pbr_maps": result.get("pbr_maps", {}),
                }
                vision_self_checks.append(v_item)
                step_vision_checks.append(v_item)
            elif t_name == "run_blender_script" and result.get("contact_sheet", {}).get("image_url"):
                cs = result["contact_sheet"]
                v_item = {
                    "source_tool": t_name,
                    "res_path": result.get("res_glb_path"),
                    "blend_path": result.get("res_blend_path"),
                    "image_url": cs["image_url"],
                    "description": cs.get("description"),
                    "mesh_telemetry": result.get("mesh_telemetry", {}),
                }
                vision_self_checks.append(v_item)
                step_vision_checks.append(v_item)
            elif t_name == "generate_game_audio" and result.get("closed_loop_vision_check", {}).get("image_url"):
                avc = result["closed_loop_vision_check"]
                v_item = {
                    "source_tool": t_name,
                    "res_path": result.get("res_path"),
                    "mode": result.get("mode"),
                    "image_url": avc["image_url"],
                    "description": avc.get("description"),
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
                        "[Automatic preview of the asset(s) you just generated: "
                        + ", ".join(str(v.get("res_path")) for v in step_vision_checks)
                        + "] Look at each image. If an asset is wrong (blank, wrong subject, broken shape, "
                        "silent audio), regenerate it; otherwise continue with the task."
                    ),
                }
            ]
            for vsc in step_vision_checks:
                vision_parts.append({"type": "image_url", "image_url": {"url": vsc["image_url"]}})
            CONVERSATION_HISTORY.append({"role": "user", "content": vision_parts, "synthetic": True})

        if paused_for_approval:
            all_clean_replies.append(
                "_(Paused: a shell command is waiting for your approval. Approve it, then tell me to continue.)_"
            )
            break
        if stopped_mid_tools or STOP_REQUESTED.is_set():
            all_clean_replies.append(stopped_note)
            break
        if stuck_in_loop:
            all_clean_replies.append(
                f"_(Stopped: the model repeated the same tool call {STUCK_REPEAT_LIMIT} times in a row "
                "without making progress. Tell it what to do differently, or say \"continue\".)_"
            )
            break
    else:
        hit_step_limit = True

    if hit_step_limit:
        all_clean_replies.append(
            f"_(Stopped after {max_steps} steps without finishing. Say \"continue\" to keep going.)_"
        )

    # 3. Assemble the reply shown in the chat
    thinking_trace = "\n\n".join(all_thinking_traces)
    clean_reply = "\n\n".join(r for r in all_clean_replies if r)

    if error_text:
        if len(CONVERSATION_HISTORY) == history_len_before_reply:
            # Nothing was produced: drop the user message so a retry does not duplicate it
            CONVERSATION_HISTORY.pop()
            _last_project_context_sig = None
        notice = f"**The local model could not respond.**\n\n{error_text}"
        clean_reply = f"{clean_reply}\n\n{notice}".strip() if clean_reply else notice
    elif not clean_reply:
        clean_reply = (
            "_(The model finished without writing a summary. The tool steps above show what it did.)_"
            if executed_tools
            else "_(The model returned an empty response. Try rephrasing or sending the message again.)_"
        )

    # 4. If Tier 2 RAM overflow was used, compact older turns and return to Tier 1 Fast VRAM Mode
    compaction_meta = None
    if overflow_triggered and orchestrator.state == EngineState.HYBRID_RAM_OVERFLOW:
        end_report = await _condense_context(user_prompt, "end of turn in overflow mode", force_full=True)
        compaction_meta = end_report.get("meta")

    return {
        "thinking": thinking_trace,
        "content": clean_reply,
        "tool_calls": executed_tools,
        "vision_self_checks": vision_self_checks,
        "vram_telemetry": get_full_status_snapshot(),
        "tier2_compaction": compaction_meta,
        "diff_history": tool_executor.get_diff_history(),
        "error": error_text,
    }


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
    snap["workspace_dir"] = str(tool_executor.workspace_dir)
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


@app.post("/api/chat/stop")
async def api_chat_stop() -> Dict[str, Any]:
    """Ask the running turn to stop: cuts off the current model output and skips tool calls not yet started."""
    if not TURN_LOCK.locked():
        return {"status": "idle", "message": "No turn is running."}
    STOP_REQUESTED.set()
    await ws_manager.broadcast("llm_status", {"message": "Stopping..."})
    return {"status": "stopping"}


@app.get("/api/transcribe/status")
async def api_transcribe_status() -> Dict[str, Any]:
    return speech_to_text.status()


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Speech-to-text for the composer's mic button. Runs on the CPU, so it works while the model is busy."""
    audio_bytes = await file.read()
    if len(audio_bytes) < 800:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    if len(audio_bytes) > 60 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="The recording is too long (limit 60 MB).")
    try:
        return await asyncio.to_thread(speech_to_text.transcribe, audio_bytes)
    except SpeechUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Speech-to-text failed")
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {type(exc).__name__}: {exc}") from exc


def _conversations_payload() -> Dict[str, Any]:
    return {
        "workspace_dir": str(tool_executor.workspace_dir),
        "active_id": ACTIVE_CONVERSATION["id"],
        "active_title": ACTIVE_CONVERSATION["title"],
        "conversations": _list_conversations(),
        "messages": _history_for_ui(),
    }


def _require_idle() -> None:
    if TURN_LOCK.locked():
        raise HTTPException(status_code=409, detail="The agent is still working. Wait for it to finish first.")


class ProjectPathRequest(BaseModel):
    path: str


def _projects_payload() -> Dict[str, Any]:
    active = tool_executor.workspace_dir
    active_key = os.path.normcase(os.path.normpath(str(active)))
    projects: List[Dict[str, Any]] = []
    for raw in _recent_project_paths(active_dir=active):
        folder = Path(raw)
        exists = folder.is_dir()
        conv_count = 0
        if exists:
            try:
                conv_count = len(list((folder / ".harness" / "conversations").glob("*.json")))
            except Exception:
                pass
        projects.append(
            {
                "path": raw,
                "name": folder.name or raw,
                "exists": exists,
                "active": os.path.normcase(raw) == active_key,
                "conversation_count": conv_count,
            }
        )
    return {"active_path": str(active), "projects": projects}


@app.get("/api/projects")
async def api_projects() -> Dict[str, Any]:
    """Previously opened project folders for the Projects dropdown."""
    return _projects_payload()


@app.post("/api/projects/forget")
async def api_projects_forget(req: ProjectPathRequest) -> Dict[str, Any]:
    """Remove a folder from the recent-projects list (nothing on disk is deleted)."""
    key = os.path.normcase(os.path.normpath(req.path))
    active = tool_executor.workspace_dir
    if key == os.path.normcase(os.path.normpath(str(active))):
        raise HTTPException(status_code=400, detail="Cannot remove the project that is currently open.")
    remaining = [x for x in _recent_project_paths(active_dir=active) if os.path.normcase(x) != key]
    _write_workspace_config(active, remaining)
    return _projects_payload()


@app.get("/api/chat/history")
async def api_chat_history() -> Dict[str, Any]:
    """Open conversation + the list of saved conversations for the active project."""
    return _conversations_payload()


@app.get("/api/conversations")
async def api_conversations() -> Dict[str, Any]:
    return _conversations_payload()


@app.post("/api/chat/reset")
@app.post("/api/conversations/new")
async def api_conversation_new() -> Dict[str, Any]:
    """Start a new conversation (the current one stays in the list; pinned MEMORY.md is kept)."""
    _require_idle()
    _persist_history()
    _start_new_conversation()
    return _conversations_payload()


@app.post("/api/conversations/select")
async def api_conversation_select(req: ConversationRequest) -> Dict[str, Any]:
    _require_idle()
    _persist_history()
    if req.id != ACTIVE_CONVERSATION["id"] and not _open_conversation(req.id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return _conversations_payload()


@app.post("/api/conversations/delete")
async def api_conversation_delete(req: ConversationRequest) -> Dict[str, Any]:
    _require_idle()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", req.id or ""):
        raise HTTPException(status_code=400, detail="Invalid conversation id.")
    path = _conversations_dir() / f"{req.id}.json"
    if path.exists():
        path.unlink()
    if req.id == ACTIVE_CONVERSATION["id"]:
        remaining = _list_conversations()
        if not (remaining and _open_conversation(remaining[0]["id"])):
            _start_new_conversation()
    return _conversations_payload()


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


@app.get("/api/workspace/info")
async def api_workspace_info() -> Dict[str, Any]:
    ws_path = tool_executor.workspace_dir
    proj_file = ws_path / "project.godot"
    proj_name = ws_path.name
    if proj_file.exists():
        try:
            txt = proj_file.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'config/name\s*=\s*"([^"]+)"', txt)
            if m:
                proj_name = m.group(1)
        except Exception:
            pass
    return {
        "workspace_dir": str(ws_path),
        "project_name": proj_name,
        "has_project_godot": proj_file.exists(),
    }


@app.post("/api/workspace/browse")
async def api_workspace_browse(req: WorkspaceBrowseRequest) -> Dict[str, Any]:
    """List subdirectories, quick-jump locations, and Windows drives for the in-app Folder Picker modal."""
    try:
        target = Path(req.folder_path).resolve() if req.folder_path else tool_executor.workspace_dir.parent
        if not target.exists() or not target.is_dir():
            target = tool_executor.workspace_dir
    except Exception:
        target = tool_executor.workspace_dir

    drives: List[str] = []
    if os.name == "nt":
        for d_letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            try:
                drive_root = Path(f"{d_letter}:\\")
                if drive_root.exists():
                    drives.append(str(drive_root))
            except Exception:
                continue

    quick_paths: List[Dict[str, str]] = []
    candidates = [
        ("💻 Projects Root", BASE_DIR.parent),
        ("📦 Default (./workspace)", DEFAULT_WORKSPACE_DIR),
        ("🏠 Desktop", Path.home() / "Desktop"),
        ("📂 Documents", Path.home() / "Documents"),
    ]
    for label, qp in candidates:
        try:
            if qp.exists() and qp.is_dir():
                quick_paths.append({"label": label, "path": str(qp.resolve())})
        except Exception:
            pass

    subdirs: List[Dict[str, Any]] = []
    try:
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            try:
                if entry.is_dir() and not entry.name.startswith("."):
                    try:
                        has_godot = (entry / "project.godot").exists()
                    except Exception:
                        has_godot = False
                    subdirs.append(
                        {
                            "name": entry.name,
                            "path": str(entry.resolve()),
                            "has_project_godot": has_godot,
                        }
                    )
            except Exception:
                continue
    except Exception as exc:
        logger.debug(f"Browse folder warning for {target}: {exc}")

    try:
        parent_str = str(target.parent) if target.parent != target else str(target)
    except Exception:
        parent_str = str(target)

    return {
        "current_dir": str(target),
        "parent_dir": parent_str,
        "drives": drives,
        "quick_paths": quick_paths,
        "subdirs": subdirs[:200],
    }


async def _apply_workspace_folder_switch(
    target_path: Path, init_starter_if_empty: bool = False
) -> Dict[str, Any]:
    global WORKSPACE_DIR
    _require_idle()  # never swap the project out from under a running agent turn
    target_path = target_path.resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    # Create standard Godot subdirectories if missing
    for sub in ("scenes", "scripts", "assets/textures", "assets/models", "assets/audio", "assets/ui"):
        (target_path / sub).mkdir(parents=True, exist_ok=True)

    if init_starter_if_empty and not (target_path / "project.godot").exists():
        ensure_starter_godot_workspace(target_path)
    elif not (target_path / "project.godot").exists():
        (target_path / "project.godot").write_text(
            "; Engine configuration file.\n\n"
            "config_version=5\n\n"
            "[application]\n\n"
            f'config/name="{target_path.name}"\n'
            'config/features=PackedStringArray("4.3", "Forward Plus")\n',
            encoding="utf-8",
        )

    # Each project folder keeps its own conversation: save the outgoing one, load the incoming one
    _persist_history()
    previous_dir = tool_executor.workspace_dir
    WORKSPACE_DIR = target_path
    _save_persisted_workspace_dir(target_path, previous_dir=previous_dir)
    tool_executor.set_workspace_dir(target_path)
    await orchestrator.switch_session(
        conversation_id=orchestrator.conversation_id,
        project_root=str(target_path),
        current_messages=CONVERSATION_HISTORY,
    )
    _load_history_from_disk()

    snap = get_full_status_snapshot()
    await ws_manager.broadcast("telemetry_update", snap)
    await ws_manager.broadcast("workspace_changed", {"workspace_dir": str(target_path)})
    return {
        "status": "success",
        "workspace_dir": str(target_path),
        "telemetry": snap,
    }


@app.post("/api/workspace/mkdir")
async def api_workspace_mkdir(req: WorkspaceMkdirRequest) -> Dict[str, Any]:
    """Create a new folder inside `parent_path` and optionally switch to it immediately."""
    parent = Path((req.parent_path or "").strip() or str(tool_executor.workspace_dir.parent)).resolve()
    raw_name = (req.new_folder_name or "").strip()
    if not raw_name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")
    safe_name = re.sub(r'[<>:"|?*]', "_", raw_name).strip(". /\\")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid folder name.")

    new_dir = (parent / safe_name).resolve()
    new_dir.mkdir(parents=True, exist_ok=True)
    if req.select_immediately:
        res = await _apply_workspace_folder_switch(new_dir, init_starter_if_empty=False)
        res["created_dir"] = str(new_dir)
        return res
    return {
        "status": "created",
        "created_dir": str(new_dir),
        "workspace_dir": str(tool_executor.workspace_dir),
    }


@app.post("/api/workspace/select")
async def api_workspace_select(req: WorkspaceSelectRequest) -> Dict[str, Any]:
    """Switch the active Godot 4 project folder to any local path."""
    raw = (req.folder_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="folder_path cannot be empty.")
    return await _apply_workspace_folder_switch(
        Path(raw), init_starter_if_empty=req.init_starter_if_empty
    )


@app.post("/api/workspace/pick_native")
async def api_workspace_pick_native() -> Dict[str, Any]:
    """Open a foreground Windows OS folder chooser dialog (with New Folder button) and switch to it."""
    import subprocess

    def _open_native_folder_dialog() -> Optional[str]:
        initial_dir = str(tool_executor.workspace_dir.parent)
        if os.name == "nt":
            try:
                ps_script = (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$f = New-Object System.Windows.Forms.Form; "
                    "$f.TopMost = $true; "
                    "$f.StartPosition = 'CenterScreen'; "
                    "$f.Width = 1; $f.Height = 1; "
                    "$f.ShowInTaskbar = $false; "
                    "$f.Opacity = 0; "
                    "$f.Show(); "
                    "$f.Activate(); "
                    "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    "$dlg.Description = 'Select or Create Godot 4 Project Folder'; "
                    "$dlg.ShowNewFolderButton = $true; "
                    f"$dlg.SelectedPath = '{initial_dir.replace(chr(39), chr(39)*2)}'; "
                    "if ($dlg.ShowDialog($f) -eq [System.Windows.Forms.DialogResult]::OK) { "
                    "  [Console]::Out.Write($dlg.SelectedPath) "
                    "} "
                    "$f.Close();"
                )
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                out = (proc.stdout or "").strip()
                if out:
                    return out
                if proc.returncode == 0:
                    return None
            except Exception as exc:
                logger.warning(f"PowerShell STA FolderBrowserDialog fallback: {exc}")

        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.attributes("-topmost", True)
            root.withdraw()
            root.focus_force()
            chosen = filedialog.askdirectory(
                parent=root,
                initialdir=initial_dir,
                title="Select or Create Godot 4 Project Folder",
            )
            root.destroy()
            return chosen or None
        except Exception as exc:
            logger.warning(f"Native folder dialog fallback: {exc}")
            return None

    chosen_dir = await asyncio.to_thread(_open_native_folder_dialog)
    if not chosen_dir:
        return {"status": "cancelled", "workspace_dir": str(tool_executor.workspace_dir)}
    return await _apply_workspace_folder_switch(Path(chosen_dir), init_starter_if_empty=False)


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
        return HTMLResponse(
            index_file.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse("<h1>Godot 4 Agentic Harness Backend Running</h1>")


def _free_port_if_in_use(port: int = 8000) -> None:
    """If an older instance of server.py is still holding port 8000, terminate it cleanly."""
    try:
        import psutil

        my_pid = os.getpid()
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
                pid = conn.pid
                if pid and pid != my_pid:
                    try:
                        proc = psutil.Process(pid)
                        logger.info(f"Stopping previous server process on port {port} (PID {pid})...")
                        proc.terminate()
                        proc.wait(timeout=4.0)
                    except Exception:
                        try:
                            psutil.Process(pid).kill()
                        except Exception:
                            pass
    except Exception as exc:
        logger.debug(f"Port check warning: {exc}")


if __name__ == "__main__":
    import uvicorn

    # Keep a log on disk so a server that exits or crashes leaves a record of why
    import atexit
    import faulthandler

    _log_dir = BASE_DIR / "cache" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    (_log_dir / "harness_server.log").write_text("", encoding="utf-8")  # start fresh each launch
    _file_handler = logging.FileHandler(_log_dir / "harness_server.log", mode="a", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_file_handler)
    _crash_file = open(_log_dir / "harness_crash.log", "w", encoding="utf-8")
    faulthandler.enable(file=_crash_file, all_threads=True)  # native crashes (access violation, abort)
    atexit.register(lambda: logger.info("Harness server process is exiting."))

    _free_port_if_in_use(8000)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    except BaseException as exc:
        logger.exception(f"Harness server stopped with {type(exc).__name__}: {exc}")
        raise
