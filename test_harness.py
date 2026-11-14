"""
End-to-End Verification Suite for the Standalone Local-First Godot 4
Agentic Coding & 2D/3D Asset-Creation Harness.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from pathlib import Path

from PIL import Image

from agent_tools import (
    TOOL_DEFINITIONS,
    AgentToolExecutor,
    parse_thinking_and_tool_calls,
)
from audio_gen_worker import AudioGenWorker
from blender_worker import BlenderWorker
from godot_parser import GodotParser
from image_gen_worker import ImageGenWorker
from memory_rag import ProjectMemoryRAG
from multimodal_ingest import MultimodalIngestor
from server import ensure_starter_godot_workspace, run_agent_turn
from vram_orchestrator import EngineState, VRAMOrchestrator


def test_think_parser_and_tool_extraction():
    raw = (
        "<think>\nLet me reason about JSON: {\"name\": \"fake_tool_inside_think\"}\n</think>\n"
        "Here is my response.\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "res://scenes/main_3d.tscn"}}</tool_call>'
    )
    parsed = parse_thinking_and_tool_calls(raw)
    assert "fake_tool_inside_think" in parsed["thinking"]
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "read_file"
    assert parsed["tool_calls"][0]["arguments"]["path"] == "res://scenes/main_3d.tscn"
    assert "<think>" not in parsed["clean_content"]


def test_godot_parser_tscn_and_gd():
    base = Path(__file__).parent / "workspace"
    tscn_res = GodotParser.parse_file(base / "scenes" / "main_3d.tscn")
    assert "3D/2D Scene Graph Hierarchy" in tscn_res["summary_text"]
    assert any(n["name"] == "Player3D" for n in tscn_res["nodes"])
    player_node = [n for n in tscn_res["nodes"] if n["name"] == "Player3D"][0]
    assert player_node["transform3d"]["position"] == (0.0, 1.2, 4.5)

    gd_res = GodotParser.parse_file(base / "scripts" / "player_controller.gd")
    assert gd_res["extends"] == "CharacterBody3D"
    assert len(gd_res["exports"]) >= 2


def test_multimodal_ingestor_image_resize_and_3d():
    ingestor = MultimodalIngestor()
    # Create a 2400x1800 image and verify it resizes to longest edge 1568px
    big_img = Image.new("RGB", (2400, 1800), (40, 120, 200))
    buf = io.BytesIO()
    big_img.save(buf, format="PNG")
    res = ingestor.ingest_image_bytes(buf.getvalue(), filename="huge_screenshot.png")
    assert res["resized"] is True
    assert res["dimensions"]["width"] == 1568
    assert res["dimensions"]["height"] == 1176
    assert res["image_url"].startswith("data:image/png;base64,")


def run_coro_sync(coro):
    """Drive a coroutine to completion without requiring a socketpair event loop in sandbox."""
    try:
        while True:
            coro.send(None)
    except StopIteration as stop:
        return stop.value


def test_full_async_pipeline():
    async def _run():
        base = Path(__file__).parent
        orch = VRAMOrchestrator(base_dir=base)
        assert orch.state == EngineState.LLM_ACTIVE

        # Verify llama-server CLI flags (3-Tier KV Cache Hierarchy: -ngl 99, --cram 8192, --cache-reuse 256, ./cache/kv_disk)
        cmd_tier1 = orch.build_llama_server_command(tier2_overflow=False)
        assert "--slots" in cmd_tier1
        assert "--cram" in cmd_tier1 and "8192" in cmd_tier1
        assert "--cache-reuse" in cmd_tier1 and "256" in cmd_tier1
        assert "-ngl" in cmd_tier1 and "99" in cmd_tier1
        assert "--flash-attn" in cmd_tier1
        assert "q8_0" in cmd_tier1
        assert "--jinja" in cmd_tier1
        assert orch.kv_cache_tier.value == "KV: VRAM"

        cmd_tier2 = orch.build_llama_server_command(tier2_overflow=True)
        assert "--fit" in cmd_tier2
        assert "96000" in cmd_tier2

        # Verify Tier 2 Lazy Host-RAM Spillover telemetry ("KV: RAM Spill")
        spill_msg = [{"role": "user", "content": "X" * int(18000 * 3.6)}]
        ready_spill = await orch.ensure_llm_ready(messages=spill_msg)
        assert ready_spill["kv_cache_tier"] == "KV: RAM Spill"

        # Verify Tier 3 DiskKVCacheManager (./cache/kv_disk/ persistence, session switch, and LRU eviction)
        import time as _time
        fresh_conv_id = f"shader_task_{int(_time.time() * 1000)}"
        sw1 = await orch.switch_session(
            conversation_id=fresh_conv_id,
            current_messages=[{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hello"}],
        )
        assert sw1["kv_cache_tier"] == "KV: VRAM"
        sw2 = await orch.switch_session(
            conversation_id="default",
            current_messages=[{"role": "system", "content": "Sys"}, {"role": "user", "content": "Shader turn"}],
        )
        assert sw2["kv_cache_tier"] == "KV: Disk Restored"
        snap = orch.get_telemetry_snapshot()
        assert snap["kv_cache_tier"] == "KV: Disk Restored"
        assert "KV: Disk Restored" in snap["status_bar_text"]
        assert snap["kv_hierarchy"]["tier3_disk_cache"]["max_cap_gb"] == 15.0

        # Verify LRU eviction when disk cap is exceeded
        lru_res = orch.disk_kv_manager.enforce_lru_cap(max_bytes=10)
        assert len(lru_res["evicted_sessions"]) >= 1

        # Tier 3: 2D Asset Generation + PBR Maps + Closed-Loop Vision Self-Check
        img_worker = ImageGenWorker(orchestrator=orch, workspace_dir=base / "workspace")
        img_res = await img_worker.generate_2d_asset(
            prompt="ancient stone dungeon tiles",
            asset_type="seamless_texture",
            output_path="res://assets/textures/test_dungeon_tiles.png",
            transparent_bg=False,
            generate_pbr_maps=True,
        )
        assert img_res["status"] == "success"
        assert Path(img_res["output_path"]).exists()
        assert Path(img_res["pbr_maps"]["normal_map_abs"]).exists()
        assert Path(img_res["pbr_maps"]["roughness_map_abs"]).exists()
        assert Path(img_res["pbr_maps"]["godot_material_tres_abs"]).exists()
        assert img_res["closed_loop_vision_check"]["image_url"].startswith("data:image/png;base64,")
        assert orch.state == EngineState.LLM_ACTIVE

        # Tier 4: Headless Blender 3D Pipeline + Godot 4 Export Harness + 3-Angle Contact Sheet
        blender_worker = BlenderWorker(orchestrator=orch, workspace_dir=base / "workspace")
        b_res = await blender_worker.run_blender_script(
            script_code=(
                "import bpy\n"
                "bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0)\n"
                "obj = bpy.context.active_object\n"
                "obj.name = 'TestPillar-colonly'\n"
            ),
            output_glb_path="res://assets/models/test_pillar.glb",
            render_preview=True,
            heavy_cycles_bake=False,
        )
        assert b_res["status"] == "success"
        assert Path(b_res["output_glb_path"]).exists()
        assert Path(b_res["source_blend_path"]).exists()
        assert b_res["contact_sheet"]["image_url"].startswith("data:image/png;base64,")

        # Ingest the generated .glb via MultimodalIngestor
        ingestor = MultimodalIngestor()
        glb_meta = ingestor.ingest_path(b_res["output_glb_path"])
        assert glb_meta["category"] == "3d_model"
        assert glb_meta["triangle_count"] > 0

        # AgentTools: apply_diff + rollback + recall_memory + generate_game_audio
        audio_worker = AudioGenWorker(orchestrator=orch, workspace_dir=base / "workspace")
        executor = AgentToolExecutor(
            workspace_dir=base / "workspace",
            image_worker=img_worker,
            blender_worker=blender_worker,
            memory_rag=orch.memory_rag,
            audio_worker=audio_worker,
        )
        pc_file = base / "workspace" / "scripts" / "player_controller.gd"
        pc_text = pc_file.read_text(encoding="utf-8").replace(
            "@export var max_health: int = 150", "@export var max_health: int = 100"
        )
        pc_file.write_text(pc_text, encoding="utf-8")

        diff_res = await executor.tool_apply_diff(
            path="res://scripts/player_controller.gd",
            search_text="@export var max_health: int = 100",
            replace_text="@export var max_health: int = 150",
        )
        assert diff_res["status"] == "success", f"Diff failed: {diff_res}"
        assert diff_res["gdscript_validation"]["valid"] is True

        rb_res = executor.rollback_diff(diff_res["diff_id"])
        assert rb_res["status"] == "success"

        # ---------------------------------------------------------------------
        # Zero-VRAM Pre-Compaction Memory Flush & On-Demand RAG Recall Tests
        # ---------------------------------------------------------------------
        assert any(t["function"]["name"] == "recall_memory" for t in TOOL_DEFINITIONS)
        assert orch.memory_rag.embedder.vram_bytes == 0

        # Step A: Verify raw read_file dump pruning (`[Read <path> lines X-Y]`)
        raw_read_dump = executor.tool_read_file(
            path="res://scripts/player_controller.gd", start_line=1, end_line=35
        )
        sample_turns = [
            {
                "role": "system",
                "content": "You are the Ornith-1.5-9B Local Godot 4 Harness.",
            },
            {
                "role": "user",
                "content": (
                    "We fixed the CharacterBody3D floor snapping bug in res://scenes/main_3d.tscn: "
                    "GroundBody uses collision_layer = 1 and Player3D uses collision_layer = 2, "
                    "collision_mask = 1. Autoload GameState is registered at res://scripts/game_state.gd."
                ),
            },
            {
                "role": "tool",
                "content": json.dumps(raw_read_dump),
            },
            {
                "role": "assistant",
                "content": (
                    "Confirmed: Player3D (`$Player3D/Camera3D`) uses `collision_layer = 2`, "
                    "`move_speed = 6.5`, and `@export var jump_velocity: float = 4.8`. "
                    "Always call `move_and_slide()` inside `_physics_process(delta)`."
                ),
            },
        ]

        pruned_text = ProjectMemoryRAG.prune_raw_read_file_dumps(
            sample_turns[2]["content"]
        )
        assert "[Read res://scripts/player_controller.gd lines 1-" in pruned_text

        # Step A + B: Run Pre-Compaction Memory Flush via compact_and_return_to_tier1
        compacted_msgs, compact_meta = await orch.compact_and_return_to_tier1(
            sample_turns, keep_recent_turns=2
        )
        flush_info = compact_meta["pre_compaction_flush"]
        assert flush_info["step_a_prune_and_archive"]["chunks_indexed"] >= 1
        assert flush_info["step_b_durable_facts"]["vram_bytes_used"] == 0
        assert Path(flush_info["step_b_durable_facts"]["memory_md_path"]).exists()

        # Verify static system prompt at index 0 includes pinned .harness/MEMORY.md
        assert "PINNED PROJECT MEMORY (.harness/MEMORY.md)" in compacted_msgs[0]["content"]

        # Verify Project Hash Isolation (another game folder does not pollute recall)
        other_project_rag = ProjectMemoryRAG(
            db_path=orch.cache_dir / "project_memory.db",
            project_root=base / "cache" / "other_godot_game_isolated",
        )
        isolated_recall = other_project_rag.recall_memory("CharacterBody3D collision_layer GameState", top_k=4)
        assert isolated_recall["project_hash"] != orch.memory_rag.project_hash
        assert isolated_recall["retrieved_count"] == 0

        # Verify on-demand recall_memory tool execution on active project
        recall_res = await executor.execute_tool(
            tool_name="recall_memory",
            arguments={"query": "Player3D collision_layer floor snapping GameState", "top_k": 4},
        )
        assert recall_res["status"] == "success"
        assert recall_res["vram_bytes_used"] == 0
        assert recall_res["retrieved_count"] >= 1
        assert "[RECALLED PROJECT MEMORY" in recall_res["formatted_injection"]

        # ---------------------------------------------------------------------
        # Tier 3B: 2-in-1 Audio Generator (Stable Audio Open 1.0 FP16) Tests
        # ---------------------------------------------------------------------
        assert any(t["function"]["name"] == "generate_game_audio" for t in TOOL_DEFINITIONS)

        # 1. Test mode="sfx" (0.5s-10.0s, trailing silence trim, -1.0 dBFS peak, 44.1kHz stereo .wav + .import + spectrogram)
        sfx_res = await executor.execute_tool(
            tool_name="generate_game_audio",
            arguments={
                "prompt": "heavy sci-fi plasma laser shot with crisp metallic transient",
                "mode": "sfx",
                "output_path": "res://audio/sfx/test_laser_shot.wav",
                "duration_sec": 2.4,
                "seamless_loop": False,
            },
        )
        assert sfx_res["status"] == "success"
        assert sfx_res["mode"] == "sfx"
        assert sfx_res["sample_rate_hz"] == 44100
        assert sfx_res["channels"] == 2
        assert 0.5 <= sfx_res["duration_sec"] < 2.4  # Trailing silence was trimmed
        assert sfx_res["silence_trimmed_sec"] > 0.0
        assert abs(sfx_res["audio_metrics"]["peak_dbfs"] - (-1.0)) <= 0.15
        assert sfx_res["audio_metrics"]["envelope_healthy"] is True
        assert sfx_res["vram_swap_lifecycle"]["step_ab_acquire"]["state"] == "AUDIOGEN_ACTIVE"
        assert sfx_res["vram_swap_lifecycle"]["step_d_restore"]["state"] == "LLM_ACTIVE"
        assert orch.state == EngineState.LLM_ACTIVE

        wav_path = Path(sfx_res["output_path"])
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 44100

        sfx_import = Path(sfx_res["godot_import_sidecar"]["path"])
        assert sfx_import.exists()
        assert "loop=false" in sfx_import.read_text(encoding="utf-8")
        assert Path(sfx_res["spectrogram_png_path"]).exists()
        assert sfx_res["closed_loop_vision_check"]["image_url"].startswith("data:image/png;base64,")

        # 2. Test mode="music_loop" (10.0s-45.0s, strictly instrumental tags, seamless_loop=True crossfade, .ogg + .import loop=true)
        music_res = await executor.execute_tool(
            tool_name="generate_game_audio",
            arguments={
                "prompt": "dark cyberpunk synthwave boss arena soundtrack",
                "mode": "music_loop",
                "output_path": "res://audio/music/test_boss_loop.ogg",
                "duration_sec": 10.5,
                "seamless_loop": True,
            },
        )
        assert music_res["status"] == "success"
        assert music_res["mode"] == "music_loop"
        assert "strictly instrumental" in music_res["conditioned_prompt"]
        assert music_res["seamless_loop"] is True
        assert music_res["crossfade_applied"] is True
        assert abs(music_res["audio_metrics"]["peak_dbfs"] - (-1.0)) <= 0.15
        assert music_res["audio_metrics"]["envelope_healthy"] is True
        assert Path(music_res["output_path"]).exists()

        music_import = Path(music_res["godot_import_sidecar"]["path"])
        assert music_import.exists()
        assert "loop=true" in music_import.read_text(encoding="utf-8")
        assert Path(music_res["spectrogram_png_path"]).exists()
        assert music_res["closed_loop_vision_check"]["image_url"].startswith("data:image/png;base64,")

        # 3. Verify `godot --headless --quit` Project-Wide Scene/Resource/GDScript Validation
        proj_val = await executor.execute_tool(
            tool_name="run_command",
            arguments={"command": "godot --headless --quit"},
        )
        assert proj_val["status"] == "success"
        assert proj_val["exit_code"] == 0
        assert proj_val["valid"] is True

        # 4. Verify Multi-Step Autonomous ReAct Loop + Batched VRAM Swap Deferral across all phases
        multi_res = await run_agent_turn(
            user_prompt=(
                "Build a vertical slice: generate `res://assets/textures/arena_floor.png` (PBR) and "
                "`res://assets/ui/powerup_icon.png` (ui_icon), model `res://assets/models/paddle.glb` in Blender, "
                "synthesize `res://assets/audio/sfx_paddle_hit.wav`, and validate with `godot --headless --quit`."
            ),
            attachments=[],
            max_steps=8,
        )
        called_names = [tc["name"] for tc in multi_res["tool_calls"]]
        assert "generate_2d_asset" in called_names
        assert "run_blender_script" in called_names
        assert "generate_game_audio" in called_names
        assert "run_command" in called_names
        # Verify the first of two consecutive 2D asset calls deferred LLM restore (batched VRAM swap)
        two_d_calls = [tc for tc in multi_res["tool_calls"] if tc["name"] == "generate_2d_asset"]
        assert len(two_d_calls) == 2
        assert two_d_calls[0]["result"]["vram_swap_telemetry"]["step_d_restore"].get("deferred") is True
        assert len(multi_res["vision_self_checks"]) >= 4

    run_coro_sync(_run())


if __name__ == "__main__":
    test_think_parser_and_tool_extraction()
    test_godot_parser_tscn_and_gd()
    test_multimodal_ingestor_image_resize_and_3d()
    test_full_async_pipeline()
    print("ALL HARNESS VERIFICATION TESTS PASSED!")
