"""
Scripted offline demo dispatcher (NOT an AI).

This keyword-matching script fakes agent replies and tool calls so the tool
pipeline can be exercised without a GGUF model. It is only used when the
environment variable HARNESS_SCRIPTED_DEMO=1 is set (see test_harness.py).
The live harness never falls back to it.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _build_blender_script_for_asset(res_glb_path: str) -> str:
    """Return a rich procedural Blender 4.x bpy script tailored to the requested .glb model."""
    stem = Path(res_glb_path).stem.lower()
    if "paddle" in stem:
        return (
            "import bpy\n"
            "# Beveled sci-fi energy paddle body\n"
            "bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.25))\n"
            "paddle = bpy.context.active_object\n"
            "paddle.name = 'PaddleBody'\n"
            "paddle.scale = (2.8, 0.55, 0.42)\n"
            "bev = paddle.modifiers.new(name='Bevel', type='BEVEL')\n"
            "bev.width = 0.08\n"
            "bev.segments = 3\n"
            "mat_body = bpy.data.materials.new(name='PaddleEnergyMat')\n"
            "mat_body.use_nodes = True\n"
            "bsdf = mat_body.node_tree.nodes.get('Principled BSDF')\n"
            "if bsdf:\n"
            "    bsdf.inputs['Base Color'].default_value = (0.12, 0.72, 0.98, 1.0)\n"
            "    bsdf.inputs['Metallic'].default_value = 0.85\n"
            "    bsdf.inputs['Roughness'].default_value = 0.22\n"
            "paddle.data.materials.append(mat_body)\n"
            "# Glowing side thrusters\n"
            "for side, x_pos in (('L', -1.55), ('R', 1.55)):\n"
            "    bpy.ops.mesh.primitive_cylinder_add(radius=0.24, depth=0.65, location=(x_pos, 0, 0.25))\n"
            "    thr = bpy.context.active_object\n"
            "    thr.name = f'PaddleThruster_{side}'\n"
            "    thr.data.materials.append(mat_body)\n"
            "# Clean -col collision mesh\n"
            "bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.25))\n"
            "col = bpy.context.active_object\n"
            "col.name = 'Paddle-col'\n"
            "col.scale = (3.3, 0.65, 0.45)\n"
        )
    if "brick" in stem:
        return (
            "import bpy, os\n"
            "# Chamfered cyber-crystal brick UV-mapped to res://assets/textures/brick_albedo.png\n"
            "bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.3))\n"
            "brick = bpy.context.active_object\n"
            "brick.name = 'CyberBrick'\n"
            "brick.scale = (1.6, 0.75, 0.55)\n"
            "bev = brick.modifiers.new(name='Bevel', type='BEVEL')\n"
            "bev.width = 0.09\n"
            "bev.segments = 2\n"
            "mat = bpy.data.materials.new(name='CyberCrystalBrickMat')\n"
            "mat.use_nodes = True\n"
            "nodes = mat.node_tree.nodes\n"
            "links = mat.node_tree.links\n"
            "bsdf = nodes.get('Principled BSDF')\n"
            "tex_node = nodes.new('ShaderNodeTexImage')\n"
            "tex_path = 'res://assets/textures/brick_albedo.png'\n"
            "if os.path.exists(tex_path):\n"
            "    tex_node.image = bpy.data.images.load(tex_path)\n"
            "    if bsdf:\n"
            "        links.new(bsdf.inputs['Base Color'], tex_node.outputs['Color'])\n"
            "brick.data.materials.append(mat)\n"
            "bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.3))\n"
            "col = bpy.context.active_object\n"
            "col.name = 'CyberBrick-col'\n"
            "col.scale = (1.62, 0.77, 0.57)\n"
        )
    if "arena" in stem or "frame" in stem or "border" in stem or "wall" in stem:
        return (
            "import bpy\n"
            "# Sleek U-shaped 3D containment border (Left, Right, and Top rails) with -col static collision\n"
            "mat = bpy.data.materials.new(name='ArenaFrameMat')\n"
            "mat.use_nodes = True\n"
            "bsdf = mat.node_tree.nodes.get('Principled BSDF')\n"
            "if bsdf:\n"
            "    bsdf.inputs['Base Color'].default_value = (0.09, 0.16, 0.28, 1.0)\n"
            "    bsdf.inputs['Metallic'].default_value = 0.9\n"
            "    bsdf.inputs['Roughness'].default_value = 0.25\n"
            "rails = [\n"
            "    ('ArenaRailLeft',  (-10.5, 0.0, 0.6), (0.8, 22.0, 1.2)),\n"
            "    ('ArenaRailRight', ( 10.5, 0.0, 0.6), (0.8, 22.0, 1.2)),\n"
            "    ('ArenaRailTop',   (  0.0, -10.6, 0.6), (21.8, 0.8, 1.2)),\n"
            "]\n"
            "for r_name, loc, scl in rails:\n"
            "    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)\n"
            "    vis = bpy.context.active_object\n"
            "    vis.name = r_name\n"
            "    vis.scale = scl\n"
            "    bev = vis.modifiers.new(name='Bevel', type='BEVEL')\n"
            "    bev.width = 0.07\n"
            "    bev.segments = 2\n"
            "    vis.data.materials.append(mat)\n"
            "    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)\n"
            "    col = bpy.context.active_object\n"
            "    col.name = f'{r_name}-col'\n"
            "    col.scale = scl\n"
        )
    title = Path(res_glb_path).stem.title().replace("_", "")
    return (
        "import bpy\n"
        f"bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0, 0, 0.6))\n"
        f"main_obj = bpy.context.active_object\n"
        f"main_obj.name = '{title}'\n"
        f"bev = main_obj.modifiers.new(name='Bevel', type='BEVEL')\n"
        f"bev.width = 0.08\n"
        f"bev.segments = 2\n"
        f"bpy.ops.mesh.primitive_cube_add(size=1.22, location=(0, 0, 0.6))\n"
        f"col_obj = bpy.context.active_object\n"
        f"col_obj.name = '{title}-col'\n"
    )


def _build_godot_code_file_content(res_path: str, user_prompt: str) -> str:
    """Generate complete, modular, playable Godot 4.3 scenes (.tscn), GDScript (.gd), or project.godot."""
    lower = res_path.lower()
    if lower.endswith("project.godot"):
        return (
            "; Engine configuration file.\n"
            "; Configured by the Local Godot 4 Agentic Harness.\n\n"
            "config_version=5\n\n"
            "[application]\n\n"
            'config/name="Nebula Breaker 3D"\n'
            'run/main_scene="res://scenes/main.tscn"\n'
            'config/features=PackedStringArray("4.3", "Forward Plus")\n'
            'config/icon="res://assets/ui/powerup_icon.png"\n\n'
            "[display]\n\n"
            "window/size/viewport_width=1280\n"
            "window/size/viewport_height=720\n\n"
            "[input]\n\n"
            "move_left={\n"
            '"deadzone": 0.2,\n'
            '"events": []\n'
            "}\n"
            "move_right={\n"
            '"deadzone": 0.2,\n'
            '"events": []\n'
            "}\n"
            "launch_ball={\n"
            '"deadzone": 0.2,\n'
            '"events": []\n'
            "}\n"
        )

    if lower.endswith("paddle.gd"):
        return (
            "extends CharacterBody3D\n"
            "class_name NebulaPaddle3D\n\n"
            "signal ball_deflected(hit_offset: float)\n"
            "signal powerup_collected(powerup_type: String)\n\n"
            "@export var move_speed: float = 16.0\n"
            "@export var arena_half_width: float = 8.4\n"
            "@export var base_width: float = 3.0\n"
            "@export var use_mouse_control: bool = false\n\n"
            "var width_multiplier: float = 1.0\n"
            "var wide_timer: float = 0.0\n\n"
            "func _ready() -> void:\n"
            "\tadd_to_group(\"paddle\")\n\n"
            "func _physics_process(delta: float) -> void:\n"
            "\tif wide_timer > 0.0:\n"
            "\t\twide_timer -= delta\n"
            "\t\tif wide_timer <= 0.0:\n"
            "\t\t\tset_wide_paddle(false)\n\n"
            "\tvar dir: float = 0.0\n"
            "\tif Input.is_Key_pressed(KEY_A) or Input.is_Key_pressed(KEY_LEFT) or Input.is_action_pressed(\"ui_left\"):\n"
            "\t\tdir -= 1.0\n"
            "\tif Input.is_Key_pressed(KEY_D) or Input.is_Key_pressed(KEY_RIGHT) or Input.is_action_pressed(\"ui_right\"):\n"
            "\t\tdir += 1.0\n\n"
            "\tvelocity.x = dir * move_speed\n"
            "\tvelocity.y = 0.0\n"
            "\tvelocity.z = 0.0\n"
            "\tmove_and_slide()\n"
            "\tvar limit: float = maxf(2.0, arena_half_width - (base_width * width_multiplier * 0.5))\n"
            "\tglobal_position.x = clampf(global_position.x, -limit, limit)\n"
            "\tglobal_position.y = 0.3\n"
            "\tglobal_position.z = 8.5\n\n"
            "func get_deflection_direction(hit_world_x: float) -> Vector3:\n"
            "\tvar half_w: float = maxf(0.5, (base_width * width_multiplier) * 0.5)\n"
            "\tvar rel: float = clampf((hit_world_x - global_position.x) / half_w, -1.0, 1.0)\n"
            "\tvar max_angle: float = deg_to_rad(62.0)\n"
            "\tvar angle: float = rel * max_angle\n"
            "\tvar dir := Vector3(sin(angle), 0.0, -cos(angle)).normalized()\n"
            "\tball_deflected.emit(rel)\n"
            "\treturn dir\n\n"
            "func set_wide_paddle(enabled: bool, duration: float = 10.0) -> void:\n"
            "\twidth_multiplier = 1.55 if enabled else 1.0\n"
            "\twide_timer = duration if enabled else 0.0\n"
            "\tscale.x = width_multiplier\n"
        ).replace("is_Key_pressed", "is_key_pressed")

    if lower.endswith("ball.gd"):
        return (
            "extends CharacterBody3D\n"
            "class_name NebulaBall3D\n\n"
            "signal brick_hit(brick_node: Node, impact_pos: Vector3)\n"
            "signal paddle_bounced(impact_pos: Vector3)\n"
            "signal ball_lost(ball_node: Node)\n\n"
            "@export var base_speed: float = 13.5\n"
            "@export var max_speed: float = 25.0\n"
            "@export var speed_ramp_per_combo: float = 0.45\n"
            "@export var arena_bounds_x: float = 9.6\n"
            "@export var arena_top_z: float = -9.8\n"
            "@export var arena_bottom_z: float = 11.2\n\n"
            "var current_speed: float = 13.5\n"
            "var move_dir: Vector3 = Vector3(0.25, 0.0, -1.0).normalized()\n"
            "var is_launched: bool = false\n"
            "var plasma_piercer_active: bool = false\n"
            "var plasma_timer: float = 0.0\n"
            "var combo_count: int = 0\n\n"
            "func _ready() -> void:\n"
            "\tadd_to_group(\"balls\")\n"
            "\tcurrent_speed = base_speed\n\n"
            "func launch(initial_dir: Vector3 = Vector3(0.2, 0.0, -1.0)) -> void:\n"
            "\tmove_dir = Vector3(initial_dir.x, 0.0, initial_dir.z).normalized()\n"
            "\tis_launched = true\n\n"
            "func activate_plasma_piercer(duration: float = 6.0) -> void:\n"
            "\tplasma_piercer_active = true\n"
            "\tplasma_timer = duration\n\n"
            "func _physics_process(delta: float) -> void:\n"
            "\tif not is_launched:\n"
            "\t\treturn\n\n"
            "\tif plasma_piercer_active:\n"
            "\t\tplasma_timer -= delta\n"
            "\t\tif plasma_timer <= 0.0:\n"
            "\t\t\tplasma_piercer_active = false\n\n"
            "\tvar step_vec: Vector3 = move_dir * current_speed * delta\n"
            "\tvar collision: KinematicCollision3D = move_and_collide(step_vec)\n"
            "\tif collision:\n"
            "\t\tvar collider: Object = collision.get_collider()\n"
            "\t\tvar hit_pos: Vector3 = collision.get_position()\n"
            "\t\tvar normal: Vector3 = collision.get_normal()\n"
            "\t\tnormal.y = 0.0\n"
            "\t\tif normal.length_squared() < 0.001:\n"
            "\t\t\tnormal = Vector3(0.0, 0.0, 1.0)\n"
            "\t\telse:\n"
            "\t\t\tnormal = normal.normalized()\n\n"
            "\t\tif collider and collider.has_method(\"get_deflection_direction\"):\n"
            "\t\t\tmove_dir = collider.get_deflection_direction(hit_pos.x)\n"
            "\t\t\tcombo_count = 0\n"
            "\t\t\tpaddle_bounced.emit(hit_pos)\n"
            "\t\telif collider and collider.has_method(\"apply_ball_damage\"):\n"
            "\t\t\tcombo_count += 1\n"
            "\t\t\tcurrent_speed = minf(max_speed, base_speed + float(combo_count) * speed_ramp_per_combo)\n"
            "\t\t\tcollider.apply_ball_damage(2 if plasma_piercer_active else 1, hit_pos)\n"
            "\t\t\tbrick_hit.emit(collider, hit_pos)\n"
            "\t\t\tif not plasma_piercer_active:\n"
            "\t\t\t\tmove_dir = move_dir.bounce(normal).normalized()\n"
            "\t\telse:\n"
            "\t\t\tmove_dir = move_dir.bounce(normal).normalized()\n\n"
            "\t# Clamp to 3D containment rails\n"
            "\tif global_position.x <= -arena_bounds_x and move_dir.x < 0.0:\n"
            "\t\tmove_dir.x = -move_dir.x\n"
            "\telif global_position.x >= arena_bounds_x and move_dir.x > 0.0:\n"
            "\t\tmove_dir.x = -move_dir.x\n"
            "\tif global_position.z <= arena_top_z and move_dir.z < 0.0:\n"
            "\t\tmove_dir.z = -move_dir.z\n"
            "\tif absf(move_dir.z) < 0.18:\n"
            "\t\tmove_dir.z = -0.25 if move_dir.z <= 0.0 else 0.25\n"
            "\t\tmove_dir = move_dir.normalized()\n"
            "\tglobal_position.y = 0.35\n\n"
            "\tif global_position.z > arena_bottom_z:\n"
            "\t\tis_launched = false\n"
            "\t\tball_lost.emit(self)\n"
        )

    if lower.endswith("brick.gd"):
        return (
            "extends StaticBody3D\n"
            "class_name NebulaBrick3D\n\n"
            "signal brick_destroyed(brick: Node3D, tier: int, score_value: int, world_pos: Vector3)\n"
            "signal brick_cracked(brick: Node3D, world_pos: Vector3)\n\n"
            "enum BrickTier {\n"
            "\tSTANDARD_NEON = 1,\n"
            "\tARMORED_REACTOR = 2,\n"
            "\tEXPLOSIVE_CORE = 3,\n"
            "}\n\n"
            "@export var brick_tier: int = BrickTier.STANDARD_NEON\n"
            "@export var hit_points: int = 1\n"
            "@export var score_value: int = 100\n"
            "@export var explosion_radius: float = 3.2\n"
            "@export var powerup_drop_chance: float = 0.28\n\n"
            "var _is_destroyed: bool = false\n\n"
            "func _ready() -> void:\n"
            "\tadd_to_group(\"bricks\")\n"
            "\tif brick_tier == BrickTier.ARMORED_REACTOR:\n"
            "\t\thit_points = max(2, hit_points)\n"
            "\t\tscore_value = 250\n"
            "\telif brick_tier == BrickTier.EXPLOSIVE_CORE:\n"
            "\t\thit_points = 1\n"
            "\t\tscore_value = 350\n\n"
            "func apply_ball_damage(amount: int, impact_pos: Vector3 = Vector3.ZERO) -> void:\n"
            "\tif _is_destroyed:\n"
            "\t\treturn\n"
            "\thit_points -= amount\n"
            "\tif hit_points > 0:\n"
            "\t\t_shift_cracked_appearance()\n"
            "\t\tbrick_cracked.emit(self, global_position)\n"
            "\telse:\n"
            "\t\t_destroy_brick()\n\n"
            "func _shift_cracked_appearance() -> void:\n"
            "\tscale = scale * 0.94\n\n"
            "func _destroy_brick() -> void:\n"
            "\tif _is_destroyed:\n"
            "\t\treturn\n"
            "\t_is_destroyed = true\n"
            "\tvar pos: Vector3 = global_position\n"
            "\tif brick_tier == BrickTier.EXPLOSIVE_CORE:\n"
            "\t\t_detonate_adjacent_bricks(pos)\n"
            "\tbrick_destroyed.emit(self, brick_tier, score_value, pos)\n"
            "\tqueue_free()\n\n"
            "func _detonate_adjacent_bricks(origin: Vector3) -> void:\n"
            "\tfor node in get_tree().get_nodes_in_group(\"bricks\"):\n"
            "\t\tif node != self and node is Node3D and is_instance_valid(node):\n"
            "\t\t\tvar other := node as Node3D\n"
            "\t\t\tif other.global_position.distance_to(origin) <= explosion_radius:\n"
            "\t\t\t\tif other.has_method(\"apply_ball_damage\"):\n"
            "\t\t\t\t\tother.apply_ball_damage(5, origin)\n"
        )

    if lower.endswith("powerup.gd"):
        return (
            "extends Area3D\n"
            "class_name NebulaPowerUp3D\n\n"
            "signal activated(powerup_type: String)\n\n"
            "# Types: 'multi_ball', 'wide_paddle', 'plasma_piercer'\n"
            "@export var powerup_type: String = \"multi_ball\"\n"
            "@export var fall_speed: float = 5.5\n"
            "@export var spin_speed: float = 2.8\n\n"
            "func _ready() -> void:\n"
            "\tadd_to_group(\"powerups\")\n"
            "\tbody_entered.connect(_on_body_entered)\n\n"
            "func _physics_process(delta: float) -> void:\n"
            "\tglobal_position.z += fall_speed * delta\n"
            "\trotate_y(spin_speed * delta)\n"
            "\tif global_position.z > 11.5:\n"
            "\t\tqueue_free()\n\n"
            "func _on_body_entered(body: Node3D) -> void:\n"
            "\tif body and body.is_in_group(\"paddle\"):\n"
            "\t\tactivated.emit(powerup_type)\n"
            "\t\tqueue_free()\n"
        )

    if lower.endswith("hud.tscn"):
        return (
            '[gd_scene load_steps=2 format=3 uid="uid://nebula_hud_scene"]\n\n'
            '[ext_resource type="Texture2D" path="res://assets/ui/powerup_icon.png" id="1_icon"]\n\n'
            '[node name="HUD" type="CanvasLayer"]\n\n'
            '[node name="TopBar" type="HBoxContainer" parent="."]\n'
            "offset_left = 24.0\n"
            "offset_top = 16.0\n"
            "offset_right = 1256.0\n"
            "offset_bottom = 64.0\n\n"
            '[node name="PowerUpIcon" type="TextureRect" parent="TopBar"]\n'
            "custom_minimum_size = Vector2(36, 36)\n"
            'texture = ExtResource("1_icon")\n'
            "expand_mode = 1\n\n"
            '[node name="ScoreLabel" type="Label" parent="TopBar"]\n'
            'text = "SCORE: 0000"\n\n'
            '[node name="LivesLabel" type="Label" parent="TopBar"]\n'
            'text = "   LIVES: 3"\n\n'
            '[node name="LevelLabel" type="Label" parent="TopBar"]\n'
            'text = "   LEVEL: 1"\n\n'
            '[node name="ComboLabel" type="Label" parent="TopBar"]\n'
            'text = "   COMBO: 1x"\n\n'
            '[node name="OverlayMessage" type="Label" parent="."]\n'
            "offset_left = 340.0\n"
            "offset_top = 320.0\n"
            "offset_right = 940.0\n"
            "offset_bottom = 400.0\n"
            'text = "NEBULA BREAKER 3D — PRESS SPACE / CLICK TO LAUNCH"\n'
        )

    if lower.endswith("main.tscn"):
        return (
            '[gd_scene load_steps=16 format=3 uid="uid://nebula_main_3d"]\n\n'
            '[ext_resource type="Material" path="res://assets/textures/arena_floor_material.tres" id="1_floor_mat"]\n'
            '[ext_resource type="PackedScene" path="res://assets/models/paddle.glb" id="2_paddle_glb"]\n'
            '[ext_resource type="PackedScene" path="res://assets/models/brick.glb" id="3_brick_glb"]\n'
            '[ext_resource type="PackedScene" path="res://assets/models/arena_frame.glb" id="4_frame_glb"]\n'
            '[ext_resource type="AudioStream" path="res://assets/audio/sfx_paddle_hit.wav" id="5_sfx_paddle"]\n'
            '[ext_resource type="AudioStream" path="res://assets/audio/sfx_brick_break.wav" id="6_sfx_brick"]\n'
            '[ext_resource type="AudioStream" path="res://assets/audio/bgm_synthwave.ogg" id="7_bgm"]\n'
            '[ext_resource type="Script" path="res://scripts/paddle.gd" id="8_paddle_gd"]\n'
            '[ext_resource type="Script" path="res://scripts/ball.gd" id="9_ball_gd"]\n'
            '[ext_resource type="Script" path="res://scripts/brick.gd" id="10_brick_gd"]\n'
            '[ext_resource type="PackedScene" path="res://scenes/hud.tscn" id="11_hud"]\n\n'
            '[sub_resource type="Environment" id="Environment_nebula"]\n'
            "background_mode = 1\n"
            "background_color = Color(0.03, 0.04, 0.09, 1)\n"
            "ambient_light_source = 2\n"
            "ambient_light_color = Color(0.22, 0.34, 0.55, 1)\n"
            "ambient_light_energy = 0.85\n"
            "tonemap_mode = 2\n"
            "glow_enabled = true\n"
            "glow_intensity = 0.9\n"
            "glow_bloom = 0.25\n\n"
            '[sub_resource type="PlaneMesh" id="PlaneMesh_floor"]\n'
            "size = Vector2(22, 22)\n\n"
            '[sub_resource type="BoxShape3D" id="BoxShape3D_paddle"]\n'
            "size = Vector3(3.0, 0.6, 0.6)\n\n"
            '[sub_resource type="SphereShape3D" id="SphereShape3D_ball"]\n'
            "radius = 0.32\n\n"
            '[sub_resource type="SphereMesh" id="SphereMesh_ball"]\n'
            "radius = 0.32\n"
            "height = 0.64\n\n"
            '[node name="NebulaBreakerMain" type="Node3D"]\n\n'
            '[node name="WorldEnvironment" type="WorldEnvironment" parent="."]\n'
            'environment = SubResource("Environment_nebula")\n\n'
            '[node name="DirectionalLight3D" type="DirectionalLight3D" parent="."]\n'
            "transform = Transform3D(0.866025, -0.383022, 0.321394, 0, 0.642788, 0.766044, -0.5, -0.663414, 0.55667, 0, 14, 8)\n"
            "light_energy = 1.6\n"
            "shadow_enabled = true\n\n"
            '[node name="Camera3D" type="Camera3D" parent="."]\n'
            "transform = Transform3D(1, 0, 0, 0, 0.422618, 0.906308, 0, -0.906308, 0.422618, 0, 16.5, 9.2)\n"
            "current = true\n"
            "fov = 60.0\n\n"
            '[node name="ArenaFloor" type="MeshInstance3D" parent="."]\n'
            'mesh = SubResource("PlaneMesh_floor")\n'
            'surface_material_override/0 = ExtResource("1_floor_mat")\n\n'
            '[node name="ArenaBorderFrame" type="StaticBody3D" parent="."]\n\n'
            '[node name="ArenaFrameModel" parent="ArenaBorderFrame" instance=ExtResource("4_frame_glb")]\n\n'
            '[node name="Paddle" type="CharacterBody3D" parent="." groups=["paddle"]]\n'
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.3, 8.5)\n"
            'script = ExtResource("8_paddle_gd")\n\n'
            '[node name="PaddleCollider" type="CollisionShape3D" parent="Paddle"]\n'
            'shape = SubResource("BoxShape3D_paddle")\n\n'
            '[node name="PaddleMesh" parent="Paddle" instance=ExtResource("2_paddle_glb")]\n\n'
            '[node name="Ball" type="CharacterBody3D" parent="." groups=["balls"]]\n'
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.35, 7.4)\n"
            'script = ExtResource("9_ball_gd")\n\n'
            '[node name="BallCollider" type="CollisionShape3D" parent="Ball"]\n'
            'shape = SubResource("SphereShape3D_ball")\n\n'
            '[node name="BallMesh" type="MeshInstance3D" parent="Ball"]\n'
            'mesh = SubResource("SphereMesh_ball")\n\n'
            '[node name="BrickGrid" type="Node3D" parent="."]\n\n'
            '[node name="BrickStandard_1" type="StaticBody3D" parent="BrickGrid" groups=["bricks"]]\n'
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -4.0, 0.35, -4.5)\n"
            'script = ExtResource("10_brick_gd")\n'
            "brick_tier = 1\n\n"
            '[node name="BrickModel1" parent="BrickGrid/BrickStandard_1" instance=ExtResource("3_brick_glb")]\n\n'
            '[node name="BrickArmored_2" type="StaticBody3D" parent="BrickGrid" groups=["bricks"]]\n'
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0.0, 0.35, -4.5)\n"
            'script = ExtResource("10_brick_gd")\n'
            "brick_tier = 2\n"
            "hit_points = 2\n\n"
            '[node name="BrickModel2" parent="BrickGrid/BrickArmored_2" instance=ExtResource("3_brick_glb")]\n\n'
            '[node name="BrickExplosive_3" type="StaticBody3D" parent="BrickGrid" groups=["bricks"]]\n'
            "transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 4.0, 0.35, -4.5)\n"
            'script = ExtResource("10_brick_gd")\n'
            "brick_tier = 3\n\n"
            '[node name="BrickModel3" parent="BrickGrid/BrickExplosive_3" instance=ExtResource("3_brick_glb")]\n\n'
            '[node name="AudioPlayers" type="Node" parent="."]\n\n'
            '[node name="SfxPaddleHit" type="AudioStreamPlayer3D" parent="AudioPlayers"]\n'
            'stream = ExtResource("5_sfx_paddle")\n\n'
            '[node name="SfxBrickBreak" type="AudioStreamPlayer3D" parent="AudioPlayers"]\n'
            'stream = ExtResource("6_sfx_brick")\n\n'
            '[node name="BgmSynthwave" type="AudioStreamPlayer" parent="AudioPlayers"]\n'
            'stream = ExtResource("7_bgm")\n'
            "autoplay = true\n\n"
            '[node name="HUD" parent="." instance=ExtResource("11_hud")]\n'
        )

    if lower.endswith(".gd"):
        cls_name = Path(res_path).stem.title().replace("_", "")
        return (
            "extends Node3D\n"
            f"class_name {cls_name}\n\n"
            "func _ready() -> void:\n"
            "\tpass\n"
        )

    return (
        '[gd_scene format=3 uid="uid://generated_scene"]\n\n'
        '[node name="RootNode3D" type="Node3D"]\n'
    )


def _extract_code_paths_from_prompt(user_prompt: str, res_paths: List[str]) -> List[str]:
    """Extract both explicit res:// code files and bare .tscn / .gd / project.godot filenames from the prompt."""
    found: List[str] = [
        p for p in res_paths if p.lower().endswith((".tscn", ".gd", ".godot"))
    ]
    if "project.godot" in user_prompt.lower() and "res://project.godot" not in found:
        found.insert(0, "res://project.godot")

    bare_matches = re.findall(r"\b([A-Za-z0-9_]+\.(?:tscn|gd))\b", user_prompt)
    for bm in bare_matches:
        bm_low = bm.lower()
        if any(p.lower().endswith("/" + bm_low) or p.lower() == "res://" + bm_low for p in found):
            continue
        if bm_low.endswith(".tscn"):
            found.append(f"res://scenes/{bm}")
        elif bm_low.endswith(".gd"):
            found.append(f"res://scripts/{bm}")

    # Order: project.godot first, then .gd scripts, then hud.tscn, then main.tscn (so ExtResource targets exist first!)
    def _sort_key(p: str) -> int:
        pl = p.lower()
        if pl.endswith("project.godot"):
            return 0
        if pl.endswith(".gd"):
            return 1
        if "hud" in pl and pl.endswith(".tscn"):
            return 2
        return 3

    return sorted(list(dict.fromkeys(found)), key=_sort_key)


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
    code_paths = _extract_code_paths_from_prompt(user_prompt, res_paths)

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
    done_code = {
        t["arguments"].get("path")
        for t in executed_tools
        if t["name"] == "write_file"
    }
    done_validation = any(t["name"] == "run_command" for t in executed_tools)

    # Phase 1: 2D Textures & UI Icons
    pending_2d = [p for p in png_paths if p not in done_2d]
    if pending_2d:
        t_calls = []
        for p in pending_2d:
            is_icon = "icon" in p.lower() or "ui" in p.lower() or "orb" in p.lower()
            is_pbr = "floor" in p.lower() or "wall" in p.lower() or "brick" in p.lower() or "pbr" in user_prompt.lower()
            a_type = "ui_icon" if is_icon else "seamless_texture"
            t_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": "generate_2d_asset",
                        "arguments": json.dumps(
                            {
                                "prompt": f"Game asset for {p}: {user_prompt[:220]}",
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
            bpy_script = _build_blender_script_for_asset(p)
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
            dur = 24.0 if is_music else (1.5 if "break" in p.lower() else 1.0)
            t_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": "generate_game_audio",
                        "arguments": json.dumps(
                            {
                                "prompt": f"Game audio for {p}: {'energetic instrumental retro-futuristic arcade synthwave, no vocals' if is_music else 'punchy sci-fi arcade impact sfx'}",
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

    # Phase 4: Assemble Godot 4 project.godot, GDScript (.gd), and Scenes (.tscn)
    pending_code = [p for p in code_paths if p not in done_code]
    if pending_code:
        t_calls = []
        for p in pending_code:
            file_content = _build_godot_code_file_content(p, user_prompt)
            t_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": p,
                                "content": file_content,
                            }
                        ),
                    },
                }
            )
        thinking = (
            f"<think>\n"
            f"Inspected Phase 3 Waveform + Mel-Spectrogram PNGs. Proceeding to Phase 4 (Scene & GDScript Assembly): "
            f"Writing {', '.join(pending_code)} with modular Godot 4 nodes, physics, 3-tier bricks, power-ups, and HUD.\n"
            f"</think>"
        )
        return f"{thinking}\n**Phase 4:** Writing Godot 4 configuration, scripts, and scenes (`{', '.join(pending_code)}`).", t_calls

    # Phase 5: Headless Godot Validation
    if not done_validation:
        thinking = (
            "<think>\n"
            "All 2D textures, 3D `.glb` models, 44.1kHz stereo audio tracks, and Godot 4 scenes/scripts are assembled. "
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
        return f"{thinking}\n**Phase 5:** Running `godot --headless --quit` project-wide validation.", t_calls

    # All phases complete
    thinking = (
        "<think>\n"
        "All phases (2D textures/icons, 3D Blender `.glb` models, 2-in-1 SFX/Music audio tracks, Godot 4 scenes/scripts, and headless Godot validation) "
        "completed with 0 errors.\n"
        "</think>"
    )
    return f"{thinking}\nAll multi-phase game assets, scenes (`main.tscn`, `hud.tscn`), scripts (`paddle.gd`, `ball.gd`, `brick.gd`, `powerup.gd`), and headless Godot 4 validations completed with **0 errors**.", []


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
