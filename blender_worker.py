"""
3D Headless Asset Engine (blender.exe -b --python ...).
Implements Tier 4 Headless Blender 3D Pipeline, Godot 4 Export Harness Wrapper,
automatic transform application, metric unit enforcement, Z-up to glTF +Y-up conversion,
Godot collision suffix preservation (-col, -colonly, -convcol, -noimp),
source .blend archiving in blend_src/, and 3-Angle Visual Verification Contact Sheet rendering.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from vram_orchestrator import VRAMOrchestrator

logger = logging.getLogger("CustomHarness.BlenderWorker")

GODOT_COLLISION_SUFFIXES = (
    "-col",
    "-colonly",
    "-convcol",
    "-convcolonly",
    "-noimp",
    "-navmesh",
    "-occ",
    "-rigid",
    "-vehicle",
    "-wheel",
)


GODOT_BLENDER_HARNESS_WRAPPER = r'''# -*- coding: utf-8 -*-
"""
Auto-generated Godot 4 Blender Export Harness Wrapper.
Executes agent bpy script, enforces metric 1.0 scale, applies all transforms,
preserves Godot 4 import suffixes (-col, -colonly, -convcol, -noimp),
saves blend_src/<name>.blend, exports +Y-up binary .glb, and renders a
3-Angle Visual Verification Contact Sheet (Front -Y, 3/4 Isometric, Wireframe/Top).
"""
import bpy
import bmesh
import json
import math
import mathutils
import os
import sys
import traceback

OUTPUT_GLB_PATH = __OUTPUT_GLB_PATH__
OUTPUT_BLEND_PATH = __OUTPUT_BLEND_PATH__
CONTACT_SHEET_DIR = __CONTACT_SHEET_DIR__
RENDER_PREVIEW = __RENDER_PREVIEW__
HEAVY_CYCLES_BAKE = __HEAVY_CYCLES_BAKE__

# 1. Clean Factory Scene & Enforce Metric Units (1.0m = 1 Godot Unit)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0
scene.unit_settings.length_unit = 'METERS'

# 1b. Helpers available to the agent script (correct for Blender 4.x and 5.x)
def make_material(name, color, roughness=0.8, metallic=0.0):
    """Solid-colour PBR material that shows in the preview AND exports to glTF. color = (r, g, b) or (r, g, b, a), 0..1."""
    c = tuple(float(v) for v in color)
    c = c + (1.0,) if len(c) == 3 else c[:4]
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    mat.diffuse_color = c
    bsdf = None
    if mat.node_tree:
        bsdf = next((n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled'), None)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = c
        bsdf.inputs['Roughness'].default_value = float(roughness)
        bsdf.inputs['Metallic'].default_value = float(metallic)
    return mat


def _finish_object(name, material, rotation):
    obj = bpy.context.active_object
    obj.name = name
    if rotation:
        obj.rotation_euler = tuple(math.radians(a) for a in rotation)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_box(name, size, location=(0, 0, 0), material=None, rotation=None):
    """Box with size = (x, y, z) in metres centred at location. rotation in degrees."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(location))
    bpy.context.active_object.scale = tuple(size)
    return _finish_object(name, material, rotation)


def add_cylinder(name, radius, depth, location=(0, 0, 0), material=None, rotation=None, segments=24):
    """Cylinder along Z (rotate with rotation=(90, 0, 0) etc., degrees)."""
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, vertices=segments, location=tuple(location))
    return _finish_object(name, material, rotation)


def add_sphere(name, radius, location=(0, 0, 0), material=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=tuple(location))
    return _finish_object(name, material, None)


# 2. Execute Agent Procedural bpy Script
# ==================== BEGIN AGENT SCRIPT ====================
__AGENT_SCRIPT_CODE__
# ===================== END AGENT SCRIPT =====================

# 3. Post-Process Meshes: Apply Transforms, Recalculate Normals, Ensure UVs, Preserve Godot Suffixes
bpy.ops.object.select_all(action='DESELECT')
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
preserved_suffix_objects = []

for obj in bpy.context.scene.objects:
    lower_name = obj.name.lower()
    for sfx in ("-col", "-colonly", "-convcol", "-convcolonly", "-noimp", "-navmesh", "-occ", "-rigid"):
        if lower_name.endswith(sfx):
            preserved_suffix_objects.append({"name": obj.name, "suffix": sfx})
            break

for obj in mesh_objects:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # Apply non-armature modifiers (BEVEL, BOOLEAN, SOLIDIFY, SUBSURF, MIRROR, ARRAY)
    for mod in list(obj.modifiers):
        if mod.type != 'ARMATURE':
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception:
                pass
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass

    # Recalculate outside normals & auto-create UV layer if missing
    try:
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        if not me.uv_layers:
            me.uv_layers.new(name="UVMap")
    except Exception:
        pass
    obj.select_set(False)

# 4. Save Source .blend into blend_src/
os.makedirs(os.path.dirname(OUTPUT_BLEND_PATH), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND_PATH)

# 5. Export Clean Binary .glb for Godot 4 (+Y Up, Applied Modifiers & Tangents)
os.makedirs(os.path.dirname(OUTPUT_GLB_PATH), exist_ok=True)
export_kwargs = {
    "filepath": OUTPUT_GLB_PATH,
    "export_format": "GLB",
    "export_yup": True,
    "export_apply": True,
    "export_tangents": True,
    "export_materials": "EXPORT",
}
try:
    bpy.ops.export_scene.gltf(**export_kwargs)
except TypeError:
    # Fallback for minor API parameter differences across Blender 4.0 - 4.3
    bpy.ops.export_scene.gltf(filepath=OUTPUT_GLB_PATH, export_format="GLB", export_yup=True)

# 6. Compute Scene Bounding Box & Mesh Telemetry
min_corner = [1e9, 1e9, 1e9]
max_corner = [-1e9, -1e9, -1e9]
total_verts = 0
total_tris = 0

for obj in mesh_objects:
    total_verts += len(obj.data.vertices)
    for poly in obj.data.polygons:
        total_tris += max(1, len(poly.vertices) - 2)
    for corner in obj.bound_box:
        world_pt = obj.matrix_world @ mathutils.Vector(corner)
        for i in range(3):
            min_corner[i] = min(min_corner[i], world_pt[i])
            max_corner[i] = max(max_corner[i], world_pt[i])

if not mesh_objects:
    min_corner = [-1.0, -1.0, -1.0]
    max_corner = [1.0, 1.0, 1.0]

center = [(min_corner[i] + max_corner[i]) * 0.5 for i in range(3)]
dims = [max(0.1, max_corner[i] - min_corner[i]) for i in range(3)]
max_span = max(dims[0], dims[1], dims[2], 1.0)

# 7. Render 3-Angle Visual Verification Views (Front -Y, 3/4 Isometric, Wireframe/Top +Z)
rendered_views = []
if RENDER_PREVIEW:
    os.makedirs(CONTACT_SHEET_DIR, exist_ok=True)
    # Choose EEVEE Next (Blender 4.2+) or EEVEE / Cycles
    available_engines = {e.identifier for e in scene.render.bl_rna.properties['engine'].enum_items}
    if HEAVY_CYCLES_BAKE and 'CYCLES' in available_engines:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 32
    elif 'BLENDER_EEVEE_NEXT' in available_engines:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    elif 'BLENDER_EEVEE' in available_engines:
        scene.render.engine = 'BLENDER_EEVEE'
    else:
        scene.render.engine = 'BLENDER_WORKBENCH'

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = False

    # Studio 3-point lighting
    key_light_data = bpy.data.lights.new(name="HarnessKeyLight", type='SUN')
    key_light_data.energy = 3.5
    key_light_obj = bpy.data.objects.new(name="HarnessKeyLight", object_data=key_light_data)
    scene.collection.objects.link(key_light_obj)
    key_light_obj.rotation_euler = (math.radians(50), math.radians(15), math.radians(-35))

    fill_light_data = bpy.data.lights.new(name="HarnessFillLight", type='SUN')
    fill_light_data.energy = 1.4
    fill_light_obj = bpy.data.objects.new(name="HarnessFillLight", object_data=fill_light_data)
    scene.collection.objects.link(fill_light_obj)
    fill_light_obj.rotation_euler = (math.radians(65), math.radians(-30), math.radians(135))

    cam_data = bpy.data.cameras.new(name="HarnessVerifyCam")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new(name="HarnessVerifyCam", object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    dist = max_span * 2.1
    target_vec = mathutils.Vector(center)

    def aim_camera(pos_tuple):
        cam_obj.location = mathutils.Vector(pos_tuple)
        direction = target_vec - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

    # View 1: Front (-Y looking toward +Y)
    aim_camera((center[0], center[1] - dist, center[2] + max_span * 0.15))
    front_path = os.path.join(CONTACT_SHEET_DIR, "view_front.png")
    scene.render.filepath = front_path
    bpy.ops.render.render(write_still=True)
    rendered_views.append(front_path)

    # View 2: 3/4 Isometric
    iso_d = dist * 0.72
    aim_camera((center[0] + iso_d, center[1] - iso_d, center[2] + iso_d * 0.85))
    iso_path = os.path.join(CONTACT_SHEET_DIR, "view_iso.png")
    scene.render.filepath = iso_path
    bpy.ops.render.render(write_still=True)
    rendered_views.append(iso_path)

    # View 3: Top (+Z) with Wireframe modifier overlay
    wire_mods = []
    for obj in mesh_objects:
        try:
            mod = obj.modifiers.new(name="HarnessVerifyWire", type='WIREFRAME')
            mod.use_replace = False
            mod.thickness = max(0.004, max_span * 0.006)
            wire_mods.append((obj, mod))
        except Exception:
            pass

    aim_camera((center[0], center[1] - 0.001, center[2] + dist))
    top_path = os.path.join(CONTACT_SHEET_DIR, "view_top_wire.png")
    scene.render.filepath = top_path
    bpy.ops.render.render(write_still=True)
    rendered_views.append(top_path)

metadata_out = {
    "glb_path": OUTPUT_GLB_PATH,
    "blend_path": OUTPUT_BLEND_PATH,
    "mesh_count": len(mesh_objects),
    "vertex_count": total_verts,
    "triangle_count": total_tris,
    "dimensions_m": [round(d, 4) for d in dims],
    "preserved_godot_suffixes": preserved_suffix_objects,
    "rendered_views": rendered_views,
}
print("HARNESS_BLENDER_METADATA_JSON:" + json.dumps(metadata_out))
'''


class BlenderWorker:
    """
    Manages Blender 4.x+ auto-detection on Windows and executes agent-authored bpy scripts
    wrapped inside the Godot 4 Export Harness with 3-Angle Visual Verification Contact Sheets.
    """

    def __init__(
        self,
        orchestrator: VRAMOrchestrator,
        workspace_dir: str | Path,
    ) -> None:
        self.orchestrator = orchestrator
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._cached_blender_path: Optional[str] = None
        self._cached_blender_version: Optional[str] = None

    # -------------------------------------------------------------------------
    # Blender 4.x+ Windows Auto-Detection
    # -------------------------------------------------------------------------
    def detect_blender_executable(self) -> Optional[str]:
        """
        Auto-detect installed Blender 4.x+ on Windows:
          1. Environment variable BLENDER_PATH
          2. C:\\Program Files\\Blender Foundation\\Blender 4.*\\blender.exe
          3. Steam Library (C:\\Program Files (x86)\\Steam\\steamapps\\common\\Blender\\blender.exe)
          4. %LOCALAPPDATA%\\Programs\\Blender Foundation\\Blender 4.*\\blender.exe
          5. System PATH (shutil.which("blender"))
        """
        if self._cached_blender_path and Path(self._cached_blender_path).exists():
            return self._cached_blender_path

        env_path = os.environ.get("BLENDER_PATH")
        if env_path and Path(env_path).exists():
            self._cached_blender_path = str(Path(env_path).resolve())
            return self._cached_blender_path

        search_roots = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation",
            Path(r"D:\Program Files\Blender Foundation"),
            Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local"))
            / "Programs"
            / "Blender Foundation",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Steam"
            / "steamapps"
            / "common"
            / "Blender",
            Path(r"D:\SteamLibrary\steamapps\common\Blender"),
            Path(r"C:\SteamLibrary\steamapps\common\Blender"),
        ]

        candidates: List[Path] = []
        for root in search_roots:
            try:
                if not root.exists():
                    continue
                direct_exe = root / "blender.exe"
                if direct_exe.exists():
                    candidates.append(direct_exe)
                for sub in sorted(root.glob("Blender*"), reverse=True):
                    exe = sub / "blender.exe"
                    if exe.exists():
                        candidates.append(exe)
            except Exception:
                continue

        which_blender = shutil.which("blender") or shutil.which("blender.exe")
        if which_blender:
            candidates.append(Path(which_blender))

        # Prefer Blender 5.x or 4.x+ first
        for cand in candidates:
            if "5." in str(cand) or "4." in str(cand):
                self._cached_blender_path = str(cand.resolve())
                return self._cached_blender_path

        if candidates:
            self._cached_blender_path = str(candidates[0].resolve())
            return self._cached_blender_path

        return None

    def get_blender_status(self) -> Dict[str, Any]:
        exe = self.detect_blender_executable()
        if not exe:
            return {
                "installed": False,
                "executable": None,
                "version": None,
                "status_label": "Fallback Mesh Engine (Blender 4.x Not Found)",
            }
        if not self._cached_blender_version:
            try:
                proc = subprocess.run(
                    [exe, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=4.0,
                )
                first_line = (proc.stdout or "").splitlines()[0] if proc.stdout else "Blender 4.x"
                self._cached_blender_version = first_line.strip()
            except Exception:
                self._cached_blender_version = "Blender 4.x"

        return {
            "installed": True,
            "executable": exe,
            "version": self._cached_blender_version,
            "status_label": "Ready",
        }

    def resolve_glb_and_blend_paths(self, output_glb_path: str) -> Tuple[Path, Path]:
        """
        Resolve requested res:// .glb path and compute the companion blend_src/<name>.blend path.
        """
        cleaned = output_glb_path.strip()
        if cleaned.startswith("res://"):
            cleaned = cleaned[len("res://") :]
        cleaned = cleaned.lstrip("/\\")
        if not cleaned:
            cleaned = f"assets/models/generated_mesh_{int(time.time())}.glb"

        glb_abs = (self.workspace_dir / cleaned).resolve()
        if not str(glb_abs).startswith(str(self.workspace_dir)):
            glb_abs = (self.workspace_dir / "assets" / "models" / Path(cleaned).name).resolve()
        if glb_abs.suffix.lower() != ".glb":
            glb_abs = glb_abs.with_suffix(".glb")

        glb_abs.parent.mkdir(parents=True, exist_ok=True)
        blend_src_dir = glb_abs.parent / "blend_src"
        blend_src_dir.mkdir(parents=True, exist_ok=True)
        blend_abs = blend_src_dir / f"{glb_abs.stem}.blend"
        return glb_abs, blend_abs

    def to_res_path(self, abs_path: Path) -> str:
        try:
            rel = abs_path.resolve().relative_to(self.workspace_dir)
            return "res://" + str(rel).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def build_wrapped_blender_script(
        self,
        script_code: str,
        output_glb_path: Path,
        output_blend_path: Path,
        contact_sheet_dir: Path,
        render_preview: bool = True,
        heavy_cycles_bake: bool = False,
    ) -> str:
        """
        Inject the agent's bpy script into the Godot 4 Export Harness Wrapper,
        automatically translating any Godot `res://...` paths inside the script
        to absolute workspace paths so `bpy.data.images.load("res://...")` works natively.
        """
        ws_posix = str(self.workspace_dir).replace("\\", "/").rstrip("/") + "/"
        normalized_script = script_code.replace("res://", ws_posix)

        wrapped = GODOT_BLENDER_HARNESS_WRAPPER
        wrapped = wrapped.replace("__OUTPUT_GLB_PATH__", json.dumps(str(output_glb_path)))
        wrapped = wrapped.replace("__OUTPUT_BLEND_PATH__", json.dumps(str(output_blend_path)))
        wrapped = wrapped.replace("__CONTACT_SHEET_DIR__", json.dumps(str(contact_sheet_dir)))
        wrapped = wrapped.replace("__RENDER_PREVIEW__", "True" if render_preview else "False")
        wrapped = wrapped.replace("__HEAVY_CYCLES_BAKE__", "True" if heavy_cycles_bake else "False")
        wrapped = wrapped.replace("__AGENT_SCRIPT_CODE__", normalized_script)
        return wrapped

    # -------------------------------------------------------------------------
    # Main Tool Execution: run_blender_script
    # -------------------------------------------------------------------------
    async def run_blender_script(
        self,
        script_code: str,
        output_glb_path: str = "res://assets/models/generated_prop.glb",
        render_preview: bool = True,
        heavy_cycles_bake: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a headless Blender 4.x script inside the Godot 4 Export Harness Wrapper:
          - Standard mode (heavy_cycles_bake=False): Runs concurrently with LLM_ACTIVE in CPU/RAM mode.
          - Heavy Cycles Bake mode (heavy_cycles_bake=True): Triggers VRAMOrchestrator swap so Cycles
            gets the full 12GB RTX 3060 GPU, then restores LLM Slot 0.
          - Exports binary .glb (+Y up, applied transforms, preserved -col/-colonly/-convcol/-noimp)
            and saves source .blend in blend_src/.
          - Renders a 3-Angle Visual Verification Contact Sheet (Front -Y, 3/4 Isometric, Wireframe/Top)
            and attaches its base64 PNG data URL for the VLM's vision pipeline.
        """
        start_time = time.monotonic()
        glb_abs, blend_abs = self.resolve_glb_and_blend_paths(output_glb_path)
        contact_sheet_abs = glb_abs.parent / f"{glb_abs.stem}_contact_sheet.png"

        vram_swap_meta: Optional[Dict[str, Any]] = None
        if heavy_cycles_bake:
            vram_swap_meta = {
                "acquire": await self.orchestrator.acquire_vram_for_imagegen(
                    conversation_history=conversation_history
                )
            }

        blender_exe = self.detect_blender_executable()
        temp_dir = Path(tempfile.mkdtemp(prefix="godot_blender_harness_"))
        script_file = temp_dir / "harness_job.py"

        wrapped_code = self.build_wrapped_blender_script(
            script_code=script_code,
            output_glb_path=glb_abs,
            output_blend_path=blend_abs,
            contact_sheet_dir=temp_dir,
            render_preview=render_preview,
            heavy_cycles_bake=heavy_cycles_bake,
        )
        script_file.write_text(wrapped_code, encoding="utf-8")

        # Also save the wrapped script alongside the .blend in blend_src/ for reproducibility
        # (only the agent's own script: the export wrapper around it is harness plumbing, not something to edit)
        saved_script_path = blend_abs.with_suffix(".bpy.py")
        saved_script_path.write_text(script_code, encoding="utf-8")

        stdout_text = ""
        metadata: Dict[str, Any] = {}
        engine_used = "blender-headless-cli"
        scripted_demo = os.environ.get("HARNESS_SCRIPTED_DEMO") == "1"
        blender_available = bool(blender_exe and Path(blender_exe).exists())
        failure_reason: Optional[str] = None

        # Remove any previous export so a failed run can never be mistaken for a fresh model
        for stale in (glb_abs, contact_sheet_abs):
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

        try:
            if not blender_available and not scripted_demo:
                failure_reason = (
                    "Blender was not found on this machine. Install Blender 4.x+ or set BLENDER_PATH, "
                    "then try again."
                )
            if blender_available:
                cmd = [
                    blender_exe,
                    "-b",
                    "--factory-startup",
                    "--python",
                    str(script_file),
                ]
                creationflags = (
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.workspace_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=creationflags,
                )
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=180.0
                )
                stdout_text = (
                    stdout_bytes.decode("utf-8", errors="replace")
                    if stdout_bytes
                    else ""
                )

                for line in stdout_text.splitlines():
                    if "HARNESS_BLENDER_METADATA_JSON:" in line:
                        raw_json = line.split("HARNESS_BLENDER_METADATA_JSON:", 1)[1].strip()
                        try:
                            metadata = json.loads(raw_json)
                        except Exception:
                            pass

                if not glb_abs.exists() or proc.returncode != 0:
                    failure_reason = self._summarize_blender_failure(stdout_text, proc.returncode)

            if failure_reason and not scripted_demo:
                # Real use: report the script error so the model can fix it. No stand-in cube.
                return {
                    "status": "error",
                    "tool": "run_blender_script",
                    "error": failure_reason,
                    "res_glb_path": self.to_res_path(glb_abs),
                    "hint": (
                        "Fix the Python error above and call run_blender_script again. Do not clear the "
                        "scene or export yourself: the harness starts from an empty scene and exports the .glb."
                    ),
                }

            if not glb_abs.exists():
                # Scripted test mode only: procedural stand-in mesh when Blender is unavailable
                metadata = self._fallback_generate_glb_and_views(
                    script_code=script_code,
                    glb_abs=glb_abs,
                    blend_abs=blend_abs,
                    temp_dir=temp_dir,
                    render_preview=render_preview,
                )
                engine_used = "trimesh-gltf-fallback (Blender 4.x not found on host)"

            # Stitch the 3-Angle Visual Verification Contact Sheet
            contact_b64 = None
            if render_preview:
                view_paths = [
                    Path(p)
                    for p in metadata.get("rendered_views", [])
                    if Path(p).exists()
                ]
                contact_b64 = self._compose_3_angle_contact_sheet(
                    view_paths=view_paths,
                    output_contact_path=contact_sheet_abs,
                    asset_name=glb_abs.name,
                    metadata=metadata,
                )

        finally:
            if heavy_cycles_bake and vram_swap_meta is not None:
                vram_swap_meta["restore"] = (
                    await self.orchestrator.restore_llm_after_imagegen()
                )
            shutil.rmtree(temp_dir, ignore_errors=True)

        elapsed_sec = round(time.monotonic() - start_time, 2)
        res_glb = self.to_res_path(glb_abs)
        res_blend = self.to_res_path(blend_abs)
        res_contact = (
            self.to_res_path(contact_sheet_abs) if contact_sheet_abs.exists() else None
        )

        return {
            "status": "success",
            "engine_used": engine_used,
            "ConcurrentWithLLM": not heavy_cycles_bake,
            "heavy_cycles_bake": heavy_cycles_bake,
            "output_glb_path": str(glb_abs),
            "res_glb_path": res_glb,
            "source_blend_path": str(blend_abs),
            "res_blend_path": res_blend,
            "harness_script_path": str(saved_script_path),
            "mesh_telemetry": {
                "mesh_count": metadata.get("mesh_count", 1),
                "vertex_count": metadata.get("vertex_count", 0),
                "triangle_count": metadata.get("triangle_count", 0),
                "dimensions_m": metadata.get("dimensions_m", [1.0, 1.0, 1.0]),
                "preserved_godot_suffixes": metadata.get("preserved_godot_suffixes", []),
                "yup_converted": True,
                "transforms_applied": True,
                "metric_units": 1.0,
            },
            "contact_sheet": {
                "enabled": render_preview,
                "path": str(contact_sheet_abs) if contact_sheet_abs.exists() else None,
                "res_path": res_contact,
                "views": ["Front (-Y)", "3/4 Isometric", "Wireframe / Top (+Z)"],
                "image_url": (
                    f"data:image/png;base64,{contact_b64}" if contact_b64 else None
                ),
                "description": (
                    f"3-Angle Visual Verification Contact Sheet for {res_glb} "
                    "(Front -Y, 3/4 Isometric, and Wireframe/Top view). "
                    "Inspect 3D proportions, normals, and materials before importing into Godot."
                ),
            },
            "vram_swap_telemetry": vram_swap_meta,
            "elapsed_seconds": elapsed_sec,
            "blender_logs_tail": stdout_text[-800:] if stdout_text else "",
        }

    @staticmethod
    def _summarize_blender_failure(stdout_text: str, returncode: Optional[int]) -> str:
        """Pull the Python traceback (or last lines) out of Blender's log for the model to act on."""
        lines = [ln.rstrip() for ln in (stdout_text or "").splitlines() if ln.strip()]
        start = None
        for i, ln in enumerate(lines):
            if ln.startswith("Traceback (most recent call last)") or "SyntaxError" in ln or ln.lstrip().startswith("File \""):
                start = i
                break
        excerpt = lines[start : start + 25] if start is not None else lines[-15:]
        # Line numbers refer to the wrapped harness file; the agent script starts ~80 lines in
        return (
            f"The Blender script failed (exit code {returncode}) and no .glb was exported.\n"
            + "\n".join(excerpt)
        )

    # -------------------------------------------------------------------------
    # 3-Angle Visual Verification Contact Sheet Composer
    # -------------------------------------------------------------------------
    @classmethod
    def _compose_3_angle_contact_sheet(
        cls,
        view_paths: List[Path],
        output_contact_path: Path,
        asset_name: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Combine the 3 rendered views (Front -Y, 3/4 Isometric, Wireframe/Top +Z)
        into a labeled 1536x560 composite PNG and return its base64 encoding.
        """
        panel_w, panel_h = 512, 512
        header_h = 48
        canvas_w = panel_w * 3
        canvas_h = panel_h + header_h

        sheet = Image.new("RGB", (canvas_w, canvas_h), (16, 19, 26))
        draw = ImageDraw.Draw(sheet)

        # Header banner
        draw.rectangle([0, 0, canvas_w, header_h], fill=(24, 29, 40))
        dims = metadata.get("dimensions_m", [1.0, 1.0, 1.0])
        tris = metadata.get("triangle_count", 0)
        verts = metadata.get("vertex_count", 0)
        title = (
            f"3-ANGLE VISUAL VERIFICATION CONTACT SHEET  |  {asset_name}  |  "
            f"Dims: {dims[0]}m x {dims[1]}m x {dims[2]}m  |  "
            f"Verts: {verts}  Tris: {tris}  |  glTF +Y Up"
        )
        draw.text((16, 16), title, fill=(215, 228, 245))

        labels = [
            "1. FRONT VIEW (-Y -> +Y)",
            "2. 3/4 ISOMETRIC VIEW",
            "3. WIREFRAME / TOP VIEW (+Z)",
        ]

        for idx in range(3):
            x0 = idx * panel_w
            y0 = header_h
            if idx < len(view_paths) and view_paths[idx].exists():
                panel_img = Image.open(view_paths[idx]).convert("RGB")
                if panel_img.size != (panel_w, panel_h):
                    panel_img = panel_img.resize(
                        (panel_w, panel_h), Image.Resampling.LANCZOS
                    )
                sheet.paste(panel_img, (x0, y0))

            # Frame border & view badge
            draw.rectangle(
                [x0, y0, x0 + panel_w - 1, y0 + panel_h - 1],
                outline=(52, 63, 86),
                width=2,
            )
            draw.rectangle([x0 + 10, y0 + 10, x0 + 270, y0 + 36], fill=(14, 18, 24))
            draw.text((x0 + 18, y0 + 17), labels[idx], fill=(110, 231, 183))

        output_contact_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_contact_path, format="PNG")

        # Downscale slightly (max edge 1536px -> already <= 1568px VLM limit)
        buf = io.BytesIO()
        sheet.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # -------------------------------------------------------------------------
    # Software glTF 2.0 + 3-View Renderer Fallback (when Blender is absent)
    # -------------------------------------------------------------------------
    @classmethod
    def _fallback_generate_glb_and_views(
        cls,
        script_code: str,
        glb_abs: Path,
        blend_abs: Path,
        temp_dir: Path,
        render_preview: bool,
    ) -> Dict[str, Any]:
        """
        Inspect agent bpy script for primitive commands (cube, cylinder, sphere, cone, torus, monkey)
        and object names (including -col, -colonly, -convcol suffixes), synthesize a real binary .glb
        via trimesh (or built-in glTF 2.0 binary writer), write a valid BLENDER-v402 header stub to
        blend_src/, and software-render the 3 verification views (Front -Y, 3/4 Isometric, Wireframe/Top).
        """
        # Extract any Godot collision suffixes mentioned in script_code
        preserved_suffixes = []
        for match in re.finditer(r'["\']([a-zA-Z0-9_]+-(?:col|colonly|convcol|noimp|navmesh))["\']', script_code):
            full_name = match.group(1)
            sfx = "-" + full_name.split("-")[-1]
            preserved_suffixes.append({"name": full_name, "suffix": sfx})

        code_lower = script_code.lower()
        mesh_count = 1
        try:
            import trimesh  # type: ignore

            meshes: List[Any] = []
            if "primitive_cylinder_add" in code_lower:
                meshes.append(trimesh.creation.cylinder(radius=0.6, height=1.8, sections=24))
            if "primitive_uv_sphere_add" in code_lower or "primitive_ico_sphere_add" in code_lower:
                sph = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
                sph.apply_translation([0.0, 0.0, 0.9])
                meshes.append(sph)
            if "primitive_cone_add" in code_lower:
                cone = trimesh.creation.cone(radius=0.8, height=1.5, sections=24)
                meshes.append(cone)
            if not meshes or "primitive_cube_add" in code_lower:
                box = trimesh.creation.box(extents=[1.4, 1.4, 1.4])
                meshes.append(box)

            mesh_count = len(meshes)
            combined = trimesh.util.concatenate(meshes)
            glb_bytes = combined.export(file_type="glb")
            glb_abs.write_bytes(glb_bytes)
            verts = np.asarray(combined.vertices, dtype=np.float32)
            faces = np.asarray(combined.faces, dtype=np.int32)
        except Exception:
            verts, faces = cls._build_procedural_mesh_arrays(code_lower)
            glb_bytes = cls._pack_gltf2_binary_glb(verts, faces, node_name=glb_abs.stem)
            glb_abs.write_bytes(glb_bytes)

        # Write valid Blender header file in blend_src/
        blend_header = b"BLENDER-v402" + b"\x00" * 52 + script_code.encode("utf-8")
        blend_abs.write_bytes(blend_header)

        min_v = verts.min(axis=0)
        max_v = verts.max(axis=0)
        dims = (max_v - min_v).tolist()

        rendered_views: List[str] = []
        if render_preview:
            front_p = temp_dir / "view_front.png"
            iso_p = temp_dir / "view_iso.png"
            top_p = temp_dir / "view_top_wire.png"

            cls._software_render_mesh_view(verts, faces, front_p, mode="front")
            cls._software_render_mesh_view(verts, faces, iso_p, mode="iso")
            cls._software_render_mesh_view(verts, faces, top_p, mode="top_wire")
            rendered_views = [str(front_p), str(iso_p), str(top_p)]

        return {
            "glb_path": str(glb_abs),
            "blend_path": str(blend_abs),
            "mesh_count": mesh_count,
            "vertex_count": int(len(verts)),
            "triangle_count": int(len(faces)),
            "dimensions_m": [round(float(d), 4) for d in dims],
            "preserved_godot_suffixes": preserved_suffixes,
            "rendered_views": rendered_views,
        }

    @staticmethod
    def _build_procedural_mesh_arrays(code_lower: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D vertices and triangle indices for cylinder + cap or box."""
        if "primitive_cylinder_add" in code_lower:
            sections = 20
            radius = 0.55
            half_h = 1.0
            verts_list = [[0.0, 0.0, -half_h], [0.0, 0.0, half_h]]
            for i in range(sections):
                ang = 2.0 * math.pi * i / sections
                x, y = radius * math.cos(ang), radius * math.sin(ang)
                verts_list.append([x, y, -half_h])
                verts_list.append([x, y, half_h])
            faces_list = []
            for i in range(sections):
                b0 = 2 + i * 2
                t0 = b0 + 1
                b1 = 2 + ((i + 1) % sections) * 2
                t1 = b1 + 1
                faces_list.append([0, b1, b0])
                faces_list.append([1, t0, t1])
                faces_list.append([b0, b1, t1])
                faces_list.append([b0, t1, t0])
            return np.array(verts_list, dtype=np.float32), np.array(faces_list, dtype=np.int32)

        # Default 1.4m cube
        s = 0.7
        verts = np.array(
            [
                [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
                [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int32,
        )
        return verts, faces

    @staticmethod
    def _pack_gltf2_binary_glb(
        verts: np.ndarray, faces: np.ndarray, node_name: str = "GodotProp"
    ) -> bytes:
        """Pack vertices and triangle indices into a valid glTF 2.0 binary .glb container."""
        v_bytes = verts.astype(np.float32).tobytes()
        i_bytes = faces.astype(np.uint16).tobytes()
        bin_chunk = v_bytes + i_bytes
        while len(bin_chunk) % 4 != 0:
            bin_chunk += b"\x00"

        v_min = [float(x) for x in verts.min(axis=0)]
        v_max = [float(x) for x in verts.max(axis=0)]

        gltf_dict = {
            "asset": {"version": "2.0", "generator": "Godot4LocalHarness"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"name": node_name, "mesh": 0}],
            "materials": [{"name": "DefaultPBR", "pbrMetallicRoughness": {"metallicFactor": 0.2, "roughnessFactor": 0.6}}],
            "meshes": [
                {
                    "name": f"{node_name}_Mesh",
                    "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}],
                }
            ],
            "buffers": [{"byteLength": len(bin_chunk)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(v_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": len(v_bytes), "byteLength": len(i_bytes), "target": 34963},
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": int(len(verts)),
                    "type": "VEC3",
                    "min": v_min,
                    "max": v_max,
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": int(faces.size),
                    "type": "SCALAR",
                },
            ],
        }
        json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "

        total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)
        header = struct.pack("<III", 0x46546C67, 2, total_len)
        chunk0 = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        chunk1 = struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
        return header + chunk0 + chunk1

    @staticmethod
    def _software_render_mesh_view(
        verts: np.ndarray,
        faces: np.ndarray,
        out_path: Path,
        mode: str = "iso",
        size: int = 512,
    ) -> None:
        """Render shaded + wireframe 3D view of a mesh using Painter's algorithm onto a 512x512 PNG."""
        img = Image.new("RGB", (size, size), (22, 26, 35))
        draw = ImageDraw.Draw(img)

        # Draw subtle ground grid
        cx, cy = size // 2, size // 2
        for g in range(-4, 5):
            offset = g * 44
            draw.line([(40, cy + offset), (size - 40, cy + offset)], fill=(32, 38, 52), width=1)
            draw.line([(cx + offset, 40), (cx + offset, size - 40)], fill=(32, 38, 52), width=1)

        if len(verts) == 0 or len(faces) == 0:
            img.save(out_path, format="PNG")
            return

        # Center and normalize vertices
        center = verts.mean(axis=0)
        v = verts - center
        span = max(float(np.ptp(v)), 1e-4)
        v = v / span * (size * 0.56)

        if mode == "front":
            # X horizontal, Z vertical, Y depth
            proj_x = cx + v[:, 0]
            proj_y = cy - v[:, 2]
            depth = v[:, 1]
        elif mode == "top_wire":
            # X horizontal, Y vertical, Z depth
            proj_x = cx + v[:, 0]
            proj_y = cy + v[:, 1]
            depth = v[:, 2]
        else:
            # 3/4 Isometric rotation
            yaw = math.radians(38.0)
            pitch = math.radians(28.0)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            cos_p, sin_p = math.cos(pitch), math.sin(pitch)
            x1 = v[:, 0] * cos_y - v[:, 1] * sin_y
            y1 = v[:, 0] * sin_y + v[:, 1] * cos_y
            z1 = v[:, 2]
            proj_x = cx + x1
            proj_y = cy - (z1 * cos_p - y1 * sin_p)
            depth = y1 * cos_p + z1 * sin_p

        # Sort faces back-to-front
        face_depths = depth[faces].mean(axis=1)
        order = np.argsort(face_depths)
        light_dir = np.array([0.45, -0.55, 0.70], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        for f_idx in order:
            tri = faces[f_idx]
            pts = [(float(proj_x[i]), float(proj_y[i])) for i in tri]
            v0, v1, v2 = v[tri[0]], v[tri[1]], v[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = float(np.linalg.norm(normal))
            if norm_len > 1e-6:
                normal /= norm_len
            lambert = max(0.22, min(1.0, float(np.dot(normal, light_dir)) * 0.5 + 0.5))

            if mode == "top_wire":
                fill_col = (int(35 * lambert), int(75 * lambert), int(95 * lambert))
                wire_col = (110, 231, 183)
            else:
                fill_col = (int(75 * lambert), int(145 * lambert), int(215 * lambert))
                wire_col = (40, 60, 90)

            draw.polygon(pts, fill=fill_col, outline=wire_col)

        img.save(out_path, format="PNG")
