# -*- coding: utf-8 -*-
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

OUTPUT_GLB_PATH = "D:\\CodingProjects\\CustomHarness\\workspace\\assets\\models\\paddle.glb"
OUTPUT_BLEND_PATH = "D:\\CodingProjects\\CustomHarness\\workspace\\assets\\models\\blend_src\\paddle.blend"
CONTACT_SHEET_DIR = "C:\\Users\\abejh\\AppData\\Local\\Temp\\godot_blender_harness_74q90r17"
RENDER_PREVIEW = True
HEAVY_CYCLES_BAKE = False

# 1. Clean Factory Scene & Enforce Metric Units (1.0m = 1 Godot Unit)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0
scene.unit_settings.length_unit = 'METERS'

# 2. Execute Agent Procedural bpy Script
# ==================== BEGIN AGENT SCRIPT ====================
import bpy
# Beveled sci-fi energy paddle body
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.25))
paddle = bpy.context.active_object
paddle.name = 'PaddleBody'
paddle.scale = (2.8, 0.55, 0.42)
bev = paddle.modifiers.new(name='Bevel', type='BEVEL')
bev.width = 0.08
bev.segments = 3
mat_body = bpy.data.materials.new(name='PaddleEnergyMat')
mat_body.use_nodes = True
bsdf = mat_body.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.12, 0.72, 0.98, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.85
    bsdf.inputs['Roughness'].default_value = 0.22
paddle.data.materials.append(mat_body)
# Glowing side thrusters
for side, x_pos in (('L', -1.55), ('R', 1.55)):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.24, depth=0.65, location=(x_pos, 0, 0.25))
    thr = bpy.context.active_object
    thr.name = f'PaddleThruster_{side}'
    thr.data.materials.append(mat_body)
# Clean -col collision mesh
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.25))
col = bpy.context.active_object
col.name = 'Paddle-col'
col.scale = (3.3, 0.65, 0.45)

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
