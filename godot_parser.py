"""
Godot 4 Scene, Resource, Script, Shader, and Project Parser.
Converts verbose .tscn, .tres, .godot, .gd, and .gdshader files into compact,
token-efficient 3D Scene Graph Summaries alongside line-numbered raw text.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GodotParser:
    """Dedicated parser for Godot 4 project files (.tscn, .tres, .godot, .gd, .gdshader)."""

    HEADER_RE = re.compile(r"^\[(gd_scene|gd_resource)\s+([^\]]*)\]")
    SECTION_RE = re.compile(r"^\[([a-zA-Z0-9_]+)(?:\s+([^\]]*))?\]")
    ATTR_RE = re.compile(
        r'([a-zA-Z0-9_]+)\s*=\s*("[^"]*"|ExtResource\([^\)]+\)|SubResource\([^\)]+\)|PackedStringArray\([^\)]*\)|[^\s\]]+)'
    )
    TRANSFORM3D_RE = re.compile(
        r"Transform3D\(\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
    )
    VECTOR3_RE = re.compile(
        r"Vector3\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
    )
    VECTOR2_RE = re.compile(
        r"Vector2\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
    )
    EXT_REF_RE = re.compile(r'ExtResource\(\s*"?([^"\)]+)"?\s*\)')
    SUB_REF_RE = re.compile(r'SubResource\(\s*"?([^"\)]+)"?\s*\)')

    @classmethod
    def parse_file(
        cls,
        file_path: str | Path,
        include_raw_lines: bool = True,
        max_raw_lines: int = 400,
    ) -> Dict[str, Any]:
        """Parse any Godot 4 file and return both structured summary and formatted text."""
        path = Path(file_path)
        ext = path.suffix.lower()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {
                "path": str(path),
                "extension": ext,
                "error": f"Failed to read file: {exc}",
                "summary_text": f"[Error reading {path.name}: {exc}]",
            }

        return cls.parse_content(
            content=content,
            filename=path.name,
            file_path=str(path),
            include_raw_lines=include_raw_lines,
            max_raw_lines=max_raw_lines,
        )

    @classmethod
    def parse_content(
        cls,
        content: str,
        filename: str = "scene.tscn",
        file_path: Optional[str] = None,
        include_raw_lines: bool = True,
        max_raw_lines: int = 400,
    ) -> Dict[str, Any]:
        """Parse Godot file content based on extension."""
        ext = Path(filename).suffix.lower()
        if ext in (".tscn", ".tres"):
            result = cls._parse_tscn_or_tres(content, filename, file_path)
        elif ext == ".gd":
            result = cls._parse_gdscript(content, filename, file_path)
        elif ext == ".gdshader":
            result = cls._parse_gdshader(content, filename, file_path)
        elif ext == ".godot" or filename.lower() == "project.godot":
            result = cls._parse_project_godot(content, filename, file_path)
        else:
            result = {
                "filename": filename,
                "path": file_path or filename,
                "extension": ext,
                "summary_text": f"=== File: {filename} ({len(content.splitlines())} lines) ===",
            }

        if include_raw_lines:
            numbered = cls.format_numbered_lines(content, max_lines=max_raw_lines)
            result["numbered_raw_text"] = numbered
            result["combined_context"] = (
                f"{result['summary_text']}\n\n"
                f"--- Raw Source ({filename}, line-numbered) ---\n{numbered}"
            )
        else:
            result["combined_context"] = result["summary_text"]

        return result

    @staticmethod
    def format_numbered_lines(
        content: str, start_line: int = 1, max_lines: int = 400
    ) -> str:
        """Return 1-indexed line-numbered text."""
        lines = content.splitlines()
        total = len(lines)
        sliced = lines[start_line - 1 : start_line - 1 + max_lines]
        formatted = [
            f"{idx:4d} | {line}"
            for idx, line in enumerate(sliced, start=start_line)
        ]
        if total > start_line - 1 + max_lines:
            remaining = total - (start_line - 1 + max_lines)
            formatted.append(f" ... | [{remaining} more lines truncated; use read_file with start_line/end_line]")
        return "\n".join(formatted)

    @classmethod
    def decode_transform3d(cls, raw_val: str) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Decode a Godot Transform3D(xx, xy, xz, yx, yy, yz, zx, zy, zz, tx, ty, tz)
        into human-readable position (x, y, z), Euler rotation_deg (rx, ry, rz), and scale (sx, sy, sz).
        """
        m = cls.TRANSFORM3D_RE.search(raw_val)
        if not m:
            return None
        vals = [float(m.group(i)) for i in range(1, 13)]
        # Godot Transform3D basis rows/columns:
        # basis.x = (vals[0], vals[1], vals[2])
        # basis.y = (vals[3], vals[4], vals[5])
        # basis.z = (vals[6], vals[7], vals[8])
        # origin  = (vals[9], vals[10], vals[11])
        bx = (vals[0], vals[1], vals[2])
        by = (vals[3], vals[4], vals[5])
        bz = (vals[6], vals[7], vals[8])
        origin = (round(vals[9], 4), round(vals[10], 4), round(vals[11], 4))

        sx = math.sqrt(bx[0] ** 2 + bx[1] ** 2 + bx[2] ** 2)
        sy = math.sqrt(by[0] ** 2 + by[1] ** 2 + by[2] ** 2)
        sz = math.sqrt(bz[0] ** 2 + bz[1] ** 2 + bz[2] ** 2)

        if sx > 1e-6 and sy > 1e-6 and sz > 1e-6:
            r00, r01, r02 = bx[0] / sx, by[0] / sy, bz[0] / sz
            r10, r11, r12 = bx[1] / sx, by[1] / sy, bz[1] / sz
            r20, r21, r22 = bx[2] / sx, by[2] / sy, bz[2] / sz

            # Extract YXZ Euler angles (Godot 3D standard Euler order)
            sin_x = max(-1.0, min(1.0, -r12))
            rx = math.asin(sin_x)
            if abs(sin_x) < 0.99999:
                ry = math.atan2(r02, r22)
                rz = math.atan2(r10, r11)
            else:
                ry = math.atan2(-r20, r00)
                rz = 0.0
            rot_deg = (
                round(math.degrees(rx), 2),
                round(math.degrees(ry), 2),
                round(math.degrees(rz), 2),
            )
        else:
            rot_deg = (0.0, 0.0, 0.0)

        scale = (round(sx, 4), round(sy, 4), round(sz, 4))
        return {
            "position": origin,
            "rotation_deg": rot_deg,
            "scale": scale,
        }

    @classmethod
    def _parse_attrs(cls, attr_str: str) -> Dict[str, str]:
        if not attr_str:
            return {}
        attrs: Dict[str, str] = {}
        for match in cls.ATTR_RE.finditer(attr_str):
            k = match.group(1)
            v = match.group(2).strip('"')
            attrs[k] = v
        return attrs

    @classmethod
    def _parse_tscn_or_tres(
        cls, content: str, filename: str, file_path: Optional[str]
    ) -> Dict[str, Any]:
        lines = content.splitlines()
        header_info: Dict[str, str] = {}
        ext_resources: Dict[str, Dict[str, Any]] = {}
        sub_resources: Dict[str, Dict[str, Any]] = {}
        nodes: List[Dict[str, Any]] = []
        connections: List[Dict[str, str]] = []
        resource_props: Dict[str, str] = {}

        current_section_type: Optional[str] = None
        current_section_attrs: Dict[str, str] = {}
        current_props: Dict[str, str] = {}
        current_line_start: int = 1

        def flush_section() -> None:
            nonlocal current_section_type, current_section_attrs, current_props
            if not current_section_type:
                return
            if current_section_type in ("gd_scene", "gd_resource"):
                header_info.update(current_section_attrs)
                header_info["kind"] = current_section_type
            elif current_section_type == "ext_resource":
                res_id = current_section_attrs.get("id", f"ext_{len(ext_resources)}")
                ext_resources[res_id] = {
                    "id": res_id,
                    "type": current_section_attrs.get("type", "Resource"),
                    "path": current_section_attrs.get("path", ""),
                    "uid": current_section_attrs.get("uid", ""),
                    "line": current_line_start,
                }
            elif current_section_type == "sub_resource":
                res_id = current_section_attrs.get("id", f"sub_{len(sub_resources)}")
                sub_resources[res_id] = {
                    "id": res_id,
                    "type": current_section_attrs.get("type", "Resource"),
                    "properties": dict(current_props),
                    "line": current_line_start,
                }
            elif current_section_type == "node":
                node_entry = cls._build_node_entry(
                    current_section_attrs,
                    current_props,
                    ext_resources,
                    sub_resources,
                    current_line_start,
                )
                nodes.append(node_entry)
            elif current_section_type == "connection":
                connections.append(dict(current_section_attrs))
            elif current_section_type == "resource":
                resource_props.update(current_props)

            current_section_type = None
            current_section_attrs = {}
            current_props = {}

        for idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            sec_match = cls.SECTION_RE.match(line)
            if sec_match:
                flush_section()
                current_section_type = sec_match.group(1)
                current_section_attrs = cls._parse_attrs(sec_match.group(2) or "")
                current_line_start = idx
                continue

            if "=" in line and current_section_type in ("node", "sub_resource", "resource"):
                key, _, val = line.partition("=")
                current_props[key.strip()] = val.strip()

        flush_section()

        summary_lines: List[str] = []
        kind_label = "Godot 4 Scene (.tscn)" if filename.endswith(".tscn") else "Godot 4 Resource (.tres)"
        uid_str = f" | uid={header_info.get('uid')}" if header_info.get("uid") else ""
        fmt_str = f" | format={header_info.get('format', '3')}"
        summary_lines.append(f"=== {kind_label}: {filename}{uid_str}{fmt_str} ===")

        if ext_resources:
            summary_lines.append(f"\n[External Resources ({len(ext_resources)})]")
            for r_id, r in ext_resources.items():
                summary_lines.append(f"  - [{r_id}] {r['type']}: {r['path']} (L{r['line']})")

        if sub_resources:
            summary_lines.append(f"\n[Sub-Resources ({len(sub_resources)})]")
            for s_id, s in sub_resources.items():
                prop_preview = ", ".join(
                    f"{k}={v}" for k, v in list(s["properties"].items())[:4]
                )
                prop_str = f" ({prop_preview})" if prop_preview else ""
                summary_lines.append(f"  - [{s_id}] {s['type']}{prop_str} (L{s['line']})")

        if nodes:
            summary_lines.append(f"\n[3D/2D Scene Graph Hierarchy ({len(nodes)} nodes)]")
            for node in nodes:
                depth = node["depth"]
                indent = "  " * (depth + 1)
                branch = "└─ " if depth > 0 else "● "
                type_label = node["type"]
                if node.get("instance_path"):
                    type_label = f"Instance({node['instance_path']})"

                line_parts = [f"{indent}{branch}{node['name']} [{type_label}] (L{node['line']})"]

                if node.get("transform3d"):
                    t3 = node["transform3d"]
                    pos = t3["position"]
                    rot = t3["rotation_deg"]
                    scl = t3["scale"]
                    t_str = f"pos=({pos[0]}, {pos[1]}, {pos[2]})"
                    if any(abs(v) > 0.01 for v in rot):
                        t_str += f" rot_deg=({rot[0]}°, {rot[1]}°, {rot[2]}°)"
                    if any(abs(v - 1.0) > 0.001 for v in scl):
                        t_str += f" scale=({scl[0]}, {scl[1]}, {scl[2]})"
                    line_parts.append(f"{{{t_str}}}")
                elif node.get("transform2d_summary"):
                    line_parts.append(f"{{{node['transform2d_summary']}}}")

                if node.get("script_path"):
                    line_parts.append(f"script={node['script_path']}")
                if node.get("collision_info"):
                    line_parts.append(node["collision_info"])
                if node.get("mesh_or_shape"):
                    line_parts.append(node["mesh_or_shape"])
                if node.get("groups"):
                    line_parts.append(f"groups={node['groups']}")
                if node.get("exports"):
                    exp_preview = ", ".join(
                        f"{k}={v}" for k, v in list(node["exports"].items())[:5]
                    )
                    line_parts.append(f"props=[{exp_preview}]")

                summary_lines.append(" ".join(line_parts))

        if resource_props:
            summary_lines.append(f"\n[Resource Properties ({len(resource_props)})]")
            for k, v in list(resource_props.items())[:25]:
                summary_lines.append(f"  - {k} = {v}")

        if connections:
            summary_lines.append(f"\n[Signal Connections ({len(connections)})]")
            for conn in connections:
                sig = conn.get("signal", "?")
                frm = conn.get("from", ".")
                to = conn.get("to", ".")
                method = conn.get("method", "?")
                summary_lines.append(f"  - {frm}.{sig} --> {to}::{method}()")

        return {
            "filename": filename,
            "path": file_path or filename,
            "extension": Path(filename).suffix.lower(),
            "header": header_info,
            "ext_resources": ext_resources,
            "sub_resources": sub_resources,
            "nodes": nodes,
            "connections": connections,
            "resource_properties": resource_props,
            "summary_text": "\n".join(summary_lines),
        }

    @classmethod
    def _build_node_entry(
        cls,
        attrs: Dict[str, str],
        props: Dict[str, str],
        ext_resources: Dict[str, Dict[str, Any]],
        sub_resources: Dict[str, Dict[str, Any]],
        line_num: int,
    ) -> Dict[str, Any]:
        name = attrs.get("name", "Node")
        node_type = attrs.get("type", "Node")
        parent = attrs.get("parent")
        if parent is None:
            full_path = name
            depth = 0
        elif parent == ".":
            full_path = name
            depth = 1
        else:
            full_path = f"{parent}/{name}"
            depth = parent.count("/") + 2

        instance_ref = attrs.get("instance", "")
        instance_path = None
        if instance_ref:
            m = cls.EXT_REF_RE.search(instance_ref)
            if m and m.group(1) in ext_resources:
                instance_path = ext_resources[m.group(1)]["path"]
            else:
                instance_path = instance_ref

        # Transform3D or explicit 3D position/rotation/scale
        transform3d = None
        if "transform" in props and "Transform3D" in props["transform"]:
            transform3d = cls.decode_transform3d(props["transform"])
        elif any(k in props for k in ("position", "rotation", "scale")):
            v3_pos = cls.VECTOR3_RE.search(props.get("position", ""))
            v3_rot = cls.VECTOR3_RE.search(props.get("rotation", ""))
            v3_scl = cls.VECTOR3_RE.search(props.get("scale", ""))
            if v3_pos or v3_rot or v3_scl:
                pos = (
                    tuple(round(float(v3_pos.group(i)), 4) for i in (1, 2, 3))
                    if v3_pos
                    else (0.0, 0.0, 0.0)
                )
                rot = (
                    tuple(round(math.degrees(float(v3_rot.group(i))), 2) for i in (1, 2, 3))
                    if v3_rot
                    else (0.0, 0.0, 0.0)
                )
                scl = (
                    tuple(round(float(v3_scl.group(i)), 4) for i in (1, 2, 3))
                    if v3_scl
                    else (1.0, 1.0, 1.0)
                )
                transform3d = {"position": pos, "rotation_deg": rot, "scale": scl}

        # 2D transform summary if Vector2
        transform2d_summary = None
        if not transform3d and "position" in props and "Vector2" in props["position"]:
            transform2d_summary = f"pos2d={props['position']}"

        # Script binding
        script_path = None
        if "script" in props:
            m = cls.EXT_REF_RE.search(props["script"])
            if m and m.group(1) in ext_resources:
                script_path = ext_resources[m.group(1)]["path"]
            else:
                script_path = props["script"]

        # Collision layers/masks
        col_parts = []
        if "collision_layer" in props:
            col_parts.append(f"layer={props['collision_layer']}")
        if "collision_mask" in props:
            col_parts.append(f"mask={props['collision_mask']}")
        collision_info = f"collision({', '.join(col_parts)})" if col_parts else None

        # Mesh or CollisionShape reference
        mesh_or_shape = None
        for key in ("mesh", "shape", "material_override", "environment"):
            if key in props:
                sub_m = cls.SUB_REF_RE.search(props[key])
                ext_m = cls.EXT_REF_RE.search(props[key])
                if sub_m and sub_m.group(1) in sub_resources:
                    sr = sub_resources[sub_m.group(1)]
                    mesh_or_shape = f"{key}={sr['type']}({sub_m.group(1)})"
                elif ext_m and ext_m.group(1) in ext_resources:
                    er = ext_resources[ext_m.group(1)]
                    mesh_or_shape = f"{key}={er['path']}"
                else:
                    mesh_or_shape = f"{key}={props[key]}"
                break

        # Custom/exported properties
        ignored_keys = {
            "transform",
            "position",
            "rotation",
            "scale",
            "script",
            "collision_layer",
            "collision_mask",
            "mesh",
            "shape",
            "material_override",
            "environment",
        }
        exports = {k: v for k, v in props.items() if k not in ignored_keys}

        return {
            "name": name,
            "type": node_type,
            "parent": parent,
            "full_path": full_path,
            "depth": depth,
            "instance_path": instance_path,
            "transform3d": transform3d,
            "transform2d_summary": transform2d_summary,
            "script_path": script_path,
            "collision_info": collision_info,
            "mesh_or_shape": mesh_or_shape,
            "groups": attrs.get("groups"),
            "exports": exports,
            "line": line_num,
        }

    @classmethod
    def _parse_gdscript(
        cls, content: str, filename: str, file_path: Optional[str]
    ) -> Dict[str, Any]:
        lines = content.splitlines()
        extends_class = "RefCounted"
        class_name = None
        is_tool = False
        signals: List[Dict[str, Any]] = []
        exports: List[Dict[str, Any]] = []
        onready_vars: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []

        for idx, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if stripped.startswith("@tool"):
                is_tool = True
            elif stripped.startswith("extends "):
                extends_class = stripped.split("extends ", 1)[1].strip()
            elif stripped.startswith("class_name "):
                class_name = stripped.split("class_name ", 1)[1].strip()
            elif stripped.startswith("signal "):
                signals.append({"declaration": stripped, "line": idx})
            elif stripped.startswith("@export"):
                exports.append({"declaration": stripped, "line": idx})
            elif stripped.startswith("@onready"):
                onready_vars.append({"declaration": stripped, "line": idx})
            elif stripped.startswith("func ") or stripped.startswith("static func "):
                functions.append({"signature": stripped.rstrip(":"), "line": idx})

        summary_lines = [
            f"=== GDScript 2.0: {filename} (extends {extends_class}"
            + (f", class_name {class_name}" if class_name else "")
            + (", @tool" if is_tool else "")
            + f", {len(lines)} lines) ==="
        ]
        if signals:
            summary_lines.append("Signals: " + ", ".join(f"{s['declaration']} (L{s['line']})" for s in signals))
        if exports:
            summary_lines.append("Exports:")
            for exp in exports:
                summary_lines.append(f"  - L{exp['line']}: {exp['declaration']}")
        if onready_vars:
            summary_lines.append("OnReady Node Bindings:")
            for onr in onready_vars:
                summary_lines.append(f"  - L{onr['line']}: {onr['declaration']}")
        if functions:
            summary_lines.append("Functions:")
            for fn in functions:
                summary_lines.append(f"  - L{fn['line']}: {fn['signature']}")

        return {
            "filename": filename,
            "path": file_path or filename,
            "extension": ".gd",
            "extends": extends_class,
            "class_name": class_name,
            "is_tool": is_tool,
            "signals": signals,
            "exports": exports,
            "onready_vars": onready_vars,
            "functions": functions,
            "summary_text": "\n".join(summary_lines),
        }

    @classmethod
    def _parse_gdshader(
        cls, content: str, filename: str, file_path: Optional[str]
    ) -> Dict[str, Any]:
        lines = content.splitlines()
        shader_type = "unknown"
        render_modes: List[str] = []
        uniforms: List[str] = []
        for idx, line in enumerate(lines, start=1):
            s = line.strip()
            if s.startswith("shader_type "):
                shader_type = s.replace("shader_type", "").rstrip(";").strip()
            elif s.startswith("render_mode "):
                render_modes.append(s.replace("render_mode", "").rstrip(";").strip())
            elif s.startswith("uniform "):
                uniforms.append(f"L{idx}: {s}")

        summary_lines = [
            f"=== Godot Shader (.gdshader): {filename} | shader_type={shader_type} ==="
        ]
        if render_modes:
            summary_lines.append(f"Render Modes: {', '.join(render_modes)}")
        if uniforms:
            summary_lines.append("Uniforms:\n  " + "\n  ".join(uniforms))

        return {
            "filename": filename,
            "path": file_path or filename,
            "extension": ".gdshader",
            "shader_type": shader_type,
            "render_modes": render_modes,
            "uniforms": uniforms,
            "summary_text": "\n".join(summary_lines),
        }

    @classmethod
    def _parse_project_godot(
        cls, content: str, filename: str, file_path: Optional[str]
    ) -> Dict[str, Any]:
        sections: Dict[str, Dict[str, str]] = {"root": {}}
        current_sec = "root"
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_sec = line[1:-1].strip()
                sections.setdefault(current_sec, {})
            elif "=" in line:
                k, _, v = line.partition("=")
                sections.setdefault(current_sec, {})[k.strip()] = v.strip()

        app_sec = sections.get("application", {})
        proj_name = app_sec.get("config/name", '"Godot Project"').strip('"')
        main_scene = app_sec.get("run/main_scene", '""').strip('"')
        features = app_sec.get("config/features", "")
        autoloads = sections.get("autoload", {})
        inputs = list(sections.get("input", {}).keys())

        summary_lines = [
            f"=== Godot 4 Project Configuration ({filename}) ===",
            f"  Project Name : {proj_name}",
            f"  Main Scene   : {main_scene or '(not set)'}",
            f"  Features     : {features}",
        ]
        if autoloads:
            summary_lines.append("  Autoload Singletons:")
            for k, v in autoloads.items():
                summary_lines.append(f"    - {k}: {v}")
        if inputs:
            summary_lines.append(f"  Input Actions ({len(inputs)}): {', '.join(inputs[:20])}")

        return {
            "filename": filename,
            "path": file_path or filename,
            "extension": ".godot",
            "project_name": proj_name,
            "main_scene": main_scene,
            "autoloads": autoloads,
            "input_actions": inputs,
            "sections": sections,
            "summary_text": "\n".join(summary_lines),
        }
