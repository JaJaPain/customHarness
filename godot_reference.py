"""
Offline Godot reference for the agent's `godot_docs` tool.

Two sources, no network:
  1. The engine's own class reference, exported once with `godot --headless --doctool` into
     cache/godot_docs (about 1,000 small XML files, a few seconds). It lists every class with its
     parent, properties, methods, signals and constants for the exact Godot version installed.
  2. A few short built-in guides for things a model gets wrong most often (the .tscn file format).
"""

from __future__ import annotations

import difflib
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("CustomHarness.GodotReference")

MAX_RESULT_CHARS = 3600

GUIDES: Dict[str, str] = {
    "tscn": """HOW TO WRITE A VALID GODOT 4 SCENE FILE (.tscn)
A .tscn file is DATA, not code. Every value is a literal. Never write expressions, function calls or `.new()`.

[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://scripts/main_menu.gd" id="1"]
[ext_resource type="Texture2D" path="res://assets/ui/logo.png" id="2"]
[ext_resource type="PackedScene" path="res://scenes/player.tscn" id="3"]

[sub_resource type="BoxShape3D" id="shape_1"]
size = Vector3(1, 1, 1)

[node name="Root" type="Node3D"]
script = ExtResource("1")

[node name="Logo" type="TextureRect" parent="."]
texture = ExtResource("2")

[node name="Player" parent="." instance=ExtResource("3")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0)

[node name="Body" type="StaticBody3D" parent="."]

[node name="Shape" type="CollisionShape3D" parent="Body"]
shape = SubResource("shape_1")

[connection signal="pressed" from="Menu/PlayButton" to="." method="_on_play_pressed"]

RULES
- Order: [gd_scene], then every [ext_resource], then every [sub_resource], then [node] blocks, then [connection] lines.
- The square brackets of a node header may contain ONLY name, type, parent and instance. `script = ExtResource("id")` is always its own line UNDER the header; `instance=ExtResource("id")` is always INSIDE the header (and that header has no type=).
- The first [node] is the root and has NO parent. Direct children use parent="." ; deeper ones use the path from the root without the root's name, e.g. parent="Body".
- A file from disk (script, texture, audio, another scene, .glb) must be declared as an [ext_resource] and used as ExtResource("id"). To place another scene or a .glb model use a node with instance=ExtResource("id") and no type.
- A resource created inside the scene (shape, material, mesh, environment) is a [sub_resource] used as SubResource("id"). Never write `Type.new()`.
- Value literals: 1.5, true, "text", Vector2(1, 2), Vector3(1, 2, 3), Color(1, 0, 0, 1), NodePath("A/B"), Transform3D(...12 numbers...). No variables, no arithmetic.
- Only set properties that exist on that node type; check with godot_docs("ClassName").
- load_steps = number of ext_resource + sub_resource entries + 1.
""",
    "tres": """HOW TO WRITE A VALID GODOT 4 RESOURCE FILE (.tres)
[gd_resource type="StandardMaterial3D" load_steps=2 format=3]

[ext_resource type="Texture2D" path="res://assets/textures/floor.png" id="1"]

[resource]
albedo_texture = ExtResource("1")
roughness = 0.8

Rules: one [resource] block holding literal property values only; files from disk are [ext_resource] entries; no code.
""",
}


class GodotReference:
    def __init__(self, cache_dir: str | Path, godot_exe_getter: Callable[[], Optional[str]]) -> None:
        self.cache_dir = Path(cache_dir)
        self._godot_exe_getter = godot_exe_getter
        self._class_files: Optional[Dict[str, Path]] = None

    # ------------------------------------------------------------------ class reference on disk
    def _classes_dir(self) -> Path:
        return self.cache_dir / "doc" / "classes"

    def _ensure_exported(self) -> Optional[str]:
        """Export the class reference once. Returns an error message, or None when it is available."""
        if self._class_files:
            return None
        classes_dir = self._classes_dir()
        if not classes_dir.exists() or not any(classes_dir.glob("*.xml")):
            godot_exe = self._godot_exe_getter()
            if not godot_exe:
                return "The Godot executable was not found, so the class reference cannot be exported (set GODOT_PATH)."
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    [godot_exe, "--headless", "--doctool", "."],
                    cwd=str(self.cache_dir),
                    capture_output=True,
                    timeout=180,
                )
            except Exception as exc:
                return f"Exporting the Godot class reference failed: {type(exc).__name__}: {exc}"
        files = {p.stem.lower(): p for p in classes_dir.glob("*.xml")}
        if not files:
            return "The Godot class reference export produced no files."
        self._class_files = files
        return None

    @staticmethod
    def _signature(method: ET.Element) -> str:
        ret = method.find("return")
        params = sorted(method.findall("param"), key=lambda e: int(e.get("index", "0")))
        args = ", ".join(
            f"{p.get('name')}: {p.get('type')}" + (f" = {p.get('default')}" if p.get("default") is not None else "")
            for p in params
        )
        return f"{method.get('name')}({args}) -> {ret.get('type') if ret is not None else 'void'}"

    def _describe_class(self, path: Path, member_filter: str = "") -> str:
        root = ET.parse(path).getroot()
        name = root.get("name", path.stem)
        flt = member_filter.lower()

        def keep(n: Optional[str]) -> bool:
            return not flt or (n is not None and flt in n.lower())

        # inheritance chain, so inherited members can be looked up next
        chain: List[str] = []
        parent = root.get("inherits")
        while parent and len(chain) < 12:
            chain.append(parent)
            parent_file = (self._class_files or {}).get(parent.lower())
            parent = ET.parse(parent_file).getroot().get("inherits") if parent_file else None

        lines = [f"class {name}" + (f"  (inherits {' < '.join(chain)})" if chain else "")]
        props = [
            f"  {m.get('name')}: {m.get('type')}" + (f" = {m.get('default')}" if m.get("default") is not None else "")
            for m in root.findall("./members/member")
            if keep(m.get("name"))
        ]
        methods = [f"  {self._signature(m)}" for m in root.findall("./methods/method") if keep(m.get("name"))]
        signals = [
            "  " + s.get("name", "") + "(" + ", ".join(f"{p.get('name')}: {p.get('type')}" for p in s.findall("param")) + ")"
            for s in root.findall("./signals/signal")
            if keep(s.get("name"))
        ]
        consts = [
            f"  {c.get('name')} = {c.get('value')}" + (f"  ({c.get('enum')})" if c.get("enum") else "")
            for c in root.findall("./constants/constant")
            if keep(c.get("name"))
        ]
        for title, items, cap in (("Properties", props, 70), ("Methods", methods, 45), ("Signals", signals, 25), ("Constants", consts, 30)):
            if items:
                lines.append(f"{title} ({len(items)}):")
                lines.extend(items[:cap])
                if len(items) > cap:
                    lines.append(f"  ... {len(items) - cap} more; narrow with godot_docs(\"{name}.<part of name>\")")
        if flt and len(lines) == 1:
            lines.append(
                f"No member of {name} itself contains '{member_filter}'. It may be inherited - try "
                + ", ".join(f'godot_docs("{c}.{member_filter}")' for c in chain[:3])
            )
        elif chain and not flt:
            lines.append(f"Inherited members are not listed here: look them up with godot_docs(\"{chain[0]}\").")
        return "\n".join(lines)

    # ------------------------------------------------------------------ public lookup
    def lookup(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip().strip("`\"' ")
        if not q:
            return {"status": "error", "error": "Give a class name (e.g. CharacterBody3D), Class.member, or a guide: " + ", ".join(GUIDES)}

        guide_key = q.lower().lstrip(".")
        if guide_key in GUIDES or guide_key in ("scene", "scene file", "tscn format"):
            key = guide_key if guide_key in GUIDES else "tscn"
            return {"status": "success", "kind": "guide", "query": q, "text": GUIDES[key]}

        err = self._ensure_exported()
        if err:
            return {"status": "error", "error": err}
        files = self._class_files or {}

        class_part, _, member_part = q.partition(".")
        class_part = class_part.strip()
        member_part = member_part.strip().rstrip("()")
        if " " in class_part and not member_part:  # "CharacterBody3D velocity"
            class_part, member_part = class_part.split(None, 1)

        hit = files.get(class_part.lower())
        if hit:
            text = self._describe_class(hit, member_part)
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "\n... (cut; narrow the query with Class.member)"
            return {"status": "success", "kind": "class", "query": q, "text": text}

        # Not a class: say so clearly and suggest what exists
        names = [p.stem for p in files.values()]
        close = difflib.get_close_matches(class_part, names, n=6, cutoff=0.6)
        containing = sorted(n for n in names if class_part.lower() in n.lower())[:12]
        suggestions = list(dict.fromkeys(close + containing))
        text = f"There is NO class named '{class_part}' in this Godot version."
        if suggestions:
            text += " Classes with similar names: " + ", ".join(suggestions) + "."
        text += " Guides available: " + ", ".join(f'godot_docs("{g}")' for g in GUIDES) + "."
        return {"status": "success", "kind": "not_found", "query": q, "text": text}
