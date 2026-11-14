"""
Universal Drag-and-Drop / Clipboard Paste Multimodal Ingestion Pipeline.
Normalizes images, Godot 4 project files (.tscn, .tres, .godot, .gd, .gdshader),
3D models (.glb, .gltf, .obj, .blend), PDFs/documents (.pdf, .csv, .json, .md, .log),
and whole folders into VLM-ready text + base64 image_url payloads.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from godot_parser import GodotParser

logger = logging.getLogger("CustomHarness.MultimodalIngest")


class MultimodalIngestor:
    """
    Universal file, folder, and clipboard payload normalizer for VLM context injection.
    """

    MAX_IMAGE_EDGE_PX = 1568
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tga"}
    GODOT_EXTENSIONS = {".tscn", ".tres", ".godot", ".gd", ".gdshader"}
    MODEL_3D_EXTENSIONS = {".glb", ".gltf", ".obj", ".blend"}
    DOC_EXTENSIONS = {".pdf", ".csv", ".json", ".md", ".log", ".txt", ".cfg", ".ini", ".import"}

    def __init__(self, blender_executable: Optional[str] = None) -> None:
        self.blender_executable = blender_executable

    # -------------------------------------------------------------------------
    # Public Entry Points
    # -------------------------------------------------------------------------
    def ingest_path(self, target_path: str | Path) -> Dict[str, Any]:
        """Ingest a file or directory from disk."""
        path = Path(target_path).resolve()
        if not path.exists():
            return {
                "status": "error",
                "filename": path.name,
                "error": f"Path does not exist: {path}",
            }

        if path.is_dir():
            return self.ingest_directory(path)

        ext = path.suffix.lower()
        if ext in self.IMAGE_EXTENSIONS:
            return self.ingest_image_bytes(path.read_bytes(), filename=path.name, source_path=str(path))
        if ext in self.GODOT_EXTENSIONS or path.name.lower() == "project.godot":
            return self.ingest_godot_file(path)
        if ext in self.MODEL_3D_EXTENSIONS:
            return self.ingest_3d_model(path)
        if ext == ".pdf":
            return self.ingest_pdf_bytes(path.read_bytes(), filename=path.name, source_path=str(path))
        return self.ingest_text_or_log(path)

    def ingest_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest raw uploaded or clipboard-pasted bytes."""
        ext = Path(filename).suffix.lower()
        if (mime_type and mime_type.startswith("image/")) or ext in self.IMAGE_EXTENSIONS:
            return self.ingest_image_bytes(raw_bytes, filename=filename)

        if ext in self.GODOT_EXTENSIONS or filename.lower() == "project.godot":
            text = raw_bytes.decode("utf-8", errors="replace")
            parsed = GodotParser.parse_content(text, filename=filename, include_raw_lines=True)
            return {
                "status": "success",
                "category": "godot_file",
                "filename": filename,
                "extension": ext,
                "summary_text": parsed["summary_text"],
                "context_text": parsed["combined_context"],
                "structured": parsed,
                "images": [],
            }

        if ext in self.MODEL_3D_EXTENSIONS:
            return self.ingest_3d_model_bytes(raw_bytes, filename=filename)

        if ext == ".pdf" or mime_type == "application/pdf":
            return self.ingest_pdf_bytes(raw_bytes, filename=filename)

        text = raw_bytes.decode("utf-8", errors="replace")
        return self._format_text_payload(text, filename=filename, ext=ext)

    # -------------------------------------------------------------------------
    # 1. Images (.png, .jpg, .webp, clipboard screenshots) -> Max 1568px Edge
    # -------------------------------------------------------------------------
    @classmethod
    def ingest_image_bytes(
        cls,
        raw_bytes: bytes,
        filename: str = "clipboard_image.png",
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resize image preserving aspect ratio so the longest edge is max 1568px
        and encode as a base64 PNG image_url.
        """
        img = Image.open(io.BytesIO(raw_bytes))
        orig_w, orig_h = img.size
        orig_mode = img.mode

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")

        longest_edge = max(orig_w, orig_h, 1)
        if longest_edge > cls.MAX_IMAGE_EDGE_PX:
            scale = float(cls.MAX_IMAGE_EDGE_PX) / float(longest_edge)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized = True
        else:
            new_w, new_h = orig_w, orig_h
            resized = False

        out_buf = io.BytesIO()
        img.save(out_buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(out_buf.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{b64_str}"

        summary = (
            f"=== Image Attachment: {filename} | "
            f"{new_w}x{new_h}px (original {orig_w}x{orig_h}px, {orig_mode})"
            + (" [Resized to <=1568px max edge]" if resized else "")
            + " ==="
        )
        return {
            "status": "success",
            "category": "image",
            "filename": filename,
            "path": source_path or filename,
            "original_dimensions": {"width": orig_w, "height": orig_h},
            "dimensions": {"width": new_w, "height": new_h},
            "resized": resized,
            "summary_text": summary,
            "context_text": summary,
            "image_url": data_url,
            "images": [
                {
                    "filename": filename,
                    "image_url": data_url,
                    "width": new_w,
                    "height": new_h,
                }
            ],
        }

    # -------------------------------------------------------------------------
    # 2. Godot 3D & Project Files (.tscn, .tres, .godot, .gd, .gdshader)
    # -------------------------------------------------------------------------
    @classmethod
    def ingest_godot_file(cls, path: Path) -> Dict[str, Any]:
        parsed = GodotParser.parse_file(path, include_raw_lines=True, max_raw_lines=400)
        return {
            "status": "success",
            "category": "godot_file",
            "filename": path.name,
            "path": str(path),
            "extension": path.suffix.lower(),
            "summary_text": parsed["summary_text"],
            "context_text": parsed["combined_context"],
            "structured": parsed,
            "images": [],
        }

    # -------------------------------------------------------------------------
    # 3. 3D Models (.glb, .gltf, .obj, .blend)
    # -------------------------------------------------------------------------
    def ingest_3d_model(self, path: Path) -> Dict[str, Any]:
        ext = path.suffix.lower()
        if ext == ".blend":
            return self._ingest_blend_file(path)
        return self.ingest_3d_model_bytes(
            path.read_bytes(), filename=path.name, source_path=str(path)
        )

    def ingest_3d_model_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract mesh hierarchy, bounding box dimensions, triangle counts, material slots,
        and animation clip names from .glb, .gltf, .obj, or .blend bytes.
        """
        ext = Path(filename).suffix.lower()
        if ext == ".blend":
            return self._parse_blend_binary_header(raw_bytes, filename=filename, source_path=source_path)

        mesh_hierarchy: List[Dict[str, Any]] = []
        materials: List[str] = []
        animations: List[str] = []
        total_verts = 0
        total_tris = 0
        bbox_dims = [0.0, 0.0, 0.0]
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]

        # Extract glTF/GLB node hierarchy, materials, and animation clips via pygltflib
        if ext in (".glb", ".gltf"):
            try:
                from pygltflib import GLTF2  # type: ignore

                if ext == ".glb":
                    gltf = GLTF2.load_from_bytes(raw_bytes)
                else:
                    gltf = GLTF2.from_json(raw_bytes.decode("utf-8", errors="replace"))

                for mat in gltf.materials or []:
                    materials.append(mat.name or f"Material_{len(materials)}")
                for anim in gltf.animations or []:
                    animations.append(anim.name or f"Animation_{len(animations)}")
                for idx, node in enumerate(gltf.nodes or []):
                    mesh_hierarchy.append(
                        {
                            "index": idx,
                            "name": node.name or f"Node_{idx}",
                            "mesh_index": node.mesh,
                            "children": node.children or [],
                            "translation": node.translation,
                            "rotation": node.rotation,
                            "scale": node.scale,
                        }
                    )
            except Exception as exc:
                logger.debug(f"pygltflib inspection warning for {filename}: {exc}")

        # Extract geometry bounding box, vertex count, and triangle count via trimesh
        try:
            import trimesh

            file_type = ext.lstrip(".")
            loaded = trimesh.load(io.BytesIO(raw_bytes), file_type=file_type)
            if isinstance(loaded, trimesh.Scene):
                for geom_name, geom in loaded.geometry.items():
                    v_cnt = int(len(getattr(geom, "vertices", [])))
                    f_cnt = int(len(getattr(geom, "faces", [])))
                    total_verts += v_cnt
                    total_tris += f_cnt
                    if not any(n.get("name") == geom_name for n in mesh_hierarchy):
                        mesh_hierarchy.append(
                            {"name": geom_name, "vertices": v_cnt, "triangles": f_cnt}
                        )
                bounds = loaded.bounds
                if bounds is not None:
                    bbox_min = [round(float(x), 4) for x in bounds[0]]
                    bbox_max = [round(float(x), 4) for x in bounds[1]]
                    bbox_dims = [round(float(bounds[1][i] - bounds[0][i]), 4) for i in range(3)]
            elif hasattr(loaded, "vertices"):
                total_verts = int(len(loaded.vertices))
                total_tris = int(len(loaded.faces))
                bounds = loaded.bounds
                if bounds is not None:
                    bbox_min = [round(float(x), 4) for x in bounds[0]]
                    bbox_max = [round(float(x), 4) for x in bounds[1]]
                    bbox_dims = [round(float(bounds[1][i] - bounds[0][i]), 4) for i in range(3)]
                if not mesh_hierarchy:
                    mesh_hierarchy.append(
                        {"name": Path(filename).stem, "vertices": total_verts, "triangles": total_tris}
                    )
        except Exception as exc:
            logger.debug(f"trimesh inspection warning for {filename}: {exc}")

        # Native glTF 2.0 binary (.glb) JSON chunk parser fallback when trimesh/pygltflib are not installed
        if ext in (".glb", ".gltf") and total_tris == 0:
            try:
                import struct

                if ext == ".glb" and len(raw_bytes) >= 20 and raw_bytes[:4] == b"glTF":
                    json_len, chunk_type = struct.unpack("<II", raw_bytes[12:20])
                    if chunk_type == 0x4E4F534A:
                        gltf_json = json.loads(raw_bytes[20 : 20 + json_len].decode("utf-8", errors="ignore"))
                    else:
                        gltf_json = {}
                else:
                    gltf_json = json.loads(raw_bytes.decode("utf-8", errors="ignore"))

                if not materials:
                    materials = [
                        m.get("name", f"Material_{i}")
                        for i, m in enumerate(gltf_json.get("materials", []))
                    ]
                if not animations:
                    animations = [
                        a.get("name", f"Anim_{i}")
                        for i, a in enumerate(gltf_json.get("animations", []))
                    ]
                if not mesh_hierarchy:
                    for i, n in enumerate(gltf_json.get("nodes", [])):
                        mesh_hierarchy.append(
                            {
                                "index": i,
                                "name": n.get("name", f"Node_{i}"),
                                "mesh_index": n.get("mesh"),
                                "children": n.get("children", []),
                            }
                        )
                accessors = gltf_json.get("accessors", [])
                for acc in accessors:
                    if acc.get("type") == "VEC3" and "min" in acc and "max" in acc:
                        total_verts += int(acc.get("count", 0))
                        bbox_min = [round(float(x), 4) for x in acc["min"][:3]]
                        bbox_max = [round(float(x), 4) for x in acc["max"][:3]]
                        bbox_dims = [round(bbox_max[j] - bbox_min[j], 4) for j in range(3)]
                    elif acc.get("type") == "SCALAR":
                        total_tris += int(acc.get("count", 0)) // 3
            except Exception as exc:
                logger.debug(f"Native GLB chunk parser warning: {exc}")

        summary_lines = [
            f"=== 3D Asset Metadata: {filename} ({ext.upper()}) ===",
            f"  Bounding Box Dimensions (W x H x D): {bbox_dims[0]}m x {bbox_dims[1]}m x {bbox_dims[2]}m",
            f"  Bounds Min / Max: {bbox_min} -> {bbox_max}",
            f"  Geometry Totals : {total_verts} vertices | {total_tris} triangles | {len(mesh_hierarchy)} nodes/meshes",
            f"  Material Slots  : {', '.join(materials) if materials else '(default/none)'}",
            f"  Animation Clips : {', '.join(animations) if animations else '(none)'}",
        ]
        if mesh_hierarchy:
            summary_lines.append("  Mesh / Node Hierarchy:")
            for node in mesh_hierarchy[:25]:
                n_name = node.get("name", "Node")
                extra = []
                if "vertices" in node:
                    extra.append(f"verts={node['vertices']}, tris={node['triangles']}")
                if node.get("translation"):
                    extra.append(f"pos={node['translation']}")
                if node.get("children"):
                    extra.append(f"children={node['children']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                summary_lines.append(f"    - {n_name}{extra_str}")

        summary_text = "\n".join(summary_lines)
        return {
            "status": "success",
            "category": "3d_model",
            "filename": filename,
            "path": source_path or filename,
            "extension": ext,
            "bounding_box_dimensions": bbox_dims,
            "bounds_min": bbox_min,
            "bounds_max": bbox_max,
            "vertex_count": total_verts,
            "triangle_count": total_tris,
            "materials": materials,
            "animations": animations,
            "mesh_hierarchy": mesh_hierarchy,
            "summary_text": summary_text,
            "context_text": summary_text,
            "images": [],
        }

    def _ingest_blend_file(self, path: Path) -> Dict[str, Any]:
        """Inspect a .blend file via headless Blender metadata dump or binary header fallback."""
        if self.blender_executable and Path(self.blender_executable).exists():
            dump_expr = (
                "import bpy, json; "
                "objs=[{'name':o.name,'type':o.type,"
                "'dims':[round(d,4) for d in o.dimensions],"
                "'tris':sum(max(1,len(p.vertices)-2) for p in o.data.polygons) if o.type=='MESH' else 0} "
                "for o in bpy.data.objects]; "
                "mats=[m.name for m in bpy.data.materials]; "
                "anims=[a.name for a in bpy.data.actions]; "
                "print('BLEND_DUMP_JSON:' + json.dumps({'objects':objs,'materials':mats,'animations':anims}))"
            )
            try:
                proc = subprocess.run(
                    [self.blender_executable, "-b", str(path), "--python-expr", dump_expr],
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                )
                for line in (proc.stdout or "").splitlines():
                    if "BLEND_DUMP_JSON:" in line:
                        data = json.loads(line.split("BLEND_DUMP_JSON:", 1)[1])
                        objs = data.get("objects", [])
                        mats = data.get("materials", [])
                        anims = data.get("animations", [])
                        total_tris = sum(o.get("tris", 0) for o in objs)
                        summary_lines = [
                            f"=== Blender Source File (.blend): {path.name} ===",
                            f"  Objects ({len(objs)}), Total Triangles: {total_tris}",
                            f"  Materials: {', '.join(mats) if mats else '(none)'}",
                            f"  Actions/Animations: {', '.join(anims) if anims else '(none)'}",
                        ]
                        for o in objs:
                            summary_lines.append(
                                f"    - {o['name']} [{o['type']}] dims={o.get('dims')} tris={o.get('tris', 0)}"
                            )
                        summary_text = "\n".join(summary_lines)
                        return {
                            "status": "success",
                            "category": "3d_model",
                            "filename": path.name,
                            "path": str(path),
                            "extension": ".blend",
                            "triangle_count": total_tris,
                            "materials": mats,
                            "animations": anims,
                            "mesh_hierarchy": objs,
                            "summary_text": summary_text,
                            "context_text": summary_text,
                            "images": [],
                        }
            except Exception as exc:
                logger.debug(f"Blender CLI dump fallback for {path.name}: {exc}")

        return self._parse_blend_binary_header(
            path.read_bytes(), filename=path.name, source_path=str(path)
        )

    @staticmethod
    def _parse_blend_binary_header(
        raw_bytes: bytes, filename: str, source_path: Optional[str] = None
    ) -> Dict[str, Any]:
        header = raw_bytes[:12].decode("ascii", errors="replace")
        version = header[-3:] if header.startswith("BLENDER") else "402"
        # Scan printable object/material names prefixed by OB / MA / ME / AC in Blender DNA blocks
        printable_names = re.findall(rb"(?:OB|ME|MA|AC)([A-Za-z0-9_\-\.]{2,32})\x00", raw_bytes)
        decoded_names = sorted({n.decode("ascii", errors="ignore") for n in printable_names})
        summary_text = (
            f"=== Blender Source File (.blend): {filename} (Blender v{version[0]}.{version[1:]}) ===\n"
            f"  File Size: {len(raw_bytes):,} bytes\n"
            f"  Detected Datablocks: {', '.join(decoded_names[:20]) if decoded_names else Path(filename).stem}"
        )
        return {
            "status": "success",
            "category": "3d_model",
            "filename": filename,
            "path": source_path or filename,
            "extension": ".blend",
            "blender_version": version,
            "datablocks": decoded_names,
            "summary_text": summary_text,
            "context_text": summary_text,
            "images": [],
        }

    # -------------------------------------------------------------------------
    # 4. Documents & Logs (.pdf with low-text vision rendering, .csv, .json, .md, .log)
    # -------------------------------------------------------------------------
    @classmethod
    def ingest_pdf_bytes(
        cls,
        raw_bytes: bytes,
        filename: str = "document.pdf",
        source_path: Optional[str] = None,
        low_text_threshold_chars: int = 60,
    ) -> Dict[str, Any]:
        """
        Extract text from PDF pages via PyMuPDF (fitz) and automatically render
        low-text PDF pages (diagrams, UI wireframes, level maps) as PNGs for the vision pipeline.
        """
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        page_texts: List[str] = []
        rendered_images: List[Dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            txt = (page.get_text("text") or "").strip()
            page_num = page_idx + 1
            if txt:
                page_texts.append(f"--- Page {page_num} Text ---\n{txt}")
            else:
                page_texts.append(f"--- Page {page_num} [Diagram / Visual Page] ---")

            # Render low-text pages (diagrams / UI mocks) as PNGs for the VLM vision pipeline
            if len(txt) < low_text_threshold_chars and len(rendered_images) < 6:
                pix = page.get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")
                img_payload = cls.ingest_image_bytes(
                    png_bytes, filename=f"{Path(filename).stem}_page_{page_num}.png"
                )
                rendered_images.append(
                    {
                        "page": page_num,
                        "filename": f"{Path(filename).stem}_page_{page_num}.png",
                        "image_url": img_payload["image_url"],
                        "width": img_payload["dimensions"]["width"],
                        "height": img_payload["dimensions"]["height"],
                    }
                )

        doc.close()
        summary = (
            f"=== PDF Document: {filename} ({len(page_texts)} pages, "
            f"{len(rendered_images)} diagram/low-text page(s) rendered to PNG for Vision) ==="
        )
        full_context = summary + "\n\n" + "\n\n".join(page_texts)
        return {
            "status": "success",
            "category": "pdf",
            "filename": filename,
            "path": source_path or filename,
            "page_count": len(page_texts),
            "rendered_diagram_pages": len(rendered_images),
            "summary_text": summary,
            "context_text": full_context,
            "images": rendered_images,
        }

    def ingest_text_or_log(self, path: Path) -> Dict[str, Any]:
        ext = path.suffix.lower()
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._format_text_payload(text, filename=path.name, ext=ext, source_path=str(path))

    @staticmethod
    def _format_text_payload(
        text: str,
        filename: str,
        ext: str,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        lines = text.splitlines()
        summary = f"=== Document/Log: {filename} ({len(lines)} lines) ==="

        if ext == ".csv":
            reader = list(csv.reader(io.StringIO(text)))
            headers = reader[0] if reader else []
            preview_rows = reader[1:16]
            summary = f"=== CSV Table: {filename} ({max(0, len(reader)-1)} rows, columns={headers}) ==="
            formatted_rows = [" | ".join(headers)] + [" | ".join(r) for r in preview_rows]
            body = "\n".join(formatted_rows)
        elif ext == ".json":
            try:
                parsed_json = json.loads(text)
                top_keys = list(parsed_json.keys())[:20] if isinstance(parsed_json, dict) else f"List[{len(parsed_json)}]"
                summary = f"=== JSON File: {filename} (keys={top_keys}) ==="
            except Exception:
                pass
            body = GodotParser.format_numbered_lines(text, max_lines=300)
        else:
            body = GodotParser.format_numbered_lines(text, max_lines=300)

        return {
            "status": "success",
            "category": "document",
            "filename": filename,
            "path": source_path or filename,
            "extension": ext,
            "line_count": len(lines),
            "summary_text": summary,
            "context_text": f"{summary}\n{body}",
            "images": [],
        }

    # -------------------------------------------------------------------------
    # 5. Dropped Folders
    # -------------------------------------------------------------------------
    def ingest_directory(self, dir_path: Path, max_files: int = 40) -> Dict[str, Any]:
        tree_lines = [f"=== Dropped Folder: {dir_path.name} ({dir_path}) ==="]
        godot_summaries: List[str] = []
        collected_images: List[Dict[str, Any]] = []

        files = [
            p
            for p in sorted(dir_path.rglob("*"))
            if p.is_file()
            and not any(part.startswith(".") or part in ("__pycache__", "node_modules", "cache") for part in p.parts)
        ]

        for idx, fpath in enumerate(files[:max_files]):
            rel = fpath.relative_to(dir_path)
            tree_lines.append(f"  - {rel} ({fpath.stat().st_size:,} bytes)")
            ext = fpath.suffix.lower()
            if (ext in self.GODOT_EXTENSIONS or fpath.name == "project.godot") and len(godot_summaries) < 12:
                parsed = GodotParser.parse_file(fpath, include_raw_lines=False)
                godot_summaries.append(parsed["summary_text"])

        summary_text = "\n".join(tree_lines)
        context_text = summary_text
        if godot_summaries:
            context_text += "\n\n--- Key Godot Files in Folder ---\n" + "\n\n".join(godot_summaries)

        return {
            "status": "success",
            "category": "folder",
            "filename": dir_path.name,
            "path": str(dir_path),
            "file_count": len(files),
            "summary_text": summary_text,
            "context_text": context_text,
            "images": collected_images,
        }
