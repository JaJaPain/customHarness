"""
2D Image Generation Engine Worker (stable-diffusion.cpp CUDA + FLUX.1-schnell / FLUX.2-klein).
Implements Tier 3 Atomic Image-Gen VRAM Swap, rembg / alpha-key background removal,
Sobel PBR Normal/Roughness/Heightmap synthesis, and Closed-Loop Vision Self-Check.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from vram_orchestrator import IMAGE_MODELS, VRAMOrchestrator

logger = logging.getLogger("CustomHarness.ImageGenWorker")


ASSET_TYPE_PROMPTS: Dict[str, str] = {
    "sprite": "2D game sprite asset, clean silhouette, centered, studio lighting, isolated background, crisp details, Godot 4 game art",
    "ui_icon": "stylized RPG game UI icon, centered emblem, crisp edges, high contrast, clean dark background, game interface art",
    "seamless_texture": "seamless tileable PBR surface texture, orthographic top-down view, uniform lighting, no harsh shadows, 4k material detail",
    "skybox_concept": "360 equirectangular panoramic environment skybox concept, atmospheric lighting, horizon aligned, game level backdrop",
}


class ImageGenWorker:
    """
    Executes 2D texture, sprite, UI icon, and skybox generation inside 12GB VRAM
    using stable-diffusion.cpp (sd.exe) with FLUX.1-schnell-Q5_K_S.gguf (4-step 1024x1024).
    """

    def __init__(
        self,
        orchestrator: VRAMOrchestrator,
        workspace_dir: str | Path,
    ) -> None:
        self.orchestrator = orchestrator
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def resolve_output_path(self, output_path: str) -> Path:
        """Resolve a Godot res:// path or relative path inside workspace_dir."""
        cleaned = output_path.strip()
        if cleaned.startswith("res://"):
            cleaned = cleaned[len("res://") :]
        cleaned = cleaned.lstrip("/\\")
        if not cleaned:
            cleaned = f"assets/textures/gen_{int(time.time())}.png"
        target = (self.workspace_dir / cleaned).resolve()
        if not str(target).startswith(str(self.workspace_dir)):
            target = (self.workspace_dir / "assets" / Path(cleaned).name).resolve()
        if not target.suffix:
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def to_res_path(self, abs_path: Path) -> str:
        try:
            rel = abs_path.resolve().relative_to(self.workspace_dir)
            return "res://" + str(rel).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def build_sd_command(
        self,
        prompt: str,
        output_file: Path,
        asset_type: str = "sprite",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: int = -1,
    ) -> List[str]:
        sd_exe = self.orchestrator.get_sd_binary_path()
        img_model_info = IMAGE_MODELS[self.orchestrator.active_image_model_id]
        model_path = self.orchestrator.models_dir / img_model_info["filename"]

        cmd = [
            str(sd_exe),
            "--diffusion-model",
            str(model_path),
            "-p",
            prompt,
            "-o",
            str(output_file),
            "-W",
            str(width),
            "-H",
            str(height),
            "--steps",
            str(steps),
            "--cfg-scale",
            "1.0",
            "--sampling-method",
            "euler",
            "--vae-tiling",
            # Keep each component (T5 / CLIP / FLUX / VAE) in system RAM and move it into VRAM only
            # while it is needed, so the 8 GB FLUX model + 3.4 GB T5 fit a 12 GB card.
            "--offload-to-cpu",
            "-s",
            str(seed),
        ]

        # FLUX needs its VAE and both text encoders alongside the diffusion model
        for flag, fname in self.FLUX_COMPANION_FILES:
            aux = self.orchestrator.models_dir / fname
            if aux.exists():
                cmd.extend([flag, str(aux)])

        return cmd

    FLUX_COMPANION_FILES = (
        ("--vae", "ae.safetensors"),
        ("--clip_l", "clip_l.safetensors"),
        ("--t5xxl", "t5xxl_q5_k.gguf"),
    )

    @staticmethod
    def _scripted_demo_enabled() -> bool:
        return os.environ.get("HARNESS_SCRIPTED_DEMO") == "1"

    def describe_missing_requirements(self) -> Optional[str]:
        """Return a human-readable reason real image generation cannot run, or None if everything is present."""
        sd_exe = self.orchestrator.get_sd_binary_path()
        img_model_info = IMAGE_MODELS[self.orchestrator.active_image_model_id]
        models_dir = self.orchestrator.models_dir
        missing: List[str] = []
        if not sd_exe.exists():
            missing.append(f"stable-diffusion.cpp binary ({sd_exe.name}) in bin/")
        if not (models_dir / img_model_info["filename"]).exists():
            missing.append(f"{img_model_info['filename']} in models/")
        for _flag, fname in self.FLUX_COMPANION_FILES:
            if not (models_dir / fname).exists():
                missing.append(f"{fname} in models/")
        if missing:
            return (
                "Image generation is not installed yet. Missing: " + "; ".join(missing)
                + ". Open Setup / Downloads to install them."
            )
        return None

    async def generate_2d_asset(
        self,
        prompt: str,
        asset_type: str = "sprite",
        output_path: str = "res://assets/textures/generated_asset.png",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        transparent_bg: bool = False,
        generate_pbr_maps: bool = False,
        seed: int = -1,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        defer_llm_restore: bool = False,
    ) -> Dict[str, Any]:
        """
        Full Tier 3 Atomic Image-Gen VRAM Swap Pipeline:
          Step A: Snapshot LLM KV slot 0 to active_session.bin + JSON fallback.
          Step B: Terminate llama-server.exe and verify free VRAM > 10.5 GB.
          Step C: Run stable-diffusion.cpp (FLUX.1-schnell 4-step 1024x1024), optional rembg/alpha-key,
                  and optional Sobel PBR Normal/Roughness/Heightmap generation.
          Step D: Terminate image-gen, verify VRAM clear, relaunch llama-server.exe, restore KV slot 0.
          Step E: Closed-Loop Vision Self-Check (attach downscaled base64 preview for the VLM).
        """
        start_time = time.monotonic()
        asset_type = asset_type if asset_type in ASSET_TYPE_PROMPTS else "sprite"
        enhanced_prompt = f"{prompt.strip()}, {ASSET_TYPE_PROMPTS[asset_type]}"
        if transparent_bg:
            # Background removal keys out a flat colour, so ask the model for one explicitly
            enhanced_prompt = (
                enhanced_prompt.replace("clean dark background, ", "").replace("isolated background, ", "")
                + ", single object isolated on a plain flat solid white background, no shadow, no gradient, no frame"
            )
        out_file = self.resolve_output_path(output_path)

        # Step A & Step B: Acquire VRAM via Orchestrator
        swap_in_meta = await self.orchestrator.acquire_vram_for_imagegen(
            conversation_history=conversation_history
        )

        sd_cmd = self.build_sd_command(
            prompt=enhanced_prompt,
            output_file=out_file,
            asset_type=asset_type,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
        )

        engine_used = "stable-diffusion.cpp (CUDA)"
        sd_stdout = ""
        bg_removal_method = None
        pbr_files: Dict[str, str] = {}
        try:
            # Step C: run the real stable-diffusion.cpp. The procedural synthesizer is only used by the
            # scripted test mode; in normal use a failure is reported, never papered over with fake art.
            sd_exe = self.orchestrator.get_sd_binary_path()
            img_model_info = IMAGE_MODELS[self.orchestrator.active_image_model_id]
            model_path = self.orchestrator.models_dir / img_model_info["filename"]
            missing_reason = self.describe_missing_requirements()

            if missing_reason and not self._scripted_demo_enabled():
                raise RuntimeError(missing_reason)

            if not missing_reason:
                if out_file.exists():
                    out_file.unlink()  # never mistake a stale file for a fresh result
                creationflags = (
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                )
                proc = await asyncio.create_subprocess_exec(
                    *sd_cmd,
                    cwd=str(self.orchestrator.base_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=creationflags,
                )
                self.orchestrator._sd_process = proc  # lets the orchestrator stop it on shutdown
                try:
                    stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=900.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    raise RuntimeError("stable-diffusion.cpp did not finish within 15 minutes and was stopped.")
                finally:
                    self.orchestrator._sd_process = None
                sd_stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
                try:
                    # Full log of the last run, for diagnosing speed or failures
                    (self.orchestrator.logs_dir / "sd_last_run.log").write_text(
                        sd_stdout.replace(chr(13), chr(10)), encoding="utf-8"
                    )
                except Exception:
                    pass
                if proc.returncode != 0 or not out_file.exists():
                    if not self._scripted_demo_enabled():
                        tail = "\n".join(sd_stdout.strip().splitlines()[-12:])
                        raise RuntimeError(
                            f"stable-diffusion.cpp failed (exit code {proc.returncode}) and produced no image. "
                            f"Last output:\n{tail}"
                        )
                    logger.warning(
                        f"sd.exe exited with code {proc.returncode}; scripted demo uses procedural synthesis."
                    )
                    self._synthesize_procedural_image(
                        prompt=prompt,
                        asset_type=asset_type,
                        out_file=out_file,
                        width=width,
                        height=height,
                        seed=seed,
                    )
                    engine_used = "procedural-synthesizer (sd.exe non-zero exit fallback)"
            else:
                self._synthesize_procedural_image(
                    prompt=prompt,
                    asset_type=asset_type,
                    out_file=out_file,
                    width=width,
                    height=height,
                    seed=seed,
                )
                engine_used = (
                    f"procedural-synthesizer ({img_model_info['filename']} weights not yet downloaded)"
                )

            # Seamless tiling pass if seamless_texture requested
            if asset_type == "seamless_texture" and out_file.exists():
                self._enforce_seamless_edges(out_file)

            # Optional local background removal (rembg / alpha-key)
            if transparent_bg and out_file.exists():
                bg_removal_method = self._remove_background(out_file)
                self._trim_to_content(out_file)

            # Optional PBR Normal / Roughness / Heightmap + Godot StandardMaterial3D .tres
            if generate_pbr_maps and out_file.exists():
                pbr_files = self.generate_pbr_maps_from_albedo(out_file)

        finally:
            # Step D: Restore LLM & KV Slot 0 unless batched with another consecutive 2D asset call
            if defer_llm_restore:
                swap_out_meta = {
                    "deferred": True,
                    "state": self.orchestrator.state.value,
                }
            else:
                swap_out_meta = await self.orchestrator.restore_llm_after_imagegen()

        # Step E: Closed-Loop Vision Self-Check (downscaled 512x512 base64 PNG preview)
        vision_preview_b64, preview_dims = self._create_vision_self_check_preview(
            out_file, max_edge=512
        )

        elapsed_sec = round(time.monotonic() - start_time, 2)
        res_path = self.to_res_path(out_file)

        return {
            "status": "success",
            "asset_type": asset_type,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "output_path": str(out_file),
            "res_path": res_path,
            "dimensions": {"width": width, "height": height},
            "steps": steps,
            "transparent_bg": transparent_bg,
            "bg_removal_method": bg_removal_method,
            "pbr_maps": pbr_files,
            "engine_used": engine_used,
            "sd_command": " ".join(sd_cmd),
            "elapsed_seconds": elapsed_sec,
            "vram_swap_telemetry": {
                "step_ab_acquire": swap_in_meta,
                "step_d_restore": swap_out_meta,
            },
            "closed_loop_vision_check": {
                "enabled": True,
                "preview_dimensions": preview_dims,
                "image_url": f"data:image/png;base64,{vision_preview_b64}",
                "description": (
                    f"Closed-Loop Vision Self-Check preview of generated {asset_type} at {res_path}. "
                    "Inspect visual quality, transparency, and framing before referencing in Godot code."
                ),
            },
            "sd_logs_tail": sd_stdout[-600:] if sd_stdout else "",
        }

    # -------------------------------------------------------------------------
    # Background Removal (rembg with automatic corner/chroma alpha-key fallback)
    # -------------------------------------------------------------------------
    @classmethod
    def _remove_background(cls, image_path: Path) -> str:
        img = Image.open(image_path).convert("RGBA")
        try:
            from rembg import remove  # type: ignore

            # Only call rembg if u2net model exists locally or can run without blocking network
            u2net_home = Path.home() / ".u2net"
            if u2net_home.exists() and any(u2net_home.glob("*.onnx")):
                out_img = remove(img)
                out_img.save(image_path, format="PNG")
                return "rembg (u2net ONNX)"
        except Exception as exc:
            logger.debug(f"rembg unavailable or offline ({exc}); using local alpha-key segmentation.")

        # Local background removal: key out the flat border colour, but only where it is
        # connected to the image edge, so same-coloured details inside the object survive.
        arr = np.array(img, dtype=np.float32)
        rgb = arr[:, :, :3]
        h, w = rgb.shape[:2]
        border = np.concatenate(
            [rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3), rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)],
            axis=0,
        )
        bg_color = np.median(border, axis=0)
        dist = np.linalg.norm(rgb - bg_color[None, None, :], axis=2)

        bg_like = dist < 60.0
        try:
            from scipy import ndimage  # type: ignore

            labels, _count = ndimage.label(bg_like)
            edge_labels = np.unique(
                np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
            )
            edge_labels = edge_labels[edge_labels != 0]
            background = np.isin(labels, edge_labels)
            # Grow one pixel so the anti-aliased fringe is feathered instead of left as a halo
            background_soft = ndimage.binary_dilation(background, iterations=2)
        except Exception:
            background = bg_like
            background_soft = bg_like

        # Fully transparent in the background, soft ramp (distance 25..70) on its fringe
        ramp = np.clip((dist - 25.0) / 45.0, 0.0, 1.0) * 255.0
        alpha = np.full((h, w), 255.0, dtype=np.float32)
        alpha[background_soft] = ramp[background_soft]
        alpha[background & (dist < 25.0)] = 0.0
        arr[:, :, 3] = np.minimum(arr[:, :, 3], alpha)
        out_img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        out_img.save(image_path, format="PNG")
        return "local border-connected colour key"

    @staticmethod
    def _trim_to_content(image_path: Path, padding_ratio: float = 0.06) -> None:
        """After background removal, crop to the object and scale it up to fill the canvas (kept square)."""
        try:
            img = Image.open(image_path).convert("RGBA")
            bbox = img.getchannel("A").point(lambda v: 255 if v > 16 else 0).getbbox()
            if not bbox:
                return
            w, h = img.size
            left, top, right, bottom = bbox
            side = max(right - left, bottom - top)
            if side >= min(w, h) * 0.85:
                return  # already fills the frame
            side = int(side * (1.0 + 2.0 * padding_ratio))
            cx, cy = (left + right) // 2, (top + bottom) // 2
            canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            crop_box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
            canvas.paste(img.crop(crop_box), (0, 0))
            canvas.resize((w, h), Image.LANCZOS).save(image_path, format="PNG")
        except Exception as exc:
            logger.debug(f"Content trim skipped: {exc}")

    # -------------------------------------------------------------------------
    # Sobel PBR Normal, Roughness, Heightmap + Godot 4 StandardMaterial3D .tres
    # -------------------------------------------------------------------------
    def generate_pbr_maps_from_albedo(
        self, albedo_path: Path, normal_strength: float = 3.0
    ) -> Dict[str, str]:
        """
        Generate matching PBR maps from an albedo texture via Sobel X/Y gradient filtering:
          - <stem>_normal.png   (OpenGL/Godot +Y tangent-space normal map)
          - <stem>_roughness.png (inverted micro-contrast roughness map)
          - <stem>_height.png   (normalized luminance heightmap)
          - <stem>_material.tres (Godot 4 StandardMaterial3D resource binding all maps)
        """
        img = Image.open(albedo_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # Luminance heightmap
        height = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # Compute Sobel gradients along X and Y (wrapping at borders for seamless textures)
        try:
            from scipy.ndimage import sobel  # type: ignore

            dx = sobel(height, axis=1, mode="wrap")
            dy = sobel(height, axis=0, mode="wrap")
        except Exception:
            dx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
            dy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)

        nx = -dx * normal_strength
        ny = -dy * normal_strength
        nz = np.ones_like(height)

        length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx /= length
        ny /= length
        nz /= length

        normal_rgb = np.stack(
            [
                np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255),
                np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255),
                np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255),
            ],
            axis=-1,
        ).astype(np.uint8)

        # Roughness map: higher roughness in crevices (low height) + micro-gradient variance
        grad_mag = np.sqrt(dx * dx + dy * dy)
        grad_norm = grad_mag / (np.max(grad_mag) + 1e-6)
        roughness = np.clip(0.35 + (1.0 - height) * 0.45 + grad_norm * 0.2, 0.05, 0.98)
        roughness_u8 = (roughness * 255.0).astype(np.uint8)

        height_min, height_max = float(np.min(height)), float(np.max(height))
        if height_max - height_min > 1e-5:
            height_norm = (height - height_min) / (height_max - height_min)
        else:
            height_norm = height
        height_u8 = (height_norm * 255.0).astype(np.uint8)

        stem = albedo_path.stem
        parent = albedo_path.parent
        normal_path = parent / f"{stem}_normal.png"
        roughness_path = parent / f"{stem}_roughness.png"
        height_path = parent / f"{stem}_height.png"
        tres_path = parent / f"{stem}_material.tres"

        Image.fromarray(normal_rgb, mode="RGB").save(normal_path, format="PNG")
        Image.fromarray(roughness_u8, mode="L").save(roughness_path, format="PNG")
        Image.fromarray(height_u8, mode="L").save(height_path, format="PNG")

        res_albedo = self.to_res_path(albedo_path)
        res_normal = self.to_res_path(normal_path)
        res_roughness = self.to_res_path(roughness_path)
        res_height = self.to_res_path(height_path)

        tres_content = (
            '[gd_resource type="StandardMaterial3D" load_steps=5 format=3]\n\n'
            f'[ext_resource type="Texture2D" path="{res_albedo}" id="1_albedo"]\n'
            f'[ext_resource type="Texture2D" path="{res_normal}" id="2_normal"]\n'
            f'[ext_resource type="Texture2D" path="{res_roughness}" id="3_roughness"]\n'
            f'[ext_resource type="Texture2D" path="{res_height}" id="4_height"]\n\n'
            "[resource]\n"
            'albedo_texture = ExtResource("1_albedo")\n'
            'roughness_texture = ExtResource("3_roughness")\n'
            "normal_enabled = true\n"
            f"normal_scale = {min(2.0, round(normal_strength / 2.0, 2))}\n"
            'normal_texture = ExtResource("2_normal")\n'
            "heightmap_enabled = true\n"
            "heightmap_scale = 0.05\n"
            'heightmap_texture = ExtResource("4_height")\n'
        )
        tres_path.write_text(tres_content, encoding="utf-8")

        return {
            "normal_map": self.to_res_path(normal_path),
            "normal_map_abs": str(normal_path),
            "roughness_map": self.to_res_path(roughness_path),
            "roughness_map_abs": str(roughness_path),
            "height_map": self.to_res_path(height_path),
            "height_map_abs": str(height_path),
            "godot_material_tres": self.to_res_path(tres_path),
            "godot_material_tres_abs": str(tres_path),
        }

    @staticmethod
    def _enforce_seamless_edges(image_path: Path, blend_margin: int = 48) -> None:
        """Cross-blend borders so tileable textures wrap without visible seams in Godot 4."""
        img = Image.open(image_path).convert("RGBA")
        arr = np.array(img, dtype=np.float32)
        h, w, _ = arr.shape
        margin = min(blend_margin, h // 4, w // 4)
        if margin < 4:
            return
        shifted = np.roll(np.roll(arr, h // 2, axis=0), w // 2, axis=1)
        y_idx = np.arange(h)[:, None]
        x_idx = np.arange(w)[None, :]
        dist_y = np.minimum(y_idx, h - 1 - y_idx)
        dist_x = np.minimum(x_idx, w - 1 - x_idx)
        edge_dist = np.minimum(dist_y, dist_x)
        weight = np.clip(edge_dist / float(margin), 0.0, 1.0)[:, :, None]
        blended = arr * weight + shifted * (1.0 - weight)
        Image.fromarray(blended.astype(np.uint8), mode="RGBA").save(image_path, format="PNG")

    @staticmethod
    def _create_vision_self_check_preview(
        image_path: Path, max_edge: int = 512
    ) -> Tuple[str, Dict[str, int]]:
        """Downscale generated asset to a compact PNG base64 string for VLM Closed-Loop Vision Self-Check."""
        img = Image.open(image_path).convert("RGBA")
        w, h = img.size
        scale = min(1.0, float(max_edge) / float(max(w, h, 1)))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, {"width": new_w, "height": new_h}

    @staticmethod
    def _synthesize_procedural_image(
        prompt: str,
        asset_type: str,
        out_file: Path,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
    ) -> None:
        """
        Deterministic procedural texture/sprite synthesizer used when FLUX GGUF weights
        are not yet downloaded, allowing full end-to-end testing of the VRAM swap,
        rembg alpha-keying, Sobel PBR normal/roughness maps, and VLM vision self-check.
        """
        digest = hashlib.sha256(f"{prompt}:{asset_type}:{seed}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

        base_r, base_g, base_b = int(digest[0]), int(digest[1]), int(digest[2])
        accent_r, accent_g, accent_b = int(digest[3]), int(digest[4]), int(digest[5])

        y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = x_grid / float(max(width, 1))
        ny = y_grid / float(max(height, 1))

        if asset_type == "seamless_texture":
            # Periodic trigonometric multi-octave pattern (naturally tileable!)
            tau = 2.0 * math.pi
            pattern = (
                0.45 * np.sin(nx * tau * 4.0) * np.cos(ny * tau * 4.0)
                + 0.30 * np.sin((nx + ny) * tau * 8.0)
                + 0.25 * np.cos((nx * 12.0 - ny * 12.0) * tau)
            )
            noise = rng.normal(0.0, 0.08, size=(height, width)).astype(np.float32)
            val = np.clip(0.5 + 0.38 * pattern + noise, 0.0, 1.0)
            rgb = np.stack(
                [
                    np.clip(45 + val * (140 + (base_r % 70)), 0, 255),
                    np.clip(45 + val * (140 + (base_g % 70)), 0, 255),
                    np.clip(45 + val * (140 + (base_b % 70)), 0, 255),
                ],
                axis=-1,
            ).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.8))
            img.save(out_file, format="PNG")
            return

        if asset_type == "skybox_concept":
            # Atmospheric gradient horizon with procedural sun/nebula
            horizon = np.clip(1.0 - np.abs(ny - 0.55) * 2.4, 0.0, 1.0)
            r = np.clip(20 + (1.0 - ny) * 60 + horizon * (120 + (base_r % 80)), 0, 255)
            g = np.clip(24 + (1.0 - ny) * 80 + horizon * (90 + (base_g % 80)), 0, 255)
            b = np.clip(48 + (1.0 - ny) * 140 + horizon * (60 + (base_b % 80)), 0, 255)
            rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(img)
            sun_x, sun_y = int(width * 0.5), int(height * 0.42)
            for rad in range(90, 10, -10):
                col = (255, min(255, 190 + rad // 2), min(255, 120 + rad))
                draw.ellipse([sun_x - rad, sun_y - rad, sun_x + rad, sun_y + rad], fill=col)
            img.save(out_file, format="PNG")
            return

        # Sprite or UI Icon: isolated uniform corner background + detailed geometric game emblem/character silhouette
        bg_col = (18, 22, 28, 255)
        img = Image.new("RGBA", (width, height), bg_col)
        draw = ImageDraw.Draw(img)

        cx, cy = width // 2, height // 2
        radius = int(min(width, height) * 0.34)

        primary = (
            max(70, min(235, base_r)),
            max(70, min(235, base_g)),
            max(70, min(235, base_b)),
            255,
        )
        secondary = (
            max(90, min(245, accent_r)),
            max(90, min(245, accent_g)),
            max(90, min(245, accent_b)),
            255,
        )

        # Outer octagon/shield silhouette
        pts = []
        num_vertices = 8 if asset_type == "ui_icon" else 6
        for i in range(num_vertices):
            ang = (2.0 * math.pi * i) / num_vertices - math.pi / 2.0
            r_mod = radius if i % 2 == 0 else int(radius * 0.86)
            pts.append((cx + int(math.cos(ang) * r_mod), cy + int(math.sin(ang) * r_mod)))
        draw.polygon(pts, fill=primary, outline=(240, 245, 255, 255))

        # Inner core detail
        inner_r = int(radius * 0.55)
        draw.ellipse(
            [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
            fill=secondary,
            outline=(255, 255, 255, 255),
            width=4,
        )
        core_r = int(radius * 0.25)
        draw.polygon(
            [
                (cx, cy - core_r),
                (cx + core_r, cy),
                (cx, cy + core_r),
                (cx - core_r, cy),
            ],
            fill=(250, 250, 255, 255),
        )
        img.save(out_file, format="PNG")
