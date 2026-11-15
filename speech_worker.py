"""
Speech-to-text for the prompt composer's mic button.

Runs Whisper (small.en by default) through faster-whisper on the CPU, so it never competes with the
chat / image / audio models for VRAM and can be used while a turn is running. The model (~460 MB)
is downloaded into models/whisper on first use and then kept in memory.

Environment overrides:
  HARNESS_WHISPER_MODEL   model name (default "small.en"; e.g. "base.en" for a slower CPU)
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("CustomHarness.Speech")

# Primes Whisper with the vocabulary it will hear, so project terms come out spelled correctly.
VOCABULARY_HINT = (
    "Godot 4, GDScript, tscn, tres, glb, res://, project.godot, CharacterBody3D, Area3D, Node3D, "
    "AudioStreamPlayer, autoload, signal, HUD, sprite, shader, Blender, main scene, main menu."
)


class SpeechUnavailableError(RuntimeError):
    """Speech-to-text cannot run; the message says what to do about it."""


class SpeechToText:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = os.environ.get("HARNESS_WHISPER_MODEL", "small.en").strip() or "small.en"
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @staticmethod
    def _package_installed() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except Exception:
            return False

    def _model_downloaded(self) -> bool:
        return self.model_dir.exists() and any(self.model_dir.rglob("model.bin"))

    def status(self) -> Dict[str, Any]:
        return {
            "installed": self._package_installed(),
            "model": self.model_name,
            "model_downloaded": self._model_downloaded(),
            "loaded": self._model is not None,
        }

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise SpeechUnavailableError(
                    "Speech-to-text needs the faster-whisper package. In the harness folder run: "
                    ".venv\\Scripts\\python.exe -m pip install faster-whisper"
                ) from exc
            self.model_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=max(2, min(8, (os.cpu_count() or 4) // 2)),
                    download_root=str(self.model_dir),
                )
            except Exception as exc:
                raise SpeechUnavailableError(
                    f"Could not load the Whisper model '{self.model_name}' (it is downloaded on first use, "
                    f"which needs an internet connection): {type(exc).__name__}: {exc}"
                ) from exc
            logger.info(f"Whisper {self.model_name} ready on CPU in {time.time() - t0:.1f}s")
            return self._model

    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Transcribe a browser recording (webm/ogg/wav/mp3...). Returns {"text", "seconds", "audio_seconds", "model"}."""
        model = self._ensure_model()
        t0 = time.time()
        with self._run_lock:
            segments, info = model.transcribe(
                io.BytesIO(audio_bytes),
                language="en" if self.model_name.endswith(".en") else None,
                beam_size=5,
                vad_filter=True,
                initial_prompt=VOCABULARY_HINT,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        return {
            "text": text,
            "seconds": round(time.time() - t0, 2),
            "audio_seconds": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
            "model": self.model_name,
        }
