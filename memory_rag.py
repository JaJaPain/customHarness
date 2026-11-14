"""
Zero-VRAM Local Pre-Compaction Memory Flush & On-Demand RAG Recall System (memory_rag.py).
- Uses CPU-only embeddings (BAAI/bge-small-en-v1.5 via fastembed / sentence-transformers on CPU,
  with a deterministic 384-d feature-hashed n-gram + BM25 hybrid fallback when offline)
  so it consumes 0 bytes of RTX 3060 VRAM.
- Stores chunks and vectors in ./cache/project_memory.db scoped per Godot project folder hash.
- Executes two-step Pre-Compaction Memory Flush at ~25,000 tokens before turn summarization:
    Step A: Prunes bulky `read_file` dumps -> `[Read <path> lines X-Y]` placeholders & indexes turns into SQLite.
    Step B: Runs deterministic fact extraction (temperature=0.1) to pull up to 8 durable Godot project
            facts/rules and appends deduplicated entries to `<project_root>/.harness/MEMORY.md`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np

logger = logging.getLogger("CustomHarness.MemoryRAG")


class ZeroVRAMCPUEmbedder:
    """
    CPU-only 384-dimensional text embedder (0 bytes of GPU VRAM).
    Attempts to load `BAAI/bge-small-en-v1.5` via `fastembed` (CPUExecutionProvider)
    or `sentence-transformers` (device="cpu") only if already cached locally on disk;
    otherwise uses a fast, deterministic 384-d signed feature-hashing (word + char-trigram)
    L2-normalized embedding on CPU so recall works offline with zero startup delay.
    """

    EMBED_DIM = 384
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    vram_bytes = 0

    def __init__(self) -> None:
        self._backend_name = "cpu-feature-hash-bge-384d"
        self.vram_bytes = 0
        self._model: Any = None
        self._lock = threading.Lock()
        self._try_load_cached_local_bge()

    def _try_load_cached_local_bge(self) -> None:
        """Load BAAI/bge-small-en-v1.5 on CPU if cached locally in ./models/bge-small-en-v1.5 (never blocks on network)."""
        try:
            local_bge_dir = Path(__file__).resolve().parent / "models" / "bge-small-en-v1.5"
            if not local_bge_dir.exists():
                return

            try:
                from fastembed import TextEmbedding  # type: ignore

                self._model = TextEmbedding(
                    model_name=self.MODEL_NAME,
                    cache_dir=str(local_bge_dir),
                    providers=["CPUExecutionProvider"],
                )
                self._backend_name = "fastembed (BAAI/bge-small-en-v1.5, CPUExecutionProvider, 0 VRAM)"
                logger.info(f"Initialized ZeroVRAMCPUEmbedder: {self._backend_name}")
                return
            except Exception:
                pass

            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(
                    str(local_bge_dir),
                    device="cpu",
                    local_files_only=True,
                )
                self._backend_name = "sentence-transformers (BAAI/bge-small-en-v1.5, device=cpu, 0 VRAM)"
                logger.info(f"Initialized ZeroVRAMCPUEmbedder: {self._backend_name}")
                return
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"Using deterministic CPU 384-d embedder ({exc})")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized (N, 384) float32 embeddings strictly on CPU."""
        if not texts:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)

        with self._lock:
            if self._model is not None:
                try:
                    if "fastembed" in self._backend_name:
                        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
                    else:
                        vecs = np.asarray(
                            self._model.encode(texts, device="cpu", normalize_embeddings=True),
                            dtype=np.float32,
                        )
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
                    return (vecs / norms).astype(np.float32)
                except Exception as exc:
                    logger.debug(f"CPU model embed fallback: {exc}")

        return np.vstack([self._hash_embed_single(t) for t in texts]).astype(np.float32)

    @classmethod
    def _hash_embed_single(cls, text: str) -> np.ndarray:
        """
        Deterministic 384-d signed Weinberger feature-hashing over word unigrams,
        bigrams, and character trigrams, L2-normalized on CPU.
        """
        vec = np.zeros(cls.EMBED_DIM, dtype=np.float32)
        cleaned = re.sub(r"[^a-z0-9_:/\.\-]+", " ", text.lower()).strip()
        tokens = [tok for tok in cleaned.split() if len(tok) >= 2]
        if not tokens:
            return vec

        features: List[Tuple[str, float]] = []
        for idx, tok in enumerate(tokens):
            # Boost Godot identifiers, node names, paths, and collision layers
            weight = 2.0 if any(c in tok for c in ("res://", "_", ".gd", ".tscn", ".glb", "layer", "mask")) else 1.0
            features.append((f"w:{tok}", weight))
            if idx + 1 < len(tokens):
                features.append((f"bi:{tok}_{tokens[idx + 1]}", 1.25))
            if len(tok) >= 4:
                for k in range(len(tok) - 2):
                    features.append((f"tri:{tok[k:k+3]}", 0.4))

        for feat, w in features:
            h = hashlib.md5(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:2], "little") % cls.EMBED_DIM
            sign = 1.0 if (h[2] & 1) == 0 else -1.0
            vec[bucket] += sign * w

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec


class ProjectMemoryRAG:
    """
    Zero-VRAM Local Pre-Compaction Memory Flush and On-Demand RAG Recall Engine.
    - SQLite database: `./cache/project_memory.db`
    - Scoped per Godot project root hash (`project_hash`)
    - Pinned durable rules file: `<project_root>/.harness/MEMORY.md`
    """

    PRE_COMPACTION_FLUSH_THRESHOLD_TOKENS = 25000
    MAX_FACTS_PER_FLUSH = 8

    # Regex patterns to detect bulky raw read_file dumps in tool responses or assistant/user turns
    READ_FILE_JSON_RE = re.compile(
        r'\{[^{}]*"path"\s*:\s*"([^"]+)"[^{}]*"res_path"\s*:\s*"([^"]+)"[^{}]*"start_line"\s*:\s*(\d+)\s*,\s*"end_line"\s*:\s*(\d+)[^{}]*"numbered_content"\s*:\s*"(?:\\.|[^"\\])*"[^{}]*\}',
        re.DOTALL,
    )
    RAW_SOURCE_BLOCK_RE = re.compile(
        r"--- Raw Source \(([^,\)]+),\s*line-numbered\)\s*---\n((?:\s*\d+\s*\|[^\n]*\n?)+)",
        re.MULTILINE,
    )
    NUMBERED_LINES_BLOCK_RE = re.compile(
        r"(?:^\s*\d+\s*\|[^\n]*\n){8,}",
        re.MULTILINE,
    )

    def __init__(
        self,
        db_path: str | Path,
        project_root: str | Path,
        llama_port: int = 8081,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root).resolve()
        self.llama_port = llama_port

        self._db_lock = threading.RLock()
        self.embedder = ZeroVRAMCPUEmbedder()
        self._init_sqlite_schema()
        self.ensure_memory_md_exists()

    # -------------------------------------------------------------------------
    # Project Scoping & Pinned .harness/MEMORY.md Management
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_project_hash(project_root: str | Path) -> str:
        norm = str(Path(project_root).resolve()).lower().replace("\\", "/")
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    @property
    def project_hash(self) -> str:
        return self.compute_project_hash(self.project_root)

    def set_project_root(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.ensure_memory_md_exists()

    @property
    def memory_md_path(self) -> Path:
        harness_dir = self.project_root / ".harness"
        harness_dir.mkdir(parents=True, exist_ok=True)
        return harness_dir / "MEMORY.md"

    def ensure_memory_md_exists(self) -> Path:
        path = self.memory_md_path
        if not path.exists():
            initial_md = (
                "# Godot 4 Project Durable Memory (.harness/MEMORY.md)\n"
                "<!-- Static pinned rules & architectural facts extracted before context compaction -->\n\n"
                "- **Engine Target**: Godot 4.3+ (`Forward Plus`), GDScript 2.0 (`@export`, `@onready`, `CharacterBody3D`, `Node3D`, `await`, `.instantiate()`).\n"
                "- **3D Coordinate & Physics Convention**: glTF `+Y` Up, `1.0m` metric scale, Layer 1 = World/Ground (`StaticBody3D`), Layer 2 = Player (`CharacterBody3D`).\n"
            )
            path.write_text(initial_md, encoding="utf-8")
        return path

    def read_pinned_memory_md(self) -> str:
        self.ensure_memory_md_exists()
        return self.memory_md_path.read_text(encoding="utf-8", errors="replace").strip()

    def build_static_system_prompt(self, base_system_prompt: str) -> str:
        """
        Combine the base System Prompt + pinned `.harness/MEMORY.md` at the very top of the prompt
        so llama-server prefix caching in VRAM/RAM is never invalidated.
        """
        marker = "\n\n=== PINNED PROJECT MEMORY (.harness/MEMORY.md) ===\n"
        clean_base = base_system_prompt.split(marker, 1)[0].strip()
        pinned_md = self.read_pinned_memory_md()
        return f"{clean_base}{marker}{pinned_md}"

    # -------------------------------------------------------------------------
    # SQLite Vector + Keyword Store (./cache/project_memory.db)
    # -------------------------------------------------------------------------
    def _init_sqlite_schema(self) -> None:
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_hash TEXT NOT NULL,
                        project_root TEXT NOT NULL,
                        chunk_hash TEXT NOT NULL,
                        role TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding_blob BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        UNIQUE(project_hash, chunk_hash)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_project_hash ON memory_chunks(project_hash)"
                )
                conn.commit()
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # Step A: Prune Bulky read_file Dumps & Archive Conversation Chunks
    # -------------------------------------------------------------------------
    @classmethod
    def prune_raw_read_file_dumps(cls, text: str) -> str:
        """
        Strip out bulky raw `read_file` dumps and line-numbered file dumps, replacing them with
        compact `[Read <path> lines X-Y]` placeholders since files already live on disk.
        """
        if not text:
            return ""

        # 1. Replace JSON read_file tool outputs containing `numbered_content`
        def _replace_json_read(match: re.Match) -> str:
            res_path = match.group(2) or match.group(1)
            start_l = match.group(3)
            end_l = match.group(4)
            return f"[Read {res_path} lines {start_l}-{end_l}]"

        pruned = cls.READ_FILE_JSON_RE.sub(_replace_json_read, text)

        # 2. Replace `--- Raw Source (filename, line-numbered) ---` blocks from multimodal ingest
        def _replace_raw_source(match: re.Match) -> str:
            fname = match.group(1).strip()
            block = match.group(2)
            line_nums = re.findall(r"^\s*(\d+)\s*\|", block, flags=re.MULTILINE)
            first_l = line_nums[0] if line_nums else "1"
            last_l = line_nums[-1] if line_nums else str(len(block.splitlines()))
            return f"[Read {fname} lines {first_l}-{last_l}]"

        pruned = cls.RAW_SOURCE_BLOCK_RE.sub(_replace_raw_source, pruned)

        # 3. Replace any remaining multi-line numbered raw dumps (>= 8 lines of `  12 | ...`)
        def _replace_numbered_block(match: re.Match) -> str:
            block = match.group(0)
            line_nums = re.findall(r"^\s*(\d+)\s*\|", block, flags=re.MULTILINE)
            first_l = line_nums[0] if line_nums else "1"
            last_l = line_nums[-1] if line_nums else str(len(line_nums))
            return f"[Read <file> lines {first_l}-{last_l}]\n"

        pruned = cls.NUMBERED_LINES_BLOCK_RE.sub(_replace_numbered_block, pruned)
        return pruned

    @classmethod
    def prune_messages_read_dumps(
        cls, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return a copy of messages with bulky raw `read_file` dumps replaced by `[Read <path> lines X-Y]`."""
        pruned_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content", "")
            if isinstance(content, str):
                msg_copy["content"] = cls.prune_raw_read_file_dumps(content)
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        p_copy = dict(part)
                        p_copy["text"] = cls.prune_raw_read_file_dumps(p_copy.get("text", ""))
                        new_parts.append(p_copy)
                    else:
                        new_parts.append(part)
                msg_copy["content"] = new_parts
            pruned_msgs.append(msg_copy)
        return pruned_msgs

    def index_conversation_turns(
        self,
        messages: List[Dict[str, Any]],
        source_type: str = "pre_compaction_flush",
    ) -> Dict[str, Any]:
        """
        Prune raw `read_file` dumps from turns, chunk the remaining conversation turns,
        debugging steps, and error fixes, compute CPU 384-d embeddings, and store in
        `./cache/project_memory.db` scoped to `self.project_hash`.
        """
        pruned_messages = self.prune_messages_read_dumps(messages)
        chunks_to_index: List[Tuple[str, str]] = []  # (role, chunk_text)

        for msg in pruned_messages:
            role = msg.get("role", "user")
            if role == "system":
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                text_str = "\n".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text_str = str(content)

            thinking = msg.get("thinking", "")
            full_turn_text = (
                f"[{role.upper()}]\n{text_str.strip()}"
                + (f"\n[Reasoning]: {thinking.strip()}" if thinking else "")
            ).strip()

            if len(full_turn_text) < 12:
                continue

            # Split long turns into ~650-character overlapping semantic chunks
            for chunk in self._chunk_text(full_turn_text, max_chars=650, overlap_chars=100):
                chunks_to_index.append((role, chunk))

        if not chunks_to_index:
            return {"indexed_chunks": 0, "project_hash": self.project_hash}

        texts = [c[1] for c in chunks_to_index]
        embeddings = self.embedder.embed_texts(texts)

        now = time.time()
        p_hash = self.project_hash
        p_root = str(self.project_root)
        inserted_count = 0

        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                for idx, (role, chunk_text) in enumerate(chunks_to_index):
                    c_hash = hashlib.sha256(f"{p_hash}:{chunk_text}".encode("utf-8")).hexdigest()[:24]
                    emb_blob = embeddings[idx].astype(np.float32).tobytes()
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_chunks
                        (project_hash, project_root, chunk_hash, role, source_type, content, embedding_blob, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (p_hash, p_root, c_hash, role, source_type, chunk_text, emb_blob, now),
                    )
                    if cur.rowcount > 0:
                        inserted_count += 1
                conn.commit()
            finally:
                conn.close()

        return {
            "indexed_chunks": inserted_count,
            "chunks_indexed": len(chunks_to_index),
            "total_candidate_chunks": len(chunks_to_index),
            "project_hash": p_hash,
            "embedder": self.embedder.backend_name,
            "vram_bytes_used": 0,
        }

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 650, overlap_chars: int = 100) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + max_chars)
            chunks.append(text[start:end].strip())
            if end >= n:
                break
            start = max(start + 1, end - overlap_chars)
        return [c for c in chunks if c]

    # -------------------------------------------------------------------------
    # Step B: Deterministic Durable Fact Extraction -> .harness/MEMORY.md
    # -------------------------------------------------------------------------
    async def extract_and_append_durable_facts(
        self,
        messages: List[Dict[str, Any]],
        llama_alive: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a quick, deterministic extraction pass (temperature=0.1) to pull up to 8 hard
        project rules/facts (collision layers, node paths, autoload names, Godot 4 API constraints,
        generated asset paths, exported properties) and append deduplicated entries to
        `<project_root>/.harness/MEMORY.md`.
        """
        pruned_messages = self.prune_messages_read_dumps(messages)
        combined_text = "\n".join(
            (
                m.get("content", "")
                if isinstance(m.get("content"), str)
                else " ".join(
                    p.get("text", "")
                    for p in m.get("content", [])
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            )
            for m in pruned_messages
            if m.get("role") != "system"
        )

        extracted_facts: List[str] = []

        # 1. If llama-server is running, run a fast deterministic extraction pass at temperature=0.1
        if llama_alive and combined_text.strip():
            llm_facts = await self._extract_facts_via_llama_server(combined_text)
            extracted_facts.extend(llm_facts)

        # 2. Deterministic rule & AST/regex fact extractor for Godot 4 conventions, paths, & debugging fixes
        deterministic_facts = self._extract_facts_deterministic(combined_text)
        for df in deterministic_facts:
            if df not in extracted_facts:
                extracted_facts.append(df)

        extracted_facts = extracted_facts[: self.MAX_FACTS_PER_FLUSH]
        appended_facts = self._append_deduplicated_facts_to_memory_md(extracted_facts)

        return {
            "memory_md_path": str(self.memory_md_path),
            "extracted_count": len(extracted_facts),
            "appended_count": len(appended_facts),
            "appended_facts": appended_facts,
            "vram_bytes_used": 0,
        }

    async def _extract_facts_via_llama_server(self, conversation_snippet: str) -> List[str]:
        url = f"http://127.0.0.1:{self.llama_port}/v1/chat/completions"
        prompt = (
            "Extract up to 8 durable Godot 4 architectural facts, node paths, collision layers/masks, "
            "autoload names, or API constraints from the following session transcript. "
            "Output ONLY concise markdown bullet points starting with '- '.\n\n"
            f"{conversation_snippet[-6000:]}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": "You extract concise, deterministic Godot 4 project facts."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 320,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"].get("content", "")
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                    bullets = [
                        line.strip()[2:].strip()
                        for line in content.splitlines()
                        if line.strip().startswith("- ")
                    ]
                    return [b for b in bullets if len(b) > 8][: self.MAX_FACTS_PER_FLUSH]
        except Exception as exc:
            logger.debug(f"LLM fact extraction skipped ({exc}); using deterministic extractor.")
        return []

    @classmethod
    def _extract_facts_deterministic(cls, text: str) -> List[str]:
        """Extract up to 8 hard Godot 4 facts (res:// paths, collision layers, @export vars, signals, nodes)."""
        facts: List[str] = []

        # 1. res:// asset/script/scene paths
        res_paths = sorted(set(re.findall(r"res://[A-Za-z0-9_/\.\-]+\.[a-z0-9]+", text)))
        if res_paths:
            facts.append(f"**Project Resources**: {', '.join(f'`{p}`' for p in res_paths[:6])}")

        # 2. Collision layer / mask rules
        col_matches = re.findall(
            r"(?:collision_layer|layer)\s*=\s*(\d+).*?(?:collision_mask|mask)\s*=\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if col_matches:
            uniq_cols = sorted(set(col_matches))
            col_str = ", ".join(f"layer={l}/mask={m}" for l, m in uniq_cols[:4])
            facts.append(f"**Collision Configuration**: Verified physics bindings ({col_str}).")

        # 3. @export variables
        exports = re.findall(r"(@export\s+var\s+[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_]+(?:\s*=\s*[^\n]+)?)", text)
        if exports:
            uniq_exp = sorted(set(e.strip() for e in exports))
            facts.append(f"**Exported Inspector Properties**: {', '.join(f'`{e}`' for e in uniq_exp[:4])}")

        # 4. Signals
        signals = re.findall(r"\bsignal\s+([a-zA-Z0-9_]+(?:\([^\)]*\))?)", text)
        if signals:
            uniq_sig = sorted(set(signals))
            facts.append(f"**Declared Signals**: {', '.join(f'`{s}`' for s in uniq_sig[:5])}")

        # 5. Blender Godot collision suffixes (-col, -colonly, -convcol)
        col_sfx = re.findall(r"\b([A-Za-z0-9_]+-(?:col|colonly|convcol|noimp|navmesh))\b", text)
        if col_sfx:
            uniq_sfx = sorted(set(col_sfx))
            facts.append(f"**3D Collision Import Nodes**: {', '.join(f'`{s}`' for s in uniq_sfx[:4])}")

        # 6. Explicit architectural decisions / debugging notes mentioned in conversation
        for pattern, label in (
            (r"(?:autoload|singleton)[:\s]+([A-Za-z0-9_]+)", "Autoload Singleton"),
            (r"class_name\s+([A-Za-z0-9_]+)", "Registered GDScript Class"),
        ):
            found = sorted(set(re.findall(pattern, text)))
            if found:
                facts.append(f"**{label}**: {', '.join(f'`{x}`' for x in found[:4])}")

        return facts[: cls.MAX_FACTS_PER_FLUSH]

    def _append_deduplicated_facts_to_memory_md(self, facts: List[str]) -> List[str]:
        """Append new facts to `.harness/MEMORY.md` only if not already present."""
        self.ensure_memory_md_exists()
        existing_text = self.memory_md_path.read_text(encoding="utf-8", errors="replace")
        existing_lines_norm = {
            re.sub(r"\s+", " ", line.strip().lstrip("-* ").lower())
            for line in existing_text.splitlines()
            if line.strip()
        }

        appended: List[str] = []
        for fact in facts:
            clean_fact = fact.strip().lstrip("-* ").strip()
            if not clean_fact:
                continue
            norm_fact = re.sub(r"\s+", " ", clean_fact.lower())
            if norm_fact not in existing_lines_norm:
                existing_lines_norm.add(norm_fact)
                appended.append(clean_fact)

        if appended:
            new_block = "\n".join(f"- {f}" for f in appended) + "\n"
            with open(self.memory_md_path, "a", encoding="utf-8") as mf:
                if not existing_text.endswith("\n"):
                    mf.write("\n")
                mf.write(new_block)

        return appended

    # -------------------------------------------------------------------------
    # Full Pre-Compaction Memory Flush Orchestration (Step A + Step B)
    # -------------------------------------------------------------------------
    async def run_pre_compaction_flush(
        self,
        messages: List[Dict[str, Any]],
        llama_alive: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the two-step Pre-Compaction Memory Flush before summarizing/discarding old turns:
          - Step A: Strip bulky `read_file` dumps (`[Read <path> lines X-Y]`), chunk turns,
                    and index into `./cache/project_memory.db` on CPU (0 VRAM).
          - Step B: Extract up to 8 durable facts (`temperature=0.1`) and append deduplicated
                    entries to `.harness/MEMORY.md`.
        Returns `(pruned_messages, flush_report)`.
        """
        pruned_messages = self.prune_messages_read_dumps(messages)
        step_a_report = self.index_conversation_turns(
            pruned_messages, source_type="pre_compaction_flush"
        )
        step_b_report = await self.extract_and_append_durable_facts(
            pruned_messages, llama_alive=llama_alive
        )

        # Refresh static system message [0] with updated `.harness/MEMORY.md` since compaction
        # is about to rewrite history anyway
        if pruned_messages and pruned_messages[0].get("role") == "system":
            pruned_messages[0] = dict(pruned_messages[0])
            pruned_messages[0]["content"] = self.build_static_system_prompt(
                str(pruned_messages[0].get("content", ""))
            )

        flush_report = {
            "triggered": True,
            "project_hash": self.project_hash,
            "step_a_archive": step_a_report,
            "step_a_prune_and_archive": step_a_report,
            "step_b_facts": step_b_report,
            "step_b_durable_facts": step_b_report,
        }
        logger.info(
            f"[Pre-Compaction Memory Flush] Indexed {step_a_report['indexed_chunks']} chunks into "
            f"project_memory.db and appended {step_b_report['appended_count']} durable facts to .harness/MEMORY.md"
        )
        return pruned_messages, flush_report

    # -------------------------------------------------------------------------
    # On-Demand Recall (`recall_memory` Tool Implementation)
    # -------------------------------------------------------------------------
    def recall_memory(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Search `./cache/project_memory.db` (scoped to `self.project_hash`) using hybrid
        CPU cosine similarity (384-d BGE / feature-hash) + keyword overlap scoring.
        Uses 0 bytes of GPU VRAM.
        """
        top_k = max(1, min(20, int(top_k)))
        p_hash = self.project_hash
        q_vec = self.embedder.embed_texts([query])[0]

        q_tokens = {
            tok
            for tok in re.sub(r"[^a-z0-9_:/\.\-]+", " ", query.lower()).split()
            if len(tok) >= 2
        }

        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                rows = conn.execute(
                    """
                    SELECT id, role, source_type, content, embedding_blob, created_at
                    FROM memory_chunks
                    WHERE project_hash = ?
                    ORDER BY id DESC
                    LIMIT 2000
                    """,
                    (p_hash,),
                ).fetchall()
            finally:
                conn.close()

        scored_results: List[Dict[str, Any]] = []
        for row_id, role, source_type, content, emb_blob, created_at in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if emb.shape[0] == self.embedder.EMBED_DIM:
                cosine_sim = float(np.dot(q_vec, emb))
            else:
                cosine_sim = 0.0

            # Keyword BM25-style overlap boost
            content_lower = content.lower()
            kw_hits = sum(1 for qt in q_tokens if qt in content_lower)
            kw_score = (kw_hits / float(max(len(q_tokens), 1))) * 0.35
            combined_score = round(cosine_sim + kw_score, 4)

            scored_results.append(
                {
                    "chunk_id": row_id,
                    "score": combined_score,
                    "cosine_similarity": round(cosine_sim, 4),
                    "role": role,
                    "source_type": source_type,
                    "content": content,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at)),
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        top_matches = scored_results[:top_k]

        snippet_lines = [
            f"[RECALLED PROJECT MEMORY — project_hash={p_hash} | 0-VRAM CPU RAG (`{self.embedder.backend_name}`)]"
        ]
        for idx, m in enumerate(top_matches, start=1):
            snippet_lines.append(
                f"{idx}. (score={m['score']}, role={m['role']}, {m['created_at']}) {m['content']}"
            )
        formatted_injection = "\n".join(snippet_lines)

        return {
            "status": "success",
            "query": query,
            "project_hash": p_hash,
            "vram_used_bytes": 0,
            "vram_bytes_used": 0,
            "embedder": self.embedder.backend_name,
            "total_indexed_chunks": len(rows),
            "retrieved_count": len(top_matches),
            "top_k": top_k,
            "matches": top_matches,
            "formatted_injection": formatted_injection,
            "pinned_memory_md_excerpt": self.read_pinned_memory_md()[:1200],
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        p_hash = self.project_hash
        with self._db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM memory_chunks WHERE project_hash = ?",
                    (p_hash,),
                ).fetchone()
                chunk_count = int(count_row[0]) if count_row else 0
            finally:
                conn.close()

        return {
            "db_path": str(self.db_path),
            "project_hash": p_hash,
            "memory_md_path": str(self.memory_md_path),
            "indexed_chunks_count": chunk_count,
            "embedder": self.embedder.backend_name,
            "vram_bytes": 0,
            "flush_threshold_tokens": self.PRE_COMPACTION_FLUSH_THRESHOLD_TOKENS,
        }
