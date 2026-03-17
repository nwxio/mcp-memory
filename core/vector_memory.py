from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .db import db, fetch_all, fetch_one, dumps, loads, row_get
from .embeddings import EmbeddingsError, embed_text
from .ids import new_id
from .redact import redact_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mono() -> float:
    return time.monotonic()


def _sha_ext_ok(path: Path, allow_set: set[str]) -> bool:
    ext = path.suffix.lower()
    if not ext:
        return False
    return ext in allow_set


def _is_probably_binary(b: bytes) -> bool:
    """Cheap binary detector.

    We don't want to embed compiled/binary assets.
    """
    if not b:
        return False
    if b"\x00" in b:
        return True
    # Heuristic: if too many non-text bytes early on, treat as binary.
    sample = b[:4096]
    bad = 0
    for x in sample:
        # allow common whitespace
        if x in (9, 10, 13):
            continue
        # allow printable ASCII
        if 32 <= x <= 126:
            continue
        bad += 1
    return (bad / max(1, len(sample))) > 0.20


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# Keep this local (mirrors memory_sqlite defaults) to avoid huge scans.
_DEFAULT_DENY: List[str] = [
    # Secrets (keep parity with memory_sqlite)
    ".env",
    "**/.env",
    ".env.local",
    ".env.development",
    ".env.test",
    ".env.production",
    ".ssh",
    "**/.ssh/**",
    "**/*.key",
    "**/*.pem",
    # VCS / deps / build artifacts
    ".git",
    "**/.git/**",
    ".venv",
    "**/.venv/**",
    "venv",
    "**/venv/**",
    "node_modules",
    "**/node_modules/**",
    "frontend/node_modules",
    "frontend/dist",
    "frontend/build",
    "__pycache__",
    "**/__pycache__/**",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".idea",
    ".vscode",
    # Large/binary-ish blobs and archives
    "**/*.db",
    "**/*.sqlite",
    "**/*.sqlite3",
    "**/*.zip",
    "**/*.tar",
    "**/*.gz",
    "**/*.7z",
    "**/*.bin",
    "**/*.so",
    "**/*.dylib",
    "**/*.dll",
    # Media (vectors are pointless here)
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.webp",
    "**/*.gif",
    "**/*.mp4",
    "**/*.mkv",
    "**/*.avi",
]


def _denied(rel: str, deny: List[str]) -> bool:
    """Best-effort deny matcher.

    Historically this module used a very cheap substring/prefix matcher.
    We keep that behavior for backward compatibility, but also support
    fnmatch-style globs ("**/*.pem", "**/.git/**") to mirror memory_sqlite.
    """
    import fnmatch

    rel = (rel or "").replace("\\", "/").lstrip("/")
    for pat in deny:
        pat = (pat or "").strip()
        if not pat:
            continue

        # Fast path: directory prefix
        if pat.endswith("/"):
            if rel.startswith(pat.lstrip("/")):
                return True
            continue

        # Fast path: extension filter
        if pat.startswith("*."):
            if rel.lower().endswith(pat[1:].lower()):
                return True
            continue

        p = pat.lstrip("/")
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, p):
            return True

        # Legacy substring match
        if p and p in rel:
            return True
    return False


def _chunk_text(text: str, *, chunk_chars: int, overlap: int) -> List[str]:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    if chunk_chars <= 100:
        chunk_chars = 100
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_chars:
        overlap = max(0, chunk_chars // 4)

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks


def _norm(v: List[float]) -> float:
    s = 0.0
    for x in v:
        s += float(x) * float(x)
    return math.sqrt(s) if s > 0.0 else 0.0


def _dot(a: List[float], b: List[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += float(x) * float(y)
    return s


async def _get_index_state(name: str) -> Dict[str, Any]:
    """Shared throttle/state storage via memory_index_state."""
    name = (name or "").strip()
    if not name:
        return {}
    try:
        async with db.connect() as conn:
            row = await fetch_one(
                conn,
                "SELECT last_scan_at, stats_json FROM memory_index_state WHERE name=?",
                (name,),
            )
    except Exception:
        return {}
    if not row:
        return {}
    try:
        stats = loads(row["stats_json"]) if row_get(row, "stats_json") else {}
    except Exception:
        stats = {}
    return {"name": name, "last_scan_at": row_get(row, "last_scan_at"), "stats": stats}


async def _set_index_state(
    name: str, *, last_scan_at: Optional[str], stats: Optional[Dict[str, Any]]
) -> None:
    name = (name or "").strip()
    if not name:
        return
    try:
        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO memory_index_state(name, last_scan_at, stats_json) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET last_scan_at=excluded.last_scan_at, stats_json=excluded.stats_json",
                (name, last_scan_at, dumps(stats or {})),
            )
            await conn.commit()
    except Exception:
        return


@dataclass
class VectorHit:
    path: str
    snippet: str
    score: float
    chunk_index: int


class VectorMemory:
    _STATE_NAME = "vector_ws"

    # Minimal schema for vector indexing/state.
    # Prevents 500s when DB schema is missing/old (e.g., after upgrade, interrupted init, or :memory: DB).
    _VECTOR_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS vector_files_meta (
      path TEXT PRIMARY KEY,
      mtime REAL NOT NULL,
      size INTEGER NOT NULL,
      sha256 TEXT,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS vector_chunks (
      id TEXT PRIMARY KEY,
      path TEXT NOT NULL,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      embedding_json TEXT NOT NULL,
      embedding_dim INTEGER NOT NULL,
      embedding_norm REAL NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_chunks_path_idx ON vector_chunks(path, chunk_index);
    CREATE INDEX IF NOT EXISTS idx_vector_chunks_path ON vector_chunks(path);
    CREATE TABLE IF NOT EXISTS memory_index_state (
      name TEXT PRIMARY KEY,
      last_scan_at TEXT,
      stats_json TEXT
    );
    """

    async def _ensure_tables(self, conn) -> None:
        try:
            await conn.executescript(self._VECTOR_SCHEMA_SQL)
        except Exception:
            # DB may be readonly/unavailable; callers should degrade gracefully.
            pass

    async def chunk_count(self) -> int:
        sql = "SELECT COUNT(1) AS c FROM vector_chunks"
        try:
            async with db.connect() as conn:
                await self._ensure_tables(conn)
                row = await fetch_one(conn, sql)
        except Exception:
            return 0
        if not row:
            return 0
        try:
            return int(row["c"])
        except Exception:
            return 0

    async def meta_count(self) -> int:
        sql = "SELECT COUNT(1) AS c FROM vector_files_meta"
        try:
            async with db.connect() as conn:
                await self._ensure_tables(conn)
                row = await fetch_one(conn, sql)
        except Exception:
            return 0
        if not row:
            return 0
        try:
            return int(row["c"])
        except Exception:
            return 0

    async def status(self) -> Dict[str, Any]:
        # Always return a 200-friendly payload; UI polls this endpoint.
        try:
            st = await _get_index_state(self._STATE_NAME)
        except Exception as e:
            st = {"error": str(e)}
        try:
            cc = await self.chunk_count()
            mc = await self.meta_count()
            return {"chunk_count": cc, "meta_count": mc, "index_state": st, "available": True}
        except Exception as e:
            return {
                "chunk_count": 0,
                "meta_count": 0,
                "index_state": st,
                "available": False,
                "error": str(e),
            }

    async def has_index(self) -> bool:
        return (await self.chunk_count()) > 0

    async def clear_index(self) -> Dict[str, Any]:
        try:
            async with db.connect() as conn:
                await self._ensure_tables(conn)
                await conn.execute("DELETE FROM vector_chunks")
                await conn.execute("DELETE FROM vector_files_meta")
                await conn.execute(
                    "DELETE FROM memory_index_state WHERE name=?", (self._STATE_NAME,)
                )
                await conn.commit()
            return {"status": "ok"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def index_workspace_incremental(
        self,
        *,
        root_path: str,
        allow_ext: Optional[List[str]] = None,
        deny_patterns: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_file_bytes: Optional[int] = None,
        chunk_chars: Optional[int] = None,
        overlap: Optional[int] = None,
        max_seconds: Optional[float] = None,
        min_interval_seconds: Optional[float] = None,
        force: bool = False,
        prune_missing: bool = False,
    ) -> Dict[str, Any]:
        """Incrementally index workspace files into vector_chunks.

        Uses vector_files_meta (path, mtime, size, sha256) to skip unchanged files.
        Throttled via memory_index_state(name='vector_ws') to avoid repeated scans.

        If force=True, ignores meta and reindexes eligible files.
        If prune_missing=True and scan completes without hitting limits, removes stale rows.
        """
        ws = Path(root_path).expanduser().resolve()
        if not ws.exists():
            return {"status": "failed", "error": f"workspace does not exist: {ws}"}

        prov = (
            str(getattr(settings, "embeddings_provider", "disabled") or "disabled").strip().lower()
        )
        if not prov or prov == "disabled":
            return {
                "status": "failed",
                "error": "embeddings_disabled",
                "detail": "Set OMNIMIND_EMBEDDINGS_PROVIDER to enable vector indexing/search",
            }

        allow_ext = allow_ext or [
            e.strip().lower() for e in (settings.vector_allow_ext or "").split(",") if e.strip()
        ]
        allow_set = set(allow_ext)
        deny = deny_patterns or _DEFAULT_DENY

        max_files = int(
            max_files
            if max_files is not None
            else getattr(settings, "vector_ws_max_files", settings.vector_max_files)
        )
        max_file_bytes = int(
            max_file_bytes if max_file_bytes is not None else settings.vector_max_file_bytes
        )
        chunk_chars = int(chunk_chars if chunk_chars is not None else settings.vector_chunk_chars)
        overlap = int(overlap if overlap is not None else settings.vector_chunk_overlap)
        max_seconds = float(
            max_seconds
            if max_seconds is not None
            else getattr(settings, "vector_ws_max_seconds", 8.0)
        )
        min_interval_seconds = float(
            min_interval_seconds
            if min_interval_seconds is not None
            else getattr(settings, "vector_ws_min_interval_seconds", 300.0)
        )

        indexed = 0
        skipped = 0
        unchanged = 0
        errors = 0
        total_chunks = 0
        started_at = _utc_now()
        t0 = _mono()
        hit_limits = False
        seen: set[str] = set()

        # Throttle unless forced.
        if not force and min_interval_seconds > 0:
            st = await _get_index_state(self._STATE_NAME)
            last = (st.get("stats") or {}).get("last_scan_mono")
            try:
                last_mono = float(last)
            except Exception:
                last_mono = 0.0
            if last_mono > 0 and (_mono() - last_mono) < float(min_interval_seconds):
                return {
                    "status": "ok",
                    "root": str(ws),
                    "started_at": started_at,
                    "throttled": True,
                    "indexed": 0,
                    "unchanged": 0,
                    "skipped": 0,
                    "errors": 0,
                    "total_chunks": 0,
                    "hit_limits": False,
                }

        async with db.connect() as conn:
            await self._ensure_tables(conn)
            ops = 0
            for p in ws.rglob("*"):
                if p.is_dir():
                    continue

                rel = str(p.relative_to(ws)).replace("\\", "/")
                if _denied(rel, deny):
                    skipped += 1
                    continue
                if not _sha_ext_ok(p, allow_set):
                    skipped += 1
                    continue

                if (indexed + skipped + unchanged + errors) >= max_files:
                    hit_limits = True
                    break
                if max_seconds > 0 and (_mono() - t0) >= max_seconds:
                    hit_limits = True
                    break

                try:
                    st = p.stat()
                except Exception:
                    errors += 1
                    continue

                if st.st_size <= 0 or st.st_size > max_file_bytes:
                    skipped += 1
                    continue

                seen.add(rel)

                # Check meta for unchanged files.
                if not force:
                    row = await fetch_one(
                        conn, "SELECT mtime, size FROM vector_files_meta WHERE path=?", (rel,)
                    )
                    if row:
                        try:
                            mtime = float(row["mtime"]) if row["mtime"] is not None else 0.0
                            size = int(row["size"]) if row["size"] is not None else -1
                        except Exception:
                            mtime = 0.0
                            size = -1
                        if abs(float(st.st_mtime) - mtime) < 1e-6 and int(st.st_size) == size:
                            unchanged += 1
                            continue

                # Read
                try:
                    b = p.read_bytes()
                    if _is_probably_binary(b):
                        skipped += 1
                        continue
                    text = b.decode("utf-8", errors="replace")
                    if getattr(settings, "workspace_redact_secrets", True):
                        text = redact_text(text)
                except Exception:
                    errors += 1
                    continue

                chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
                if not chunks:
                    skipped += 1
                    continue

                # Keep file consistent: remove existing chunks then add.
                await conn.execute("DELETE FROM vector_chunks WHERE path = ?", (rel,))

                file_ok = True
                for ci, chunk in enumerate(chunks):
                    try:
                        try:
                            emb = await embed_text(chunk)
                        except EmbeddingsError as e:
                            return {
                                "status": "failed",
                                "error": "embeddings_unavailable",
                                "detail": str(e),
                            }
                        vec = emb.vector
                        nrm = _norm(vec)
                        await conn.execute(
                            "INSERT INTO vector_chunks (id, path, chunk_index, content, embedding_json, embedding_dim, embedding_norm, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET path=EXCLUDED.path, chunk_index=EXCLUDED.chunk_index, content=EXCLUDED.content, embedding_json=EXCLUDED.embedding_json, embedding_dim=EXCLUDED.embedding_dim, embedding_norm=EXCLUDED.embedding_norm, updated_at=EXCLUDED.updated_at",
                            (
                                new_id("vch"),
                                rel,
                                int(ci),
                                chunk,
                                dumps(vec),
                                int(emb.dim),
                                float(nrm),
                                _utc_now(),
                            ),
                        )
                        total_chunks += 1
                    except EmbeddingsError as e:
                        # Embeddings unavailable: stop indexing and return a clear error.
                        file_ok = False
                        errors += 1
                        return {
                            "status": "failed",
                            "root": str(ws),
                            "started_at": started_at,
                            "error": f"embeddings_error: {e}",
                            "indexed": indexed,
                            "unchanged": unchanged,
                            "skipped": skipped,
                            "errors": errors,
                            "total_chunks": total_chunks,
                            "hit_limits": hit_limits,
                        }
                    except Exception:
                        errors += 1
                        file_ok = False
                        continue

                if file_ok:
                    now = _utc_now()
                    sha = _sha256_bytes(b)
                    await conn.execute(
                        "INSERT INTO vector_files_meta(path, mtime, size, sha256, updated_at) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, sha256=excluded.sha256, updated_at=excluded.updated_at",
                        (rel, float(st.st_mtime), int(st.st_size), sha, now),
                    )
                    indexed += 1
                    ops += 1

                if ops >= 20:
                    await conn.commit()
                    ops = 0

            # Optional prune (only safe when scan completed)
            if prune_missing and not hit_limits:
                rows_meta = await fetch_all(conn, "SELECT path FROM vector_files_meta")
                stale = [str(r[0]) for r in rows_meta if str(r[0]) not in seen]
                if stale:
                    q = ",".join(["?"] * len(stale))
                    await conn.execute(
                        f"DELETE FROM vector_files_meta WHERE path IN ({q})", tuple(stale)
                    )
                    await conn.execute(
                        f"DELETE FROM vector_chunks WHERE path IN ({q})", tuple(stale)
                    )

            await conn.commit()

        stats = {
            "last_scan_mono": _mono(),
            "indexed": indexed,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
            "hit_limits": hit_limits,
            "max_files": max_files,
            "max_seconds": max_seconds,
            "force": bool(force),
            "prune_missing": bool(prune_missing),
        }
        await _set_index_state(self._STATE_NAME, last_scan_at=_utc_now(), stats=stats)

        return {
            "status": "ok",
            "root": str(ws),
            "started_at": started_at,
            "throttled": False,
            "indexed": indexed,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
            "total_chunks": total_chunks,
            "hit_limits": hit_limits,
        }

    async def index_workspace(
        self,
        *,
        root_path: str,
        allow_ext: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_file_bytes: Optional[int] = None,
        chunk_chars: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full (forced) rebuild."""
        return await self.index_workspace_incremental(
            root_path=root_path,
            allow_ext=allow_ext,
            max_files=max_files,
            max_file_bytes=max_file_bytes,
            chunk_chars=chunk_chars,
            overlap=overlap,
            force=True,
            prune_missing=True,
            min_interval_seconds=0.0,
        )

    async def index_workspace_paths(
        self,
        *,
        root_path: str,
        paths: List[str],
        allow_ext: Optional[List[str]] = None,
        max_file_bytes: Optional[int] = None,
        max_files: Optional[int] = None,
        chunk_chars: Optional[int] = None,
        overlap: Optional[int] = None,
        max_seconds: Optional[float] = None,
        deny_patterns: Optional[List[str]] = None,
        force: bool = True,
    ) -> Dict[str, Any]:
        """Index a specific list of workspace-relative files into vector memory.

        This is intended for post-write updates: small, focused, and fast.
        """
        ws = Path(root_path).expanduser().resolve()
        if not ws.exists():
            return {"status": "failed", "error": f"workspace does not exist: {ws}", "root": str(ws)}

        prov = (
            str(getattr(settings, "embeddings_provider", "disabled") or "disabled").strip().lower()
        )
        if not prov or prov == "disabled":
            return {
                "status": "failed",
                "error": "embeddings_disabled",
                "detail": "Set OMNIMIND_EMBEDDINGS_PROVIDER to enable vector indexing/search",
                "root": str(ws),
            }

        allow_ext = allow_ext or [
            e.strip().lower() for e in (settings.vector_allow_ext or "").split(",") if e.strip()
        ]
        allow_set = set(allow_ext)
        deny = deny_patterns or _DEFAULT_DENY

        max_files_i = int(
            max_files
            if max_files is not None
            else getattr(settings, "workspace_index_on_write_max_files", 25)
        )
        max_file_bytes_i = int(
            max_file_bytes if max_file_bytes is not None else settings.vector_max_file_bytes
        )
        chunk_chars_i = int(chunk_chars if chunk_chars is not None else settings.vector_chunk_chars)
        overlap_i = int(overlap if overlap is not None else settings.vector_chunk_overlap)
        max_seconds_f = float(
            max_seconds
            if max_seconds is not None
            else getattr(settings, "workspace_index_on_write_vectors_max_seconds", 5.0)
        )

        started_at = _utc_now()
        t0 = _mono()
        indexed = 0
        skipped = 0
        unchanged = 0
        errors = 0
        total_chunks = 0
        hit_limits = False

        # Normalize + de-dupe while preserving order.
        seen: set[str] = set()
        norm_paths: List[str] = []
        for p in paths or []:
            rel = str(p or "").lstrip("/").replace("\\", "/")
            if not rel or rel in seen:
                continue
            seen.add(rel)
            norm_paths.append(rel)

        async with db.connect() as conn:
            await self._ensure_tables(conn)
            ops = 0
            for rel in norm_paths:
                if indexed + skipped + unchanged + errors >= max_files_i:
                    hit_limits = True
                    break
                if max_seconds_f > 0 and (_mono() - t0) >= max_seconds_f:
                    hit_limits = True
                    break

                if _denied(rel, deny):
                    skipped += 1
                    continue
                abs_p = (ws / rel).resolve()
                if not str(abs_p).startswith(str(ws)):
                    skipped += 1
                    continue
                if not abs_p.exists() or not abs_p.is_file():
                    skipped += 1
                    continue
                if not _sha_ext_ok(abs_p, allow_set):
                    skipped += 1
                    continue

                try:
                    st = abs_p.stat()
                except Exception:
                    errors += 1
                    continue

                if st.st_size <= 0 or st.st_size > max_file_bytes_i:
                    skipped += 1
                    continue

                if not force:
                    row = await fetch_one(
                        conn, "SELECT mtime, size FROM vector_files_meta WHERE path=?", (rel,)
                    )
                    if row:
                        try:
                            mtime = float(row["mtime"]) if row["mtime"] is not None else 0.0
                            size = int(row["size"]) if row["size"] is not None else -1
                        except Exception:
                            mtime = 0.0
                            size = -1
                        if abs(float(st.st_mtime) - mtime) < 1e-6 and int(st.st_size) == size:
                            unchanged += 1
                            continue

                try:
                    b = abs_p.read_bytes()
                    if _is_probably_binary(b):
                        skipped += 1
                        continue
                    text = b.decode("utf-8", errors="replace")
                    if getattr(settings, "workspace_redact_secrets", True):
                        text = redact_text(text)
                except Exception:
                    errors += 1
                    continue

                chunks = _chunk_text(text, chunk_chars=chunk_chars_i, overlap=overlap_i)
                if not chunks:
                    skipped += 1
                    continue

                # Keep file consistent: remove existing chunks then add.
                await conn.execute("DELETE FROM vector_chunks WHERE path = ?", (rel,))

                file_ok = True
                for ci, chunk in enumerate(chunks):
                    try:
                        try:
                            emb = await embed_text(chunk)
                        except EmbeddingsError as e:
                            return {
                                "status": "failed",
                                "error": "embeddings_unavailable",
                                "detail": str(e),
                            }
                        vec = emb.vector
                        nrm = _norm(vec)
                        await conn.execute(
                            "INSERT INTO vector_chunks (id, path, chunk_index, content, embedding_json, embedding_dim, embedding_norm, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET path=EXCLUDED.path, chunk_index=EXCLUDED.chunk_index, content=EXCLUDED.content, embedding_json=EXCLUDED.embedding_json, embedding_dim=EXCLUDED.embedding_dim, embedding_norm=EXCLUDED.embedding_norm, updated_at=EXCLUDED.updated_at",
                            (
                                new_id("vch"),
                                rel,
                                int(ci),
                                chunk,
                                dumps(vec),
                                int(emb.dim),
                                float(nrm),
                                _utc_now(),
                            ),
                        )
                        total_chunks += 1
                    except EmbeddingsError:
                        # propagate (caller decides how to handle)
                        raise
                    except Exception:
                        errors += 1
                        file_ok = False
                        continue

                if file_ok:
                    now = _utc_now()
                    sha = _sha256_bytes(b)
                    await conn.execute(
                        "INSERT INTO vector_files_meta(path, mtime, size, sha256, updated_at) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, sha256=excluded.sha256, updated_at=excluded.updated_at",
                        (rel, float(st.st_mtime), int(st.st_size), sha, now),
                    )
                    indexed += 1
                    ops += 1
                else:
                    errors += 1

                if ops >= 10:
                    await conn.commit()
                    ops = 0

            await conn.commit()

        return {
            "status": "ok",
            "root": str(ws),
            "started_at": started_at,
            "indexed": indexed,
            "unchanged": unchanged,
            "skipped": skipped,
            "errors": errors,
            "total_chunks": total_chunks,
            "hit_limits": hit_limits,
        }

    async def index_workspace_if_needed(
        self, *, root_path: str, max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        # Backward-compatible: only build if empty.
        if await self.has_index():
            return {
                "status": "ok",
                "already_indexed": True,
                "indexed": 0,
                "skipped": 0,
                "errors": 0,
                "chunk_count": await self.chunk_count(),
            }

        cap = (
            max_files
            if max_files is not None
            else int(getattr(settings, "vector_autobuild_max_files_startup", 200))
        )
        res = await self.index_workspace(root_path=root_path, max_files=cap)
        res["already_indexed"] = False
        res["chunk_count"] = await self.chunk_count()
        return res

    async def semantic_search(self, query: str, *, limit: int = 5) -> List[VectorHit]:
        query = (query or "").strip()
        if not query:
            return []

        try:
            emb = await embed_text(query)
        except EmbeddingsError:
            return []
        q = emb.vector
        qn = _norm(q)
        if qn <= 0:
            return []

        cand = int(getattr(settings, "vector_search_candidate_limit", 2500) or 2500)
        if cand < 200:
            cand = 200
        # Note: PostgreSQL doesn't have rowid, use id for ordering
        sql = (
            "SELECT id, path, chunk_index, content, embedding_json, embedding_norm, embedding_dim "
            "FROM vector_chunks WHERE embedding_dim = ? ORDER BY id DESC LIMIT ?"
        )
        try:
            async with db.connect() as conn:
                await self._ensure_tables(conn)
                rows = await fetch_all(conn, sql, (int(emb.dim), int(cand)))
        except Exception:
            return []

        scored: List[Tuple[float, str, int, str]] = []
        for r in rows:
            try:
                dim = int(r["embedding_dim"])
                if dim != emb.dim:
                    continue
                vn = float(r["embedding_norm"]) if r["embedding_norm"] is not None else 0.0
                if vn <= 0:
                    continue
                vec = loads(r["embedding_json"])
                if not isinstance(vec, list):
                    continue
                vec = [float(x) for x in vec]
                score = _dot(q, vec) / (qn * vn)
                scored.append(
                    (float(score), str(r["path"]), int(r["chunk_index"]), str(r["content"]))
                )
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[VectorHit] = []
        for score, path, ci, content in scored[: max(1, int(limit))]:
            snip = content.strip().replace("\n", " ")
            if len(snip) > 240:
                snip = snip[:240] + "…"
            out.append(VectorHit(path=path, snippet=snip, score=float(score), chunk_index=ci))
        return out

    async def get_neighbor_text(
        self,
        *,
        path: str,
        chunk_index: int,
        radius: int = 1,
        max_chars: int = 1400,
    ) -> Dict[str, Any]:
        """Fetch chunk_index plus neighbors and format as a compact excerpt."""
        path = (path or "").strip()
        if not path:
            return {}
        try:
            ci = int(chunk_index)
        except Exception:
            return {}
        r = max(0, int(radius))
        lo = max(0, ci - r)
        hi = ci + r

        sql = (
            "SELECT chunk_index, content FROM vector_chunks "
            "WHERE path=? AND chunk_index BETWEEN ? AND ? "
            "ORDER BY chunk_index ASC"
        )
        try:
            async with db.connect() as conn:
                await self._ensure_tables(conn)
                rows = await fetch_all(conn, sql, (path, int(lo), int(hi)))
        except Exception:
            return {}

        if not rows:
            return {}

        parts: List[str] = []
        for row in rows:
            try:
                cii = int(row["chunk_index"])
            except Exception:
                cii = None
            content = str(row["content"] or "").strip()
            if not content:
                continue
            parts.append(f"--- chunk {cii} ---\n{content}")

        text = "\n\n".join(parts).strip()
        if not text:
            return {}
        mc = max(200, int(max_chars))
        if len(text) > mc:
            text = text[: mc - 1] + "…"
        return {"path": path, "chunk_index": ci, "range": [lo, hi], "text": text}


vector_memory = VectorMemory()
