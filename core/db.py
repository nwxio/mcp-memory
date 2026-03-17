from __future__ import annotations

import asyncio
import json
import re
import warnings
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import quote

from .config import settings

# aiosqlite is preferred (true async, uses a worker thread internally).
# For dev environments where deps are incomplete (or selfcheck runs outside Docker),
# we provide a small compatibility shim based on sqlite3 + asyncio.to_thread.
try:  # pragma: no cover
    import aiosqlite  # type: ignore
except Exception:  # pragma: no cover
    import sqlite3
    from dataclasses import dataclass as _dataclass
    from typing import Any as _Any

    _Row = sqlite3.Row

    @_dataclass
    class _Cursor:
        _cur: sqlite3.Cursor
        _lock: asyncio.Lock

        @property
        def rowcount(self) -> int:
            # sqlite3.Cursor provides rowcount; expose it for compatibility with aiosqlite.
            return int(getattr(self._cur, "rowcount", -1))

        async def fetchall(self) -> list[_Any]:
            async with self._lock:
                return await asyncio.to_thread(self._cur.fetchall)

        async def fetchone(self) -> _Any:
            async with self._lock:
                return await asyncio.to_thread(self._cur.fetchone)

        async def close(self) -> None:
            async with self._lock:
                await asyncio.to_thread(self._cur.close)

    class _Connection:
        def __init__(self, path: str):
            self._lock = asyncio.Lock()
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._row_factory = None
            self._conn.row_factory = None

        @property
        def row_factory(self):  # type: ignore
            return self._row_factory

        @row_factory.setter
        def row_factory(self, v):  # type: ignore
            self._row_factory = v
            self._conn.row_factory = v

        async def execute(self, sql: str, params: Sequence[_Any] | None = None) -> _Cursor:
            if params is None:
                params = ()
            async with self._lock:
                cur = await asyncio.to_thread(self._conn.execute, sql, params)
                return _Cursor(cur, self._lock)

        async def execute_fetchall(
            self, sql: str, params: Sequence[_Any] | None = None
        ) -> list[_Any]:
            cur = await self.execute(sql, params)
            try:
                return await cur.fetchall()
            finally:
                await cur.close()

        async def executescript(self, script: str) -> None:
            async with self._lock:
                await asyncio.to_thread(self._conn.executescript, script)

        async def commit(self) -> None:
            async with self._lock:
                await asyncio.to_thread(self._conn.commit)

        async def close(self) -> None:
            async with self._lock:
                await asyncio.to_thread(self._conn.close)

        async def __aenter__(self) -> "_Connection":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            await self.close()

    # Expose shim with the same attributes used by this module.
    class _AioSqliteShim:  # pragma: no cover
        Row = _Row

        @staticmethod
        def connect(path: str) -> _Connection:
            # IMPORTANT: aiosqlite.connect returns an async context manager object.
            # Our shim returns a Connection that is itself an async context manager.
            return _Connection(path)

    aiosqlite = _AioSqliteShim()  # type: ignore

try:  # pragma: no cover
    import psycopg2  # type: ignore
    from psycopg2 import extras as psycopg2_extras  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore
    psycopg2_extras = None  # type: ignore


def _pg_driver_available() -> bool:
    return psycopg2 is not None and psycopg2_extras is not None


def _pg_dsn() -> str:
    user = str(getattr(settings, "postgres_user", "postgres") or "postgres")
    password = str(getattr(settings, "postgres_password", "") or "")
    host = str(getattr(settings, "postgres_host", "localhost") or "localhost")
    port = int(getattr(settings, "postgres_port", 5432) or 5432)
    dbname = str(getattr(settings, "postgres_db", "memory") or "memory")
    user_q = quote(user, safe="")
    dbname_q = quote(dbname, safe="")
    if password:
        password_q = quote(password, safe="")
        return f"postgresql://{user_q}:{password_q}@{host}:{port}/{dbname_q}"
    return f"postgresql://{user_q}@{host}:{port}/{dbname_q}"


def _split_sql_script(script: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(script):
        ch = script[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(script) and script[i + 1] == "'":
                cur.append("''")
                i += 2
                continue
            in_single = not in_single
            cur.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            cur.append(ch)
            i += 1
            continue
        if ch == ";" and not in_single and not in_double:
            stmt = "".join(cur).strip()
            if stmt:
                parts.append(stmt)
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _replace_qmark_placeholders(sql: str) -> str:
    out: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            in_single = not in_single
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            i += 1
            continue
        if ch == "?" and not in_single and not in_double:
            out.append("%s")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_sql_comments(stmt: str) -> str:
    lines = []
    for line in (stmt or "").splitlines():
        s = line.strip()
        if s.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _pg_transform_sql(sql: str, *, schema: bool) -> tuple[str, bool]:
    stmt = _strip_sql_comments(sql)
    if not stmt:
        return "", True

    low = stmt.lower().strip()
    if low.startswith("pragma"):
        return "", True

    if schema:
        if low.startswith("create virtual table"):
            return "", True
        if low.startswith("create trigger"):
            return "", True
        if "lessons_fts" in low or "memory_fts" in low:
            return "", True
    else:
        # SQLite maintenance statements that don't exist in PostgreSQL.
        if low.startswith("pragma"):
            return "", True

    if "insert into lessons_fts(lessons_fts) values('rebuild')" in low:
        return "", True

    stmt = re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
        "BIGSERIAL PRIMARY KEY",
        stmt,
        flags=re.IGNORECASE,
    )
    stmt = re.sub(r"\bAUTOINCREMENT\b", "", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"datetime\('now'\)", "CURRENT_TIMESTAMP", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"\bBLOB\b", "BYTEA", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"\bIS\s+\?", "IS NOT DISTINCT FROM ?", stmt, flags=re.IGNORECASE)

    if re.match(r"^\s*INSERT\s+OR\s+IGNORE\s+INTO\s+", stmt, flags=re.IGNORECASE):
        stmt = re.sub(
            r"^\s*INSERT\s+OR\s+IGNORE\s+INTO\s+",
            "INSERT INTO ",
            stmt,
            flags=re.IGNORECASE,
        ).strip()
        if " on conflict " not in stmt.lower():
            stmt = stmt.rstrip(";") + " ON CONFLICT DO NOTHING"

    # Handle INSERT OR REPLACE for PostgreSQL
    # INSERT OR REPLACE INTO table (col1, col2, ...) VALUES (?, ?, ...)
    # -> INSERT INTO table (col1, col2, ...) VALUES (?, ?, ...) ON CONFLICT (pk) DO UPDATE SET col1=EXCLUDED.col1, ...
    if re.match(r"^\s*INSERT\s+OR\s+REPLACE\s+INTO\s+", stmt, flags=re.IGNORECASE):
        # Extract table name and columns
        insert_match = re.match(
            r"^\s*INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*(?:\(([^)]+)\))?\s*VALUES\s*\(",
            stmt,
            flags=re.IGNORECASE,
        )
        if insert_match:
            table = insert_match.group(1).lower()
            cols_str = insert_match.group(2)
            # Handle case where columns are not specified (INSERT OR REPLACE INTO table VALUES (...))
            if cols_str:
                cols = [c.strip() for c in cols_str.split(",")]
            else:
                # Fallback: use common columns for known tables
                default_cols = {
                    "vector_chunks": ["id", "path", "chunk_index", "content", "embedding_json", "embedding_dim", "embedding_norm", "updated_at"],
                    "llm_cache": ["prompt_hash", "response", "created_at"],
                    "lessons": ["key", "lesson", "meta_json", "created_at", "updated_at"],
                    "preferences": ["key", "value_json", "source", "updated_at"],
                    "episodes": ["id", "session_id", "title", "summary", "created_at"],
                    "memory_docs": ["id", "path", "content", "updated_at"],
                    "vector_files_meta": ["path", "mtime", "size", "sha256", "updated_at"],
                    "memory_index_state": ["name", "last_scan_at", "stats_json"],
                    "procedural_memory": ["key", "title", "steps", "metadata", "updated_at"],
                    "semantic_entities": ["name", "entity_type", "properties", "updated_at"],
                }
                cols = default_cols.get(table, ["id"])
            # Map known tables to their primary keys
            pk_map = {
                "vector_chunks": "id",
                "llm_cache": "prompt_hash",
                "lessons": "key",
                "preferences": "key",
                "episodes": "id",
                "memory_docs": "id",
                "vector_files_meta": "path",
                "memory_index_state": "name",
                "procedural_memory": "key",
                "semantic_entities": "name",
            }
            pk = pk_map.get(table, "id")
            # Remove INSERT OR REPLACE prefix
            stmt = re.sub(
                r"^\s*INSERT\s+OR\s+REPLACE\s+INTO\s+",
                "INSERT INTO ",
                stmt,
                flags=re.IGNORECASE,
            ).strip()
            # Add ON CONFLICT ... DO UPDATE SET ...
            if " on conflict " not in stmt.lower():
                update_cols = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols])
                stmt = stmt.rstrip(";") + f" ON CONFLICT ({pk}) DO UPDATE SET {update_cols}"

    stmt = _replace_qmark_placeholders(stmt)
    return stmt, False


@dataclass
class _NoopCursor:
    rowcount: int = 0

    async def fetchall(self) -> list[Any]:
        return []

    async def fetchone(self) -> Any:
        return None

    async def close(self) -> None:
        return None


@dataclass
class _PgCursor:
    _cur: Any
    _lock: asyncio.Lock

    @property
    def rowcount(self) -> int:
        return int(getattr(self._cur, "rowcount", -1))

    async def fetchall(self) -> list[Any]:
        async with self._lock:
            return await asyncio.to_thread(self._cur.fetchall)

    async def fetchone(self) -> Any:
        async with self._lock:
            return await asyncio.to_thread(self._cur.fetchone)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._cur.close)


class _PgConnection:
    def __init__(self, dsn: str):
        if not _pg_driver_available():  # pragma: no cover
            raise RuntimeError("psycopg2 is not installed")
        self._dsn = dsn
        self._lock = asyncio.Lock()
        self._conn = None
        self._row_factory = None

    async def _ensure_open(self) -> None:
        if self._conn is not None:
            return
        self._conn = await asyncio.to_thread(psycopg2.connect, self._dsn)  # type: ignore[arg-type]
        self._conn.autocommit = False

    @property
    def row_factory(self):  # compatibility with sqlite connection
        return self._row_factory

    @row_factory.setter
    def row_factory(self, v):  # compatibility with sqlite connection
        self._row_factory = v

    async def execute(self, sql: str, params: Sequence[Any] | None = None):
        await self._ensure_open()
        conn = self._conn
        if conn is None:  # pragma: no cover
            raise RuntimeError("postgres connection is not initialized")
        if params is None:
            params = ()
        sql2, skip = _pg_transform_sql(sql, schema=False)
        if skip:
            return _NoopCursor()

        async with self._lock:
            cursor_factory = getattr(psycopg2_extras, "DictCursor", None)
            if cursor_factory is None:
                cur = await asyncio.to_thread(conn.cursor)
            else:
                cur = await asyncio.to_thread(
                    conn.cursor,
                    cursor_factory=cursor_factory,
                )
            await asyncio.to_thread(cur.execute, sql2, tuple(params))
            return _PgCursor(cur, self._lock)

    async def execute_fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[Any]:
        cur = await self.execute(sql, params)
        try:
            return await cur.fetchall()
        finally:
            await cur.close()

    async def executescript(self, script: str) -> None:
        await self._ensure_open()
        conn = self._conn
        if conn is None:  # pragma: no cover
            raise RuntimeError("postgres connection is not initialized")
        for stmt in _split_sql_script(script):
            sql2, skip = _pg_transform_sql(stmt, schema=True)
            if skip:
                continue
            async with self._lock:
                cur = await asyncio.to_thread(conn.cursor)
                try:
                    await asyncio.to_thread(cur.execute, sql2)
                finally:
                    await asyncio.to_thread(cur.close)

    async def commit(self) -> None:
        await self._ensure_open()
        async with self._lock:
            await asyncio.to_thread(self._conn.commit)  # type: ignore[union-attr]

    async def rollback(self) -> None:
        if self._conn is None:
            return
        async with self._lock:
            await asyncio.to_thread(self._conn.rollback)

    async def close(self) -> None:
        if self._conn is None:
            return
        async with self._lock:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def __aenter__(self) -> "_PgConnection":
        await self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                await self.commit()
            else:
                await self.rollback()
        finally:
            await self.close()


def _resolve_requested_backend() -> tuple[str, str]:
    legacy = str(getattr(settings, "db_type", "sqlite") or "sqlite").strip().lower()
    pg_flag = getattr(settings, "postgres_enabled", None)
    sqlite_flag = getattr(settings, "sqlite_enabled", None)

    if pg_flag is None and sqlite_flag is None:
        return legacy, ""

    # Single-flag mode is supported for convenience:
    # - postgres_enabled=true  => sqlite_enabled=false
    # - sqlite_enabled=true    => postgres_enabled=false
    if pg_flag is None:
        pg_flag = not bool(sqlite_flag)
    if sqlite_flag is None:
        sqlite_flag = not bool(pg_flag)

    if pg_flag and not sqlite_flag:
        return (
            "postgres",
            "backend requested via OMNIMIND_POSTGRES_ENABLED=true and OMNIMIND_SQLITE_ENABLED=false",
        )
    if sqlite_flag and not pg_flag:
        return (
            "sqlite",
            "backend requested via OMNIMIND_SQLITE_ENABLED=true and OMNIMIND_POSTGRES_ENABLED=false",
        )

    # Ambiguous configuration: keep backward compatibility with OMNIMIND_DB_TYPE.
    return (
        legacy,
        "ambiguous backend flags (both true or both false); using OMNIMIND_DB_TYPE",
    )


_DB_REQUESTED_BACKEND, _DB_SELECTION_NOTE = _resolve_requested_backend()
_DB_EFFECTIVE_BACKEND = "sqlite"
_DB_BACKEND_NOTE = ""


def db_backend_info() -> dict[str, str]:
    return {
        "requested": _DB_REQUESTED_BACKEND,
        "effective": _DB_EFFECTIVE_BACKEND,
        "note": _DB_BACKEND_NOTE,
        "selection": _DB_SELECTION_NOTE,
        "strict": "true" if bool(getattr(settings, "db_strict_backend", False)) else "false",
        "postgres_driver": "available" if _pg_driver_available() else "missing",
    }


@dataclass
class DB:
    path: str

    @asynccontextmanager
    async def connect(self):
        """Async context manager returning a configured aiosqlite connection.

        NOTE: With aiosqlite you must NOT do `async with await aiosqlite.connect(...)` because
        awaiting the connection and then entering the context starts the worker thread twice.
        """
        # Ensure parent directory exists (common in fresh Docker volumes)
        try:
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        async with aiosqlite.connect(self.path) as conn:
            # Close cursors explicitly (works for both real aiosqlite and the shim).
            cur = await conn.execute("PRAGMA journal_mode=WAL;")
            try:
                await cur.close()
            except Exception:
                pass
            cur = await conn.execute("PRAGMA foreign_keys=ON;")
            try:
                await cur.close()
            except Exception:
                pass
            # Reduce SQLITE_BUSY under concurrent access
            try:
                bt = int(getattr(settings, "db_busy_timeout_ms", 5000) or 5000)
            except Exception:
                bt = 5000
            cur = await conn.execute(f"PRAGMA busy_timeout={bt};")
            try:
                await cur.close()
            except Exception:
                pass
            conn.row_factory = aiosqlite.Row
            yield conn

    async def init(self) -> None:
        """Initialize DB schema and apply lightweight migrations (idempotent)."""
        # Best-effort: create DB parent dir
        try:
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        async with self.connect() as conn:
            await conn.executescript(SCHEMA_SQL)

            # Keep lessons full-text index in sync (idempotent, cheap).
            try:
                await conn.execute("INSERT INTO lessons_fts(lessons_fts) VALUES('rebuild');")
            except Exception:
                pass

            # --- lightweight migrations (idempotent) ---
            async def _has_column(table: str, col: str) -> bool:
                cur = await conn.execute(f"PRAGMA table_info({table});")
                rows = await cur.fetchall()
                await cur.close()
                return any(r[1] == col for r in rows)

            # sessions: chat history metadata (title/archive/last activity)
            if not await _has_column("sessions", "last_activity"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN last_activity TEXT")
            if not await _has_column("sessions", "title"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
            if not await _has_column("sessions", "archived"):
                # SQLite allows adding NOT NULL with DEFAULT.
                await conn.execute(
                    "ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
                )

            # sessions: capabilities (back-compat for older DBs)
            if not await _has_column("sessions", "capabilities_json"):
                await conn.execute(
                    "ALTER TABLE sessions ADD COLUMN capabilities_json TEXT NOT NULL DEFAULT '{}'"
                )

            # sessions: per-chat LLM settings (model/provider/sampling overrides)
            if not await _has_column("sessions", "llm_settings_json"):
                await conn.execute(
                    "ALTER TABLE sessions ADD COLUMN llm_settings_json TEXT NOT NULL DEFAULT '{}'"
                )

            # sessions: per-chat ImageGen settings (local backend/presets)
            if not await _has_column("sessions", "imagegen_settings_json"):
                await conn.execute(
                    "ALTER TABLE sessions ADD COLUMN imagegen_settings_json TEXT NOT NULL DEFAULT '{}'"
                )

            # sessions: cross-session memory fields
            if not await _has_column("sessions", "session_status"):
                await conn.execute(
                    "ALTER TABLE sessions ADD COLUMN session_status TEXT NOT NULL DEFAULT 'active'"
                )
            if not await _has_column("sessions", "started_at"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN started_at TEXT")
            if not await _has_column("sessions", "ended_at"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN ended_at TEXT")
            if not await _has_column("sessions", "summary"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
            if not await _has_column("sessions", "observations_json"):
                await conn.execute(
                    "ALTER TABLE sessions ADD COLUMN observations_json TEXT NOT NULL DEFAULT '[]'"
                )
            if not await _has_column("sessions", "user_prompt"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN user_prompt TEXT")

            # sessions: selected ImageGen profile (system-level backends/keys)
            if not await _has_column("sessions", "imagegen_profile_id"):
                await conn.execute("ALTER TABLE sessions ADD COLUMN imagegen_profile_id TEXT")

            # Backfill (idempotent)
            try:
                await conn.execute("UPDATE sessions SET archived=0 WHERE archived IS NULL")
                await conn.execute(
                    "UPDATE sessions SET last_activity=created_at WHERE last_activity IS NULL"
                )
                await conn.execute(
                    "UPDATE sessions SET capabilities_json='{}' WHERE capabilities_json IS NULL OR capabilities_json=''"
                )
                await conn.execute(
                    "UPDATE sessions SET llm_settings_json='{}' WHERE llm_settings_json IS NULL OR llm_settings_json=''"
                )
                await conn.execute(
                    "UPDATE sessions SET imagegen_settings_json='{}' WHERE imagegen_settings_json IS NULL OR imagegen_settings_json=''"
                )
            except Exception:
                pass

            # Index for fast list ordering.
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(archived, last_activity DESC)"
                )
            except Exception:
                pass

            # monitor_rules: add last_status_json + last_checked_at (for UI dashboard)
            if not await _has_column("monitor_rules", "last_status_json"):
                await conn.execute("ALTER TABLE monitor_rules ADD COLUMN last_status_json TEXT")
            if not await _has_column("monitor_rules", "last_checked_at"):
                await conn.execute("ALTER TABLE monitor_rules ADD COLUMN last_checked_at TEXT")

            # monitor_rules: auto-actions status (for UI visibility)
            if not await _has_column("monitor_rules", "last_auto_json"):
                await conn.execute("ALTER TABLE monitor_rules ADD COLUMN last_auto_json TEXT")
            if not await _has_column("monitor_rules", "last_auto_at"):
                await conn.execute("ALTER TABLE monitor_rules ADD COLUMN last_auto_at TEXT")

            # monitor_auto_events: history of auto-actions decisions (for UI/debug)
            try:
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS monitor_auto_events ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  rule_id TEXT NOT NULL REFERENCES monitor_rules(id) ON DELETE CASCADE,"
                    "  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
                    "  created_at TEXT NOT NULL,"
                    "  payload_json TEXT NOT NULL"
                    " );"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_monitor_auto_events_rule_ts ON monitor_auto_events(rule_id, created_at DESC);"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_monitor_auto_events_session_ts ON monitor_auto_events(session_id, created_at DESC);"
                )
            except Exception:
                pass

            # preferences: provenance + locking so auto-capture can't stomp user intent
            if not await _has_column("preferences", "source"):
                await conn.execute("ALTER TABLE preferences ADD COLUMN source TEXT")
            if not await _has_column("preferences", "is_locked"):
                await conn.execute("ALTER TABLE preferences ADD COLUMN is_locked INTEGER")
            if not await _has_column("preferences", "created_at"):
                await conn.execute("ALTER TABLE preferences ADD COLUMN created_at TEXT")
            if not await _has_column("preferences", "updated_by"):
                await conn.execute("ALTER TABLE preferences ADD COLUMN updated_by TEXT")

            # Backfill defaults (idempotent)
            await conn.execute(
                "UPDATE preferences SET source='manual' WHERE source IS NULL OR source=''"
            )
            await conn.execute("UPDATE preferences SET is_locked=1 WHERE is_locked IS NULL")
            await conn.execute(
                "UPDATE preferences SET created_at=updated_at WHERE created_at IS NULL"
            )
            await conn.execute(
                "UPDATE preferences SET updated_by='migration' WHERE updated_by IS NULL OR updated_by=''"
            )

            # episodes: add fingerprint column for deduplication (idempotent)
            if not await _has_column("episodes", "fingerprint"):
                await conn.execute("ALTER TABLE episodes ADD COLUMN fingerprint TEXT")
            # Index for fast dedupe checks.
            try:
                if await _has_column("episodes", "fingerprint"):
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_episodes_fingerprint ON episodes(session_id, fingerprint)"
                    )
            except Exception:
                pass

            # Procedural Memory tables
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS procedural_memory (
                        id TEXT PRIMARY KEY,
                        key TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        steps TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_procedural_key ON procedural_memory(key)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_procedural_created ON procedural_memory(created_at)"
                )

                # Backfill legacy rows that may have NULL id values.
                cur = await conn.execute(
                    "SELECT rowid FROM procedural_memory WHERE id IS NULL OR id = ''"
                )
                null_rows = await cur.fetchall()
                await cur.close()
                for r in null_rows:
                    rowid = r[0]
                    await conn.execute(
                        "UPDATE procedural_memory SET id = lower(hex(randomblob(16))) WHERE rowid = ?",
                        (rowid,),
                    )
            except Exception:
                pass

            # Semantic Memory tables
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        properties TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_semantic_name ON semantic_entities(name)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_entities(entity_type)"
                )

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_relations (
                        id TEXT PRIMARY KEY,
                        subject_id TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object_id TEXT NOT NULL,
                        properties TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_subject ON semantic_relations(subject_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_object ON semantic_relations(object_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_predicate ON semantic_relations(predicate)"
                )
            except Exception:
                pass

            # Temporal KG columns/tables for evolving-relationship graph.
            try:
                if not await _has_column("kg_triples", "updated_at"):
                    await conn.execute("ALTER TABLE kg_triples ADD COLUMN updated_at TEXT")
                if not await _has_column("kg_triples", "valid_from"):
                    await conn.execute("ALTER TABLE kg_triples ADD COLUMN valid_from TEXT")
                if not await _has_column("kg_triples", "valid_to"):
                    await conn.execute("ALTER TABLE kg_triples ADD COLUMN valid_to TEXT")
                if not await _has_column("kg_triples", "is_active"):
                    await conn.execute(
                        "ALTER TABLE kg_triples ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
                    )
                if not await _has_column("kg_triples", "version"):
                    await conn.execute(
                        "ALTER TABLE kg_triples ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
                    )
                if not await _has_column("kg_triples", "last_event_type"):
                    await conn.execute("ALTER TABLE kg_triples ADD COLUMN last_event_type TEXT")

                await conn.execute(
                    "UPDATE kg_triples SET updated_at=created_at WHERE updated_at IS NULL"
                )
                await conn.execute(
                    "UPDATE kg_triples SET valid_from=created_at WHERE valid_from IS NULL"
                )
                await conn.execute("UPDATE kg_triples SET is_active=1 WHERE is_active IS NULL")
                await conn.execute("UPDATE kg_triples SET version=1 WHERE version IS NULL")
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_triples_active ON kg_triples(is_active, predicate_id, subject_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_triples_validity ON kg_triples(valid_from, valid_to)"
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kg_triple_events (
                        id TEXT PRIMARY KEY,
                        triple_id TEXT,
                        subject_id TEXT NOT NULL REFERENCES kg_subjects(id) ON DELETE CASCADE,
                        predicate_id TEXT NOT NULL REFERENCES kg_predicates(id) ON DELETE CASCADE,
                        object_id TEXT NOT NULL REFERENCES kg_objects(id) ON DELETE CASCADE,
                        action TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        valid_from TEXT,
                        valid_to TEXT,
                        state_active INTEGER NOT NULL,
                        state_version INTEGER NOT NULL,
                        confidence REAL,
                        source_type TEXT,
                        source_id TEXT,
                        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_subject_predicate_ts ON kg_triple_events(subject_id, predicate_id, observed_at DESC)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_object_ts ON kg_triple_events(object_id, observed_at DESC)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_triple_ts ON kg_triple_events(triple_id, observed_at DESC)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_session_ts ON kg_triple_events(session_id, observed_at DESC)"
                )
            except Exception:
                pass

            # Audit log table
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        action TEXT NOT NULL,
                        resource_type TEXT,
                        resource_id TEXT,
                        details TEXT,
                        ip_address TEXT
                    )
                """)
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id)"
                )
            except Exception:
                pass

            await conn.commit()


@dataclass
class PostgresDB:
    path: str
    dsn: str

    @asynccontextmanager
    async def connect(self):
        async with _PgConnection(self.dsn) as conn:
            yield conn

    async def init(self) -> None:
        async with self.connect() as conn:
            await conn.executescript(SCHEMA_SQL)

            # Mirror SQLite migration behavior for existing PostgreSQL databases.
            alter_statements = [
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_activity TEXT",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS capabilities_json TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS llm_settings_json TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS imagegen_settings_json TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_status TEXT NOT NULL DEFAULT 'active'",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS started_at TEXT",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS ended_at TEXT",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS observations_json TEXT NOT NULL DEFAULT '[]'",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_prompt TEXT",
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS imagegen_profile_id TEXT",
                "ALTER TABLE monitor_rules ADD COLUMN IF NOT EXISTS last_status_json TEXT",
                "ALTER TABLE monitor_rules ADD COLUMN IF NOT EXISTS last_checked_at TEXT",
                "ALTER TABLE monitor_rules ADD COLUMN IF NOT EXISTS last_auto_json TEXT",
                "ALTER TABLE monitor_rules ADD COLUMN IF NOT EXISTS last_auto_at TEXT",
                "ALTER TABLE preferences ADD COLUMN IF NOT EXISTS source TEXT",
                "ALTER TABLE preferences ADD COLUMN IF NOT EXISTS is_locked INTEGER",
                "ALTER TABLE preferences ADD COLUMN IF NOT EXISTS created_at TEXT",
                "ALTER TABLE preferences ADD COLUMN IF NOT EXISTS updated_by TEXT",
                "ALTER TABLE episodes ADD COLUMN IF NOT EXISTS fingerprint TEXT",
                "ALTER TABLE monitor_results ADD COLUMN IF NOT EXISTS session_id TEXT",
                "ALTER TABLE monitor_results ADD COLUMN IF NOT EXISTS checked_at TEXT",
                "ALTER TABLE kg_triples ADD COLUMN IF NOT EXISTS updated_at TEXT",
                "ALTER TABLE kg_triples ADD COLUMN IF NOT EXISTS valid_from TEXT",
                "ALTER TABLE kg_triples ADD COLUMN IF NOT EXISTS valid_to TEXT",
                "ALTER TABLE kg_triples ADD COLUMN IF NOT EXISTS is_active INTEGER NOT NULL DEFAULT 1",
                "ALTER TABLE kg_triples ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1",
                "ALTER TABLE kg_triples ADD COLUMN IF NOT EXISTS last_event_type TEXT",
            ]
            for stmt in alter_statements:
                try:
                    await conn.execute(stmt)
                except Exception:
                    pass

            backfill_statements = [
                "UPDATE sessions SET archived=0 WHERE archived IS NULL",
                "UPDATE sessions SET last_activity=created_at WHERE last_activity IS NULL",
                "UPDATE sessions SET capabilities_json='{}' WHERE capabilities_json IS NULL OR capabilities_json=''",
                "UPDATE sessions SET llm_settings_json='{}' WHERE llm_settings_json IS NULL OR llm_settings_json=''",
                "UPDATE sessions SET imagegen_settings_json='{}' WHERE imagegen_settings_json IS NULL OR imagegen_settings_json=''",
                "UPDATE preferences SET source='manual' WHERE source IS NULL OR source=''",
                "UPDATE preferences SET is_locked=1 WHERE is_locked IS NULL",
                "UPDATE preferences SET created_at=updated_at WHERE created_at IS NULL",
                "UPDATE preferences SET updated_by='migration' WHERE updated_by IS NULL OR updated_by=''",
                "UPDATE kg_triples SET updated_at=created_at WHERE updated_at IS NULL",
                "UPDATE kg_triples SET valid_from=created_at WHERE valid_from IS NULL",
                "UPDATE kg_triples SET is_active=1 WHERE is_active IS NULL",
                "UPDATE kg_triples SET version=1 WHERE version IS NULL",
            ]
            for stmt in backfill_statements:
                try:
                    await conn.execute(stmt)
                except Exception:
                    pass

            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_triples_active ON kg_triples(is_active, predicate_id, subject_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_triples_validity ON kg_triples(valid_from, valid_to)"
                )
            except Exception:
                pass

            # Temporal triple event stream for time-slice reasoning.
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kg_triple_events (
                        id TEXT PRIMARY KEY,
                        triple_id TEXT,
                        subject_id TEXT NOT NULL REFERENCES kg_subjects(id) ON DELETE CASCADE,
                        predicate_id TEXT NOT NULL REFERENCES kg_predicates(id) ON DELETE CASCADE,
                        object_id TEXT NOT NULL REFERENCES kg_objects(id) ON DELETE CASCADE,
                        action TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        valid_from TEXT,
                        valid_to TEXT,
                        state_active INTEGER NOT NULL,
                        state_version INTEGER NOT NULL,
                        confidence REAL,
                        source_type TEXT,
                        source_id TEXT,
                        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_subject_predicate_ts ON kg_triple_events(subject_id, predicate_id, observed_at DESC)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_object_ts ON kg_triple_events(object_id, observed_at DESC)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_triple_ts ON kg_triple_events(triple_id, observed_at DESC)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kg_events_session_ts ON kg_triple_events(session_id, observed_at DESC)"
                )
            except Exception:
                pass

            # Procedural memory tables (parity with SQLite init path).
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS procedural_memory (
                        id TEXT PRIMARY KEY,
                        key TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        steps TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_procedural_key ON procedural_memory(key)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_procedural_created ON procedural_memory(created_at)"
                )
            except Exception:
                pass

            # Semantic memory tables.
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        properties TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_semantic_name ON semantic_entities(name)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_entities(entity_type)"
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_relations (
                        id TEXT PRIMARY KEY,
                        subject_id TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object_id TEXT NOT NULL,
                        properties TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_subject ON semantic_relations(subject_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_object ON semantic_relations(object_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_predicate ON semantic_relations(predicate)"
                )
            except Exception:
                pass

            # Audit log table used by quality/security tests.
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        action TEXT NOT NULL,
                        resource_type TEXT,
                        resource_id TEXT,
                        details TEXT,
                        ip_address TEXT
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id)"
                )
            except Exception:
                pass

            # PostgreSQL FTS indexes used by lessons/workspace search in parity mode.
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_lessons_search_tsv ON lessons "
                    "USING GIN (to_tsvector('simple', coalesce(key,'') || ' ' || coalesce(lesson,'') || ' ' || coalesce(meta_json,'')))"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_docs_search_tsv ON memory_docs "
                    "USING GIN (to_tsvector('simple', coalesce(path,'') || ' ' || coalesce(content,'')))"
                )
            except Exception:
                pass


def _build_db() -> DB | PostgresDB:
    global _DB_EFFECTIVE_BACKEND, _DB_BACKEND_NOTE

    strict_backend = bool(getattr(settings, "db_strict_backend", False))

    def _raise_if_strict_mismatch() -> None:
        if not strict_backend:
            return
        if _DB_REQUESTED_BACKEND != _DB_EFFECTIVE_BACKEND:
            raise RuntimeError(
                "strict backend mode: requested "
                f"'{_DB_REQUESTED_BACKEND}' but effective backend is '{_DB_EFFECTIVE_BACKEND}'. "
                "Adjust OMNIMIND_POSTGRES_ENABLED/OMNIMIND_SQLITE_ENABLED or disable OMNIMIND_DB_STRICT_BACKEND."
            )

    if _DB_REQUESTED_BACKEND in ("", "sqlite"):
        _DB_EFFECTIVE_BACKEND = "sqlite"
        _DB_BACKEND_NOTE = ""
        _raise_if_strict_mismatch()
        return DB(settings.db_path)

    if _DB_REQUESTED_BACKEND == "postgres":
        if not _pg_driver_available():
            _DB_EFFECTIVE_BACKEND = "sqlite"
            _DB_BACKEND_NOTE = (
                "postgres requested but psycopg2 is not installed; using sqlite backend"
            )
            warnings.warn(_DB_BACKEND_NOTE)
            _raise_if_strict_mismatch()
            return DB(settings.db_path)

        _DB_EFFECTIVE_BACKEND = "postgres"
        _DB_BACKEND_NOTE = ""
        _raise_if_strict_mismatch()
        return PostgresDB(path=settings.db_path, dsn=_pg_dsn())

    _DB_EFFECTIVE_BACKEND = "sqlite"
    _DB_BACKEND_NOTE = f"unknown db_type '{_DB_REQUESTED_BACKEND}', using sqlite backend"
    warnings.warn(_DB_BACKEND_NOTE)
    _raise_if_strict_mismatch()
    return DB(settings.db_path)


db = _build_db()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  last_activity TEXT,
  title TEXT,
  archived INTEGER NOT NULL DEFAULT 0,
  role TEXT NOT NULL,
  capabilities_json TEXT NOT NULL,
  llm_settings_json TEXT NOT NULL DEFAULT '{}',
  imagegen_profile_id TEXT,
  imagegen_settings_json TEXT NOT NULL DEFAULT '{}',
  session_status TEXT NOT NULL DEFAULT 'active',
  started_at TEXT,
  ended_at TEXT,
  summary TEXT,
  observations_json TEXT NOT NULL DEFAULT '[]',
  user_prompt TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(session_status);
CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(ended_at);


CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  result_json TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS attachments (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
  filename TEXT NOT NULL,
  content_type TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  sha256 TEXT,
  storage_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attachments_session_created ON attachments(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_attachments_task_created ON attachments(task_id, created_at DESC);

CREATE TABLE IF NOT EXISTS imagegen_profiles (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  backend TEXT NOT NULL,
  base_url TEXT NOT NULL,
  auth_header_name TEXT,
  auth_header_value TEXT,
  auth_query_name TEXT,
  auth_query_value TEXT,
  settings_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_imagegen_profiles_name ON imagegen_profiles(name);
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  decided_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT,
  task_id TEXT,
  level TEXT NOT NULL,
  type TEXT NOT NULL,
  data_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT,
  trace_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_traces_session_ts ON retrieval_traces(session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_traces_task_id ON retrieval_traces(task_id);


CREATE TABLE IF NOT EXISTS retrieval_eval_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  query TEXT NOT NULL,
  top_n INTEGER NOT NULL,
  modes_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  results_json TEXT NOT NULL,
  app_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_retrieval_eval_runs_session_ts ON retrieval_eval_runs(session_id, ts DESC);

CREATE TABLE IF NOT EXISTS staged_files (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  old_sha256 TEXT,
  new_content TEXT NOT NULL,
  status TEXT NOT NULL
);


-- Patch sets (transactional changes)
CREATE TABLE IF NOT EXISTS patch_sets (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  summary TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  applied_at TEXT,
  rolled_back_at TEXT
);

CREATE TABLE IF NOT EXISTS patch_entries (
  id TEXT PRIMARY KEY,
  patch_id TEXT NOT NULL REFERENCES patch_sets(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  preview_old_sha256 TEXT,
  preview_new_sha256 TEXT NOT NULL,
  preview_diff TEXT NOT NULL,
  new_content TEXT NOT NULL,
  pre_apply_exists INTEGER,
  pre_apply_sha256 TEXT,
  pre_apply_content TEXT,
  status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_staged_files_task ON staged_files(task_id);

CREATE TABLE IF NOT EXISTS monitor_rules (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  rule_json TEXT NOT NULL,
  is_enabled INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  last_status_json TEXT,
  last_checked_at TEXT,
  last_auto_json TEXT,
  last_auto_at TEXT
);

CREATE TABLE IF NOT EXISTS monitor_results (
  id TEXT PRIMARY KEY,
  rule_id TEXT NOT NULL REFERENCES monitor_rules(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  rule_name TEXT NOT NULL,
  rule_type TEXT,
  checked_at TEXT NOT NULL,
  status TEXT NOT NULL,
  details_json TEXT NOT NULL,
  duration_ms INTEGER,
  alert_emitted INTEGER NOT NULL DEFAULT 0,
  sig TEXT
);

CREATE INDEX IF NOT EXISTS idx_monitor_results_rule_ts ON monitor_results(rule_id, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_monitor_results_session_ts ON monitor_results(session_id, checked_at DESC);

CREATE TABLE IF NOT EXISTS monitor_auto_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rule_id TEXT NOT NULL REFERENCES monitor_rules(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_monitor_auto_events_rule_ts ON monitor_auto_events(rule_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_monitor_auto_events_session_ts ON monitor_auto_events(session_id, created_at DESC);

-- Memory: session snapshot (STM)
CREATE TABLE IF NOT EXISTS session_memory (
  session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  snapshot_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Memory: working memory (WM)
-- A fast mutable scratchpad for the current session.
-- Stored as a single text blob to keep things simple and predictable.
CREATE TABLE IF NOT EXISTS working_memory (
  session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Memory: preferences (durable user/system preferences)
-- Preferences can be global (scope=global) or session-scoped (scope=session).
-- For session-scoped prefs, session_id should be set.
CREATE TABLE IF NOT EXISTS preferences (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  session_id TEXT,
  key TEXT NOT NULL,
  value_json TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'manual',
  is_locked INTEGER NOT NULL DEFAULT 1,
  created_at TEXT,
  updated_by TEXT,
  updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_scope_session_key ON preferences(scope, session_id, key);


-- Memory: lessons DB (experience memory)
-- TTL: added created_at and expires_at for auto-expiry
CREATE TABLE IF NOT EXISTS lessons (
  key TEXT PRIMARY KEY,
  lesson TEXT NOT NULL,
  meta_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at TEXT,
  updated_at TEXT NOT NULL
);

-- Lessons FTS (queryable experience memory)
-- We keep lessons searchable without scanning the whole table by adding an FTS5 index
-- and syncing it via triggers (like memory_docs/memory_fts).
CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
  key,
  lesson,
  meta_json,
  content='lessons',
  content_rowid='rowid',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
  INSERT INTO lessons_fts(rowid, key, lesson, meta_json) VALUES (new.rowid, new.key, new.lesson, new.meta_json);
END;

CREATE TRIGGER IF NOT EXISTS lessons_ad AFTER DELETE ON lessons BEGIN
  INSERT INTO lessons_fts(lessons_fts, rowid, key, lesson, meta_json) VALUES('delete', old.rowid, old.key, old.lesson, old.meta_json);
END;

CREATE TRIGGER IF NOT EXISTS lessons_au AFTER UPDATE ON lessons BEGIN
  INSERT INTO lessons_fts(lessons_fts, rowid, key, lesson, meta_json) VALUES('delete', old.rowid, old.key, old.lesson, old.meta_json);
  INSERT INTO lessons_fts(rowid, key, lesson, meta_json) VALUES (new.rowid, new.key, new.lesson, new.meta_json);
END;


-- Memory: episodic logs (what happened in tasks)
-- TTL: added expires_at for auto-expiry
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  data_json TEXT NOT NULL,
  fingerprint TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_session_ts ON episodes(session_id, created_at DESC);


-- Memory: conversations (chat history)
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  model TEXT,
  tokens INTEGER,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations(session_id, created_at DESC);


-- Knowledge Base: document storage
CREATE TABLE IF NOT EXISTS knowledge_base (
  id TEXT PRIMARY KEY,
  session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_url TEXT,
  source_path TEXT,
  format TEXT NOT NULL,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  indexed INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_kb_session_created ON knowledge_base(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_kb_source ON knowledge_base(source_type, source_url);


-- Semantic Triples (Knowledge Graph)
CREATE TABLE IF NOT EXISTS kg_subjects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  entity_type TEXT,
  created_at TEXT NOT NULL,
  mention_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS kg_predicates (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_objects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  entity_type TEXT,
  created_at TEXT NOT NULL,
  mention_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS kg_triples (
  id TEXT PRIMARY KEY,
  subject_id TEXT NOT NULL REFERENCES kg_subjects(id) ON DELETE CASCADE,
  predicate_id TEXT NOT NULL REFERENCES kg_predicates(id) ON DELETE CASCADE,
  object_id TEXT NOT NULL REFERENCES kg_objects(id) ON DELETE CASCADE,
  confidence REAL NOT NULL DEFAULT 1.0,
  source_type TEXT NOT NULL,
  source_id TEXT,
  session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
   updated_at TEXT,
   valid_from TEXT,
   valid_to TEXT,
   is_active INTEGER NOT NULL DEFAULT 1,
   version INTEGER NOT NULL DEFAULT 1,
   last_event_type TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(subject_id, predicate_id, object_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_triples_subject ON kg_triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_kg_triples_object ON kg_triples(object_id);
CREATE INDEX IF NOT EXISTS idx_kg_triples_predicate ON kg_triples(predicate_id);
CREATE INDEX IF NOT EXISTS idx_kg_triples_session ON kg_triples(session_id);

CREATE TABLE IF NOT EXISTS kg_triple_events (
  id TEXT PRIMARY KEY,
  triple_id TEXT,
  subject_id TEXT NOT NULL REFERENCES kg_subjects(id) ON DELETE CASCADE,
  predicate_id TEXT NOT NULL REFERENCES kg_predicates(id) ON DELETE CASCADE,
  object_id TEXT NOT NULL REFERENCES kg_objects(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  valid_from TEXT,
  valid_to TEXT,
  state_active INTEGER NOT NULL,
  state_version INTEGER NOT NULL,
  confidence REAL,
  source_type TEXT,
  source_id TEXT,
  session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_events_subject_predicate_ts
  ON kg_triple_events(subject_id, predicate_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_kg_events_object_ts
  ON kg_triple_events(object_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_kg_events_triple_ts
  ON kg_triple_events(triple_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_kg_events_session_ts
  ON kg_triple_events(session_id, observed_at DESC);


-- Memory Attribution (Entity/Process/Session)
CREATE TABLE IF NOT EXISTS memory_attribution (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  process_id TEXT,
  session_id TEXT,
  memory_type TEXT NOT NULL,
  memory_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attribution_entity ON memory_attribution(entity_id);
CREATE INDEX IF NOT EXISTS idx_attribution_process ON memory_attribution(process_id);
CREATE INDEX IF NOT EXISTS idx_attribution_session ON memory_attribution(session_id);


-- Extracted Memory Types (Advanced Augmentation)
CREATE TABLE IF NOT EXISTS extracted_memories (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  session_id TEXT,
  memory_type TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0,
  source_text TEXT,
  vector_embedding BLOB,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_extracted_entity ON extracted_memories(entity_id);
CREATE INDEX IF NOT EXISTS idx_extracted_type ON extracted_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_extracted_session ON extracted_memories(session_id);


-- Web cache (HTTP conditional GET)
CREATE TABLE IF NOT EXISTS web_cache (
  url TEXT PRIMARY KEY,
  etag TEXT,
  last_modified TEXT,
  status INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  text TEXT NOT NULL,
  sha256 TEXT NOT NULL
);


-- Web research history (Stage 6)
CREATE TABLE IF NOT EXISTS research_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  question TEXT NOT NULL,
  answer_preview TEXT NOT NULL,
  sources_count INTEGER NOT NULL DEFAULT 0,
  payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_research_history_session_ts ON research_history(session_id, created_at DESC);

-- Memory: vector chunks (semantic search)
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



-- Memory: project knowledge (SQLite FTS5)
CREATE TABLE IF NOT EXISTS memory_docs (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Workspace file metadata for incremental indexing (FTS).
-- mtime is stored as float seconds since epoch (from stat().st_mtime).
CREATE TABLE IF NOT EXISTS workspace_files_meta (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  sha256 TEXT,
  updated_at TEXT NOT NULL
);

-- Index state table to throttle background indexing and keep debug info.
CREATE TABLE IF NOT EXISTS memory_index_state (
  name TEXT PRIMARY KEY,
  last_scan_at TEXT,
  stats_json TEXT
);



-- Graph memory (associative links)
CREATE TABLE IF NOT EXISTS graph_nodes (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  type TEXT NOT NULL,
  content TEXT,
  metadata_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  last_accessed TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_label ON graph_nodes(label);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(type);

CREATE TABLE IF NOT EXISTS graph_edges (
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  metadata_json TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (source_id, target_id, relation),
  FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
  FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
CREATE TABLE IF NOT EXISTS maintenance_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  path,
  content,
  content='memory_docs',
  content_rowid='rowid',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_docs_ai AFTER INSERT ON memory_docs BEGIN
  INSERT INTO memory_fts(rowid, path, content) VALUES (new.rowid, new.path, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_docs_ad AFTER DELETE ON memory_docs BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, path, content) VALUES('delete', old.rowid, old.path, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_docs_au AFTER UPDATE ON memory_docs BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, path, content) VALUES('delete', old.rowid, old.path, old.content);
  INSERT INTO memory_fts(rowid, path, content) VALUES (new.rowid, new.path, new.content);
END;


-- Vector index metadata (optional incremental rebuild)
CREATE TABLE IF NOT EXISTS vector_files_meta (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  sha256 TEXT,
  updated_at TEXT NOT NULL
);

-- Autonomy action log (anti-loop / dedupe)
CREATE TABLE IF NOT EXISTS autonomy_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  session_id TEXT,
  task_id TEXT,
  tool TEXT NOT NULL,
  action TEXT NOT NULL,
  fingerprint TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_autonomy_actions_fp_ts ON autonomy_actions(fingerprint, ts);
CREATE INDEX IF NOT EXISTS idx_autonomy_actions_session_ts ON autonomy_actions(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_autonomy_actions_ts ON autonomy_actions(ts);

"""


async def init_db() -> None:
    """Backward-compatible DB init entrypoint.

    Historically the app used init_db() from main.py. The DB class now
    owns the schema + migrations, so we delegate to db.init() to ensure
    all migrations (including newer ones) are consistently applied.
    """
    await db.init()


async def fetch_one(conn: Any, sql: str, args: Sequence[Any] = ()) -> Optional[Any]:
    cur = await conn.execute(sql, args)
    row = await cur.fetchone()
    await cur.close()
    return row


async def fetch_all(conn: Any, sql: str, args: Sequence[Any] = ()) -> list[Any]:
    cur = await conn.execute(sql, args)
    rows = await cur.fetchall()
    await cur.close()
    return rows


def row_get(row: Any, key: str, default: Any = None) -> Any:
    """Dict-like .get() for sqlite rows.

    OmniMind uses aiosqlite in most environments, but falls back to a sqlite3
    shim in selfcheck / minimal installs. sqlite3.Row behaves like a mapping
    (row["col"]) and has .keys(), but it does NOT implement .get().

    This helper makes call sites robust across:
      - dict
      - sqlite3.Row / aiosqlite.Row
      - other mapping-like objects
    """
    if row is None:
        return default

    # Fast-path: dict
    if isinstance(row, dict):
        return row.get(key, default)

    # sqlite3.Row / aiosqlite.Row: mapping access via [] + .keys()
    try:
        keys = row.keys()  # type: ignore[attr-defined]
        if key in keys:
            return row[key]
        return default
    except Exception:
        pass

    try:
        return row[key]  # type: ignore[index]
    except Exception:
        return default


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads(s: str) -> Any:
    return json.loads(s)
