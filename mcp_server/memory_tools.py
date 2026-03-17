"""MCP memory tools - FastMCP based."""

import asyncio
import time
from functools import wraps
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

from core.config import settings
from core.db import init_db
from core.memory import MemoryStore
from core.metrics import metrics, request_metrics
from core.search_match import query_tokens, score_fields
from core.security.rate_limit import distributed_rate_limiter

mcp = FastMCP("omnimind-memory")

# Initialize memory store
memory = MemoryStore()


_db_init_lock = asyncio.Lock()
_db_initialized = False


async def _ensure_db_ready() -> None:
    global _db_initialized
    if _db_initialized:
        return
    async with _db_init_lock:
        if _db_initialized:
            return
        await init_db()
        _db_initialized = True


def _with_db_ready(fn):
    @wraps(fn)
    async def _wrapped(*args, **kwargs):
        # Per-tool simple rate limit (keyed by function name).
        if not distributed_rate_limiter.is_allowed(fn.__name__):
            raise RuntimeError(f"Rate limit exceeded for tool: {fn.__name__}")

        started = time.time()
        await _ensure_db_ready()
        metrics.increment(f"tool.calls.{fn.__name__}")

        status = 200
        try:
            result = await fn(*args, **kwargs)
            return result
        except Exception:
            status = 500
            metrics.increment(f"tool.errors.{fn.__name__}")
            raise
        finally:
            duration_ms = (time.time() - started) * 1000.0
            metrics.timing(f"tool.duration_ms.{fn.__name__}", duration_ms)
            request_metrics.record_request(fn.__name__, status=status, duration_ms=duration_ms)

    return _wrapped


def _sort_by_score(
    items: List[Dict[str, Any]], *, score_key: str = "search_score"
) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            float(x.get(score_key) or 0.0),
            str(x.get("updated_at") or x.get("created_at") or ""),
        ),
        reverse=True,
    )


@mcp.tool()
@_with_db_ready
async def memory_search(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Hybrid search: combine FTS keyword and vector semantic search."""
    q = (query or "").strip()
    if not q:
        return []

    # Ensure workspace FTS is available; operation is throttled in storage layer.
    try:
        await memory.ensure_indexed(getattr(settings, "workspace", "/workspace"))
    except Exception:
        pass

    # Uses unified hybrid path (BM25/query-expansion/rerank settings aware).
    hits = await memory.search_hybrid(q, limit=limit)
    return [
        {
            "source": h.source,
            "text": h.text,
            "score": h.score,
            "meta": h.meta,
        }
        for h in hits
    ]


@mcp.tool()
@_with_db_ready
async def memory_search_lessons(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search lessons by query string."""
    q = (query or "").strip()
    if not q:
        return []

    pool = max(int(limit) * 3, int(limit), 20)
    lessons = await memory.search_lessons(query=q, limit=pool)
    if lessons:
        out = []
        tokens = query_tokens(q, max_terms=12)
        for item in lessons:
            score = score_fields(
                q,
                [str(item.get("key") or ""), str(item.get("lesson") or "")],
                tokens=tokens,
            )
            # FTS BM25 rank: lower is better.
            rank = float(item.get("rank") or 0.0)
            score += 1.0 / (1.0 + max(0.0, rank))
            if score <= 0.0:
                continue
            it2 = dict(item)
            it2["search_score"] = float(score)
            out.append(it2)
        return _sort_by_score(out)[: int(limit)]

    # Fallback: lexical scan over recent lessons.
    fallback = await memory.list_lessons(limit=max(100, pool))
    tokens = query_tokens(q, max_terms=12)
    out = []
    for item in fallback:
        score = score_fields(
            q,
            [str(item.get("key") or ""), str(item.get("lesson") or "")],
            tokens=tokens,
        )
        if score <= 0.0:
            continue
        it2 = dict(item)
        it2["search_score"] = float(score)
        out.append(it2)
    return _sort_by_score(out)[: int(limit)]


@mcp.tool()
@_with_db_ready
async def memory_search_preferences(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search preferences by key prefix or value content."""
    prefs = await memory.list_preferences(scope="global", session_id=None, limit=500)
    q = (query or "").strip()
    if not q:
        return prefs[:limit]

    tokens = query_tokens(q, max_terms=12)
    if not tokens:
        tokens = [q.casefold()]

    results: List[Dict[str, Any]] = []
    for p in prefs:
        score = score_fields(
            q,
            [str(p.get("key") or ""), str(p.get("value") or "")],
            tokens=tokens,
        )
        if score <= 0.0:
            continue
        item = dict(p)
        item["search_score"] = float(score)
        results.append(item)

    return _sort_by_score(results)[: int(limit)]


@mcp.tool()
@_with_db_ready
async def memory_search_all(query: str, limit: int = 10) -> Dict[str, Any]:
    """Federated search across memory domains.

    Keeps backward compatibility (`lessons`, `preferences`) and adds
    broader result buckets for easier retrieval.
    """
    q = (query or "").strip()
    out: Dict[str, Any] = {
        "lessons": [],
        "preferences": [],
        "procedures": [],
        "entities": [],
        "kb_documents": [],
        "kg_entities": [],
        "extracted_memories": [],
        "cross_sessions": [],
        "workspace": [],
        "top": [],
    }
    if not q:
        return out

    tokens = query_tokens(q, max_terms=12)
    pool = max(int(limit) * 4, int(limit), 20)

    async def _safe(coro):
        try:
            return await coro
        except Exception:
            return []

    # Ensure workspace index exists before hybrid search.
    try:
        await memory.ensure_indexed(getattr(settings, "workspace", "/workspace"))
    except Exception:
        pass

    (
        lessons,
        preferences,
        procedures,
        entities,
        kb_docs,
        kg_entities,
        extracted,
        sessions,
        workspace_hits,
    ) = await asyncio.gather(
        _safe(memory.search_lessons(query=q, limit=pool)),
        _safe(memory.list_preferences(scope="global", session_id=None, limit=500)),
        _safe(memory.search_procedures(query=q, limit=pool)),
        _safe(memory.search_entities(query=q, limit=pool)),
        _safe(memory.kb_search_documents(query=q, session_id=None, limit=pool)),
        _safe(memory.kg_search_entities(query=q, entity_type=None, limit=pool)),
        _safe(memory.search_extracted_memories(query=q, entity_id=None, limit=pool)),
        _safe(memory.cross_session_search(query=q, limit=pool)),
        _safe(memory.search_hybrid(query=q, limit=pool)),
    )

    # lessons
    lesson_items: List[Dict[str, Any]] = []
    for it in lessons:
        score = score_fields(
            q, [str(it.get("key") or ""), str(it.get("lesson") or "")], tokens=tokens
        )
        score += 1.0 / (1.0 + max(0.0, float(it.get("rank") or 0.0)))
        if score <= 0.0:
            continue
        row = dict(it)
        row["search_score"] = float(score)
        lesson_items.append(row)
    out["lessons"] = _sort_by_score(lesson_items)[: int(limit)]

    # preferences
    pref_items: List[Dict[str, Any]] = []
    for it in preferences:
        score = score_fields(
            q, [str(it.get("key") or ""), str(it.get("value") or "")], tokens=tokens
        )
        if score <= 0.0:
            continue
        row = dict(it)
        row["search_score"] = float(score)
        pref_items.append(row)
    out["preferences"] = _sort_by_score(pref_items)[: int(limit)]

    # procedures / semantic entities / kb / kg / extracted / cross-session
    def _score_bucket(items: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for it in items:
            vals = [str(it.get(f) or "") for f in fields]
            score = score_fields(q, vals, tokens=tokens)
            if score <= 0.0:
                continue
            row = dict(it)
            row["search_score"] = float(score)
            scored.append(row)
        return _sort_by_score(scored)[: int(limit)]

    out["procedures"] = _score_bucket(procedures, ["key", "title", "steps"])
    out["entities"] = _score_bucket(entities, ["name", "entity_type"])
    out["kb_documents"] = _score_bucket(kb_docs, ["title", "snippet"])
    out["kg_entities"] = _score_bucket(kg_entities, ["name", "type", "role"])
    out["extracted_memories"] = _score_bucket(extracted, ["memory_type", "content", "entity_id"])
    out["cross_sessions"] = _score_bucket(sessions, ["title", "summary", "observations"])

    # workspace hits
    ws_items: List[Dict[str, Any]] = []
    for h in workspace_hits:
        text = str(h.text or "")
        meta = dict(h.meta or {})
        score = float(h.score or 0.0)
        score += score_fields(q, [text, str(meta.get("path") or "")], tokens=tokens) * 0.2
        if score <= 0.0:
            continue
        ws_items.append(
            {
                "source": h.source,
                "text": text,
                "path": meta.get("path"),
                "meta": meta,
                "search_score": float(score),
            }
        )
    out["workspace"] = _sort_by_score(ws_items)[: int(limit)]

    # Unified top list for model-friendly consumption.
    top: List[Dict[str, Any]] = []
    buckets = {
        "lessons": out["lessons"],
        "preferences": out["preferences"],
        "procedures": out["procedures"],
        "entities": out["entities"],
        "kb_documents": out["kb_documents"],
        "kg_entities": out["kg_entities"],
        "extracted_memories": out["extracted_memories"],
        "cross_sessions": out["cross_sessions"],
        "workspace": out["workspace"],
    }
    for source, items in buckets.items():
        for it in items:
            top.append(
                {
                    "source": source,
                    "score": float(it.get("search_score") or 0.0),
                    "item": it,
                }
            )
    top.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    out["top"] = top[: max(int(limit) * 3, int(limit))]

    return out


@mcp.tool()
@_with_db_ready
async def memory_upsert(
    key: str,
    value: str,
    type: str = "lesson",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Upsert a memory entry (lesson, preference)."""
    if type == "lesson":
        await memory.add_lesson(key, value, meta=meta)
        return {"ok": True, "type": "lesson", "key": key}
    elif type == "preference":
        await memory.set_preference(
            scope="global",
            session_id=None,
            key=key,
            value=value,
            source="system",
        )
        return {"ok": True, "type": "preference", "key": key}
    else:
        return {"ok": False, "error": f"Unknown type: {type}"}


@mcp.tool()
@_with_db_ready
async def memory_get(key: str, type: str = "lesson") -> Dict[str, Any]:
    """Get a memory entry by key."""
    if type == "lesson":
        lessons = await memory.list_lessons(limit=100)
        for lesson in lessons:
            if lesson.get("key") == key:
                return {"ok": True, "lesson": lesson}
        return {"ok": False, "error": "Not found"}
    elif type == "preference":
        pref = await memory.get_preference(scope="global", session_id=None, key=key)
        return {"ok": True, "preference": pref}
    else:
        return {"ok": False, "error": f"Unknown type: {type}"}


@mcp.tool()
@_with_db_ready
async def memory_list(
    type: str = "lessons",
    limit: int = 50,
) -> Dict[str, Any]:
    """List memory entries (lessons, preferences)."""
    if type == "lessons":
        lessons = await memory.list_lessons(limit=limit)
        return {"ok": True, "lessons": lessons}
    elif type == "preferences":
        prefs = await memory.list_preferences(scope="global", session_id=None, limit=limit)
        return {"ok": True, "preferences": prefs}
    else:
        return {"ok": False, "error": f"Unknown type: {type}"}


@mcp.tool()
@_with_db_ready
async def memory_delete(key: str, type: str = "lesson") -> Dict[str, Any]:
    """Delete a memory entry."""
    if type == "preference":
        await memory.delete_preference(scope="global", session_id=None, key=key)
        return {"ok": True}
    else:
        return {"ok": False, "error": "Use consolidate to clean up lessons"}


@mcp.tool()
@_with_db_ready
async def memory_add_episode(
    session_id: str,
    title: str,
    summary: str,
    task_id: str = "",
    tags: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add an episode to the session memory.

    Episodes are short-term memory entries that can be consolidated into
    long-term lessons and preferences. Use this after completing tasks
    to record what happened for later consolidation.

    Args:
        session_id: Current session identifier
        title: Short title describing the episode
        summary: Detailed summary of what happened
        task_id: Optional task identifier (pass empty string if not needed)
        tags: Optional list of tags for categorization
        data: Optional additional structured data

    Returns:
        dict with ok=True and episode_id on success
    """
    return await memory.add_episode(
        session_id=session_id,
        task_id=task_id if task_id else None,
        title=title,
        summary=summary,
        tags=tags,
        data=data,
    )


@mcp.tool()
@_with_db_ready
async def memory_list_episodes(
    session_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List episodes for a session."""
    return await memory.list_episodes(session_id, limit=limit)


@mcp.tool()
@_with_db_ready
async def memory_search_episodes(
    session_id: str,
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search episodes by query."""
    return await memory.search_episodes(session_id, query, limit=limit)


@mcp.tool()
@_with_db_ready
async def memory_consolidate(
    session_id: str,
    dry_run: bool = True,
    max_lessons: int = 10,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Consolidate recent episodes into lessons and preferences."""
    return await memory.consolidate(
        session_id=session_id,
        dry_run=dry_run,
        max_lessons=max_lessons,
        use_llm=use_llm,
    )


@mcp.tool()
@_with_db_ready
async def memory_index_workspace(
    root_path: str = "/workspace",
    force: bool = False,
) -> Dict[str, Any]:
    """Index workspace files for search."""
    if force:
        return await memory.index_project_incremental(root_path, force=True)
    else:
        return await memory.ensure_indexed(root_path)


@mcp.tool()
@_with_db_ready
async def memory_health() -> Dict[str, Any]:
    """Get memory system health status."""
    return await memory.health()


@mcp.tool()
@_with_db_ready
async def memory_ttl_cleanup(dry_run: bool = True) -> Dict[str, Any]:
    """Cleanup expired lessons and episodes based on TTL."""
    from core.memory_sqlite import memory_sql

    return await memory_sql.ttl_cleanup(dry_run=dry_run)


@mcp.tool()
@_with_db_ready
async def memory_add_procedure(
    key: str,
    title: str,
    steps: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add or update a procedural memory entry (how-to)."""
    await memory.add_procedure(key=key, title=title, steps=steps, metadata=metadata)
    return {"ok": True, "key": key}


@mcp.tool()
@_with_db_ready
async def memory_get_procedure(key: str) -> Dict[str, Any]:
    """Get a procedural memory entry by key."""
    item = await memory.get_procedure(key)
    if not item:
        return {"ok": False, "error": "Not found"}
    return {"ok": True, "procedure": item}


@mcp.tool()
@_with_db_ready
async def memory_search_procedures(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search procedural memory entries."""
    return await memory.search_procedures(query=query, limit=limit)


@mcp.tool()
@_with_db_ready
async def memory_add_entity(
    name: str,
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a semantic entity."""
    entity_id = await memory.add_entity(name=name, entity_type=entity_type, properties=properties)
    return {"ok": True, "id": entity_id}


@mcp.tool()
@_with_db_ready
async def memory_search_entities(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search semantic entities."""
    return await memory.search_entities(query=query, limit=limit)


@mcp.tool()
@_with_db_ready
async def memory_add_relation(
    subject_id: str,
    predicate: str,
    object_id: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a semantic relation between entities."""
    rel_id = await memory.add_relation(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        properties=properties,
    )
    return {"ok": True, "id": rel_id}


@mcp.tool()
@_with_db_ready
async def memory_get_relations(entity_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get relations for an entity."""
    return await memory.get_relations(entity_id=entity_id, limit=limit)


# --- Cross-session memory ---


@mcp.tool()
@_with_db_ready
async def cross_session_start(session_id: str, user_prompt: str = "") -> Dict[str, Any]:
    """Start a new cross-session with automatic context injection from previous sessions."""
    return await memory.cross_session_start(session_id, user_prompt)


@mcp.tool()
@_with_db_ready
async def cross_session_message(
    session_id: str, content: str, role: str = "user"
) -> Dict[str, Any]:
    """Record a message event in cross-session memory."""
    return await memory.cross_session_record_message(session_id, content, role)


@mcp.tool()
@_with_db_ready
async def cross_session_tool_use(
    session_id: str,
    tool_name: str,
    tool_input: str,
    tool_output: str,
) -> Dict[str, Any]:
    """Record a tool use event in cross-session memory."""
    return await memory.cross_session_record_tool_use(
        session_id, tool_name, tool_input, tool_output
    )


@mcp.tool()
@_with_db_ready
async def cross_session_stop(session_id: str) -> Dict[str, Any]:
    """Finalize cross-session: extract observations and generate summary."""
    return await memory.cross_session_stop(session_id)


@mcp.tool()
@_with_db_ready
async def cross_session_end(session_id: str) -> Dict[str, Any]:
    """End cross-session and cleanup."""
    return await memory.cross_session_end(session_id)


@mcp.tool()
@_with_db_ready
async def cross_session_context(user_prompt: str = "", max_tokens: int = 2000) -> Dict[str, Any]:
    """Get token-budgeted context from previous sessions for system prompt injection."""
    return await memory.cross_session_get_context(user_prompt, max_tokens)


@mcp.tool()
@_with_db_ready
async def cross_session_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search across all session memories."""
    return await memory.cross_session_search(query, limit)


@mcp.tool()
@_with_db_ready
async def cross_session_stats() -> Dict[str, Any]:
    """Get cross-session memory statistics."""
    return await memory.cross_session_stats()


@mcp.tool()
@_with_db_ready
async def cross_session_check_timeout(session_id: str) -> Dict[str, Any]:
    """Check if session has timed out and finalize if needed.

    Returns:
        dict with timeout_detected (bool), finalized (bool)
    """
    timed_out = await memory.cross_session_check_session_timeout(session_id)
    result = {"session_id": session_id, "timeout_detected": timed_out, "finalized": False}

    if timed_out:
        await memory.cross_session_stop(session_id)
        result["finalized"] = True
        result["reason"] = "inactivity_timeout"

    return result


# --- Memory consolidation (decay/merge/prune) ---


@mcp.tool()
@_with_db_ready
async def memory_consolidate_decay(dry_run: bool = True) -> Dict[str, Any]:
    """Run memory consolidation: decay old memories, merge similar, prune low-importance.

    This maintains memory quality over time by:
    - Decay: reducing importance of old memories
    - Merge: combining similar memories
    - Prune: removing low-importance/old memories

    Set dry_run=True to see what would be done without making changes.
    """
    return await memory.consolidate_memory(dry_run=dry_run)


@mcp.tool()
@_with_db_ready
async def memory_consolidation_status() -> Dict[str, Any]:
    """Get memory consolidation settings and status."""
    return await memory.get_consolidation_status()


# --- Conversations ---


@mcp.tool()
@_with_db_ready
async def conversation_add_message(
    session_id: str,
    role: str,
    content: str,
    model: Optional[str] = None,
    tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Add a message to conversation history.

    Args:
        session_id: Session ID
        role: "user", "assistant", or "system"
        content: Message content
        model: Optional model name used
        tokens: Optional token count
    """
    return await memory.add_conversation_message(session_id, role, content, model, tokens)


@mcp.tool()
@_with_db_ready
async def conversation_get_messages(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get conversation messages (newest first).

    Args:
        session_id: Session ID
        limit: Max messages to return
        offset: Number of messages to skip
    """
    return await memory.get_conversation_messages(session_id, limit, offset)


@mcp.tool()
@_with_db_ready
async def conversation_get_messages_asc(
    session_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get conversation messages (oldest first - for context injection).

    Args:
        session_id: Session ID
        limit: Max messages to return
    """
    return await memory.get_conversation_messages_asc(session_id, limit)


@mcp.tool()
@_with_db_ready
async def conversation_search(
    session_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search conversation messages by content.

    Args:
        session_id: Session ID
        query: Search query
        limit: Max results
    """
    return await memory.search_conversation(session_id, query, limit)


@mcp.tool()
@_with_db_ready
async def conversation_stats() -> Dict[str, Any]:
    """Get conversation statistics."""
    return await memory.get_conversation_stats()


# --- Knowledge Base ---


@mcp.tool()
@_with_db_ready
async def kb_add_document(
    title: str,
    content: str,
    source_type: str,
    source_url: Optional[str] = None,
    source_path: Optional[str] = None,
    format: str = "markdown",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a document to knowledge base.

    Args:
        title: Document title
        content: Document content
        source_type: "file", "url", "text"
        source_url: Source URL (for url type)
        source_path: Source file path (for file type)
        format: Format type (markdown, text, html, pdf, docx)
        session_id: Optional session ID
    """
    return await memory.kb_add_document(
        title, content, source_type, source_url, source_path, format, session_id
    )


@mcp.tool()
@_with_db_ready
async def kb_add_document_from_file(
    file_path: str,
    source_type: str = "file",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a document by parsing a file.

    Args:
        file_path: Path to the file
        source_type: Source type (file, url)
        session_id: Optional session ID
    """
    return await memory.kb_add_document_from_file(file_path, source_type, session_id)


@mcp.tool()
@_with_db_ready
async def kb_add_document_from_url(
    url: str,
    content: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a document from URL.

    Args:
        url: Source URL
        content: Optional pre-fetched content
        session_id: Optional session ID
    """
    return await memory.kb_add_document_from_url(url, content, session_id)


@mcp.tool()
@_with_db_ready
async def kb_get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Get a document by ID."""
    return await memory.kb_get_document(doc_id)


@mcp.tool()
@_with_db_ready
async def kb_list_documents(
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List documents.

    Args:
        session_id: Optional session filter
        limit: Max results
        offset: Offset
    """
    return await memory.kb_list_documents(session_id, limit, offset)


@mcp.tool()
@_with_db_ready
async def kb_search_documents(
    query: str,
    session_id: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search documents by content.

    Args:
        query: Search query
        session_id: Optional session filter
        limit: Max results
    """
    return await memory.kb_search_documents(query, session_id, limit)


@mcp.tool()
@_with_db_ready
async def kb_delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a document."""
    return await memory.kb_delete_document(doc_id)


@mcp.tool()
@_with_db_ready
async def kb_stats() -> Dict[str, Any]:
    """Get knowledge base statistics."""
    return await memory.kb_get_stats()


# --- Knowledge Graph ---


@mcp.tool()
@_with_db_ready
async def kg_add_triple(
    subject: str,
    predicate: str,
    object_name: str,
    confidence: float = 1.0,
    source_type: str = "text",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a semantic triple to knowledge graph.

    Args:
        subject: Subject entity (e.g., "John")
        predicate: Relationship (e.g., "works_for")
        object_name: Object entity (e.g., "Google")
        confidence: Confidence score (0.0 - 1.0)
        source_type: Source type ("text", "conversation", "document")
        session_id: Session identifier
    """
    return await memory.kg_add_triple(
        subject, predicate, object_name, confidence, source_type, None, session_id
    )


@mcp.tool()
@_with_db_ready
async def kg_upsert_fact(
    subject: str,
    predicate: str,
    object_name: str,
    action: str = "assert",
    confidence: float = 1.0,
    source_type: str = "text",
    session_id: Optional[str] = None,
    observed_at: Optional[str] = None,
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert temporal fact state in knowledge graph.

    Args:
        subject: Subject entity
        predicate: Relationship name
        object_name: Object entity
        action: "assert" or "retract"
        confidence: Confidence score
        source_type: Source type
        session_id: Session identifier
        observed_at: Event observation timestamp (ISO8601)
        valid_from: Optional validity start (ISO8601)
        valid_to: Optional validity end (ISO8601)
    """
    return await memory.kg_upsert_fact(
        subject=subject,
        predicate=predicate,
        object_name=object_name,
        action=action,
        confidence=confidence,
        source_type=source_type,
        source_id=None,
        session_id=session_id,
        metadata=None,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_to=valid_to,
    )


@mcp.tool()
@_with_db_ready
async def kg_get_triples(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_name: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Query triples from knowledge graph.

    Args:
        subject: Filter by subject
        predicate: Filter by predicate
        object_name: Filter by object
        session_id: Filter by session
        limit: Max results
    """
    return await memory.kg_get_triples(subject, predicate, object_name, session_id, limit)


@mcp.tool()
@_with_db_ready
async def kg_get_triples_as_of(
    as_of: Optional[str] = None,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_name: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Query triples valid at a given point in time."""
    return await memory.kg_get_triples_as_of(
        as_of=as_of,
        subject=subject,
        predicate=predicate,
        object_name=object_name,
        session_id=session_id,
        limit=limit,
    )


@mcp.tool()
@_with_db_ready
async def kg_get_fact_history(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_name: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get chronological event history for facts in knowledge graph."""
    return await memory.kg_get_fact_history(
        subject=subject,
        predicate=predicate,
        object_name=object_name,
        session_id=session_id,
        limit=limit,
    )


@mcp.tool()
@_with_db_ready
async def kg_get_entity_timeline_summary(
    entity: str,
    predicate: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Get aggregated temporal timeline summary for an entity."""
    return await memory.kg_get_entity_timeline_summary(
        entity=entity,
        predicate=predicate,
        session_id=session_id,
        limit=limit,
    )


@mcp.tool()
@_with_db_ready
async def kg_get_neighbors(
    entity: str,
    direction: str = "both",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get neighboring entities in the graph.

    Args:
        entity: Entity to find neighbors for
        direction: "out" (subject→), "in" (→object), or "both"
        limit: Max results
    """
    return await memory.kg_get_neighbors(entity, direction, 1, limit)


@mcp.tool()
@_with_db_ready
async def kg_find_path(
    from_entity: str,
    to_entity: str,
    max_depth: int = 3,
) -> Optional[Dict[str, Any]]:
    """Find a path between two entities.

    Args:
        from_entity: Starting entity
        to_entity: Target entity
        max_depth: Maximum search depth
    """
    return await memory.kg_find_path(from_entity, to_entity, max_depth)


@mcp.tool()
@_with_db_ready
async def kg_find_path_as_of(
    from_entity: str,
    to_entity: str,
    as_of: Optional[str] = None,
    max_depth: int = 3,
) -> Optional[Dict[str, Any]]:
    """Find a path between two entities at a given timestamp."""
    return await memory.kg_find_path_as_of(from_entity, to_entity, as_of, max_depth)


@mcp.tool()
@_with_db_ready
async def kg_search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search entities in knowledge graph.

    Args:
        query: Search query
        entity_type: Filter by type (person, organization, etc.)
        limit: Max results
    """
    return await memory.kg_search_entities(query, entity_type, limit)


@mcp.tool()
@_with_db_ready
async def kg_get_entity_facts(
    entity: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get all facts about an entity.

    Args:
        entity: Entity name
        limit: Max results
    """
    return await memory.kg_get_entity_facts(entity, limit)


@mcp.tool()
@_with_db_ready
async def kg_stats() -> Dict[str, Any]:
    """Get knowledge graph statistics."""
    return await memory.kg_get_stats()


# --- Memory Extraction ---


@mcp.tool()
@_with_db_ready
async def extract_memories(
    text: str,
    entity_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract memories and triples from text automatically.

    Extracts 8 memory types: facts, events, people, preferences,
    relationships, rules, skills, attributes.

    Args:
        text: Text to extract from
        entity_id: Entity ID for attribution
        session_id: Session ID for attribution
    """
    return await memory.extract_memories(text, entity_id, session_id)


@mcp.tool()
@_with_db_ready
async def get_extracted_memories(
    entity_id: Optional[str] = None,
    memory_type: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get extracted memories.

    Args:
        entity_id: Filter by entity
        memory_type: Filter by type (facts, events, people, etc.)
        session_id: Filter by session
        limit: Max results
    """
    return await memory.get_extracted_memories(entity_id, memory_type, session_id, limit)


@mcp.tool()
@_with_db_ready
async def search_extracted_memories(
    query: str,
    entity_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search extracted memories by content.

    Args:
        query: Search query
        entity_id: Filter by entity
        limit: Max results
    """
    return await memory.search_extracted_memories(query, entity_id, limit)


@mcp.tool()
@_with_db_ready
async def extraction_stats() -> Dict[str, Any]:
    """Get extraction statistics."""
    return await memory.get_extraction_stats()


@mcp.tool()
@_with_db_ready
async def memory_metrics() -> Dict[str, Any]:
    """Get runtime metrics for memory tools."""
    return {
        "ok": True,
        "metrics": metrics.get_stats(),
        "requests": request_metrics.get_stats(),
    }


@mcp.tool()
@_with_db_ready
async def memory_correct(
    key: str,
    value: Any,
    memory_type: str = "preference",
    scope: str = "global",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Direct correction of a memory entry by key.

    Use this to update or replace an existing memory entry.
    - key: The memory key to correct
    - value: The new value to set
    - memory_type: 'preference' or 'lesson'
    - scope: 'global' or 'session' (for preferences)
    - session_id: Required for session-scoped preferences
    """
    return await memory.memory_correct(
        key=key,
        value=value,
        memory_type=memory_type,
        scope=scope,
        session_id=session_id,
    )


@mcp.tool()
@_with_db_ready
async def memory_feedback(
    feedback: str,
    session_id: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Process natural language feedback to correct/update memory.

    Uses LLM to parse feedback and determine what needs to be changed.
    Falls back to rule-based parsing if LLM unavailable.

    Examples:
    - "I don't like coffee, I prefer tea"
    - "I work at Google now"
    - "Forget that I hate dogs"

    Args:
        feedback: Natural language feedback
        session_id: Optional session for session-scoped preferences
        use_llm: Whether to use LLM for parsing (default True)
    """
    return await memory.memory_feedback(
        feedback=feedback,
        session_id=session_id,
        use_llm=use_llm,
    )
