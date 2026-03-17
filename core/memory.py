from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import settings
from .redact import redact_dict
from .memory_sqlite import memory_sql, MemorySearchHit
from .db import db_backend_info
from .prefs_infer import infer_preferences_from_text
from .memory_consolidation import propose_from_episodes
from .vector_memory import vector_memory, VectorHit
from .embeddings import EmbeddingsError
from .llm_config import llm_settings
from .llm.client import llm_client
from .search import hybrid_search as hs
from .search_config import search_settings
from .health import health_monitor
from .cross_session import cross_session_manager
from .memory_decay import memory_consolidator
from .conversations import conversation_store
from .knowledge_base import knowledge_base
from .knowledge_graph import knowledge_graph
from .memory_extractor import memory_extractor


@dataclass
class MemoryHit:
    source: str
    text: str
    score: float
    meta: Dict[str, Any]


class MemoryStore:
    def __init__(self) -> None:
        self._task_complete_hooks: List[Callable] = []
        self._last_consolidation: Dict[str, float] = {}

    async def register_task_complete_hook(self, hook: Callable) -> None:
        """Register a hook to be called when a task completes."""
        self._task_complete_hooks.append(hook)

    async def _on_task_complete(self, session_id: str, task_id: str, result: Dict[str, Any]) -> None:
        """Called when a task completes. Triggers auto-consolidation if enabled."""
        if not session_id:
            return

        # Check if auto-consolidation is enabled
        if not getattr(settings, "auto_consolidate_on_task_complete", False):
            return

        # Check minimum interval
        last = self._last_consolidation.get(session_id, 0.0)
        now = time.time()
        min_interval = int(getattr(settings, "auto_consolidate_min_interval_s", 1800))
        if now - last < min_interval:
            return

        # Trigger consolidation
        try:
            await self.add_episode(
                session_id=session_id,
                task_id=task_id,
                title="Task completed",
                summary=f"Task {task_id} completed with result: {result.get('status', 'unknown')}",
                tags=["task_complete"],
                data={"task_id": task_id, "result": result},
            )
            await self.consolidate(
                session_id=session_id,
                dry_run=False,
                episode_limit=int(getattr(settings, "auto_consolidate_episode_limit", 50)),
                max_lessons=int(getattr(settings, "auto_consolidate_max_lessons", 10)),
                include_preferences=bool(getattr(settings, "auto_consolidate_include_preferences", True)),
                preferences_scope=str(getattr(settings, "auto_consolidate_preferences_scope", "session")),
            )
            self._last_consolidation[session_id] = now
        except Exception:
            # Never block task completion
            pass

        # Call registered hooks
        for hook in self._task_complete_hooks:
            try:
                await hook(session_id, task_id, result)
            except Exception:
                pass

    async def _on_session_end(self, session_id: str) -> None:
        """Called when a session ends. Triggers final consolidation."""
        if not session_id:
            return

        # Always consolidate at session end (not subject to interval limits)
        try:
            # Add final episode
            await self.add_episode(
                session_id=session_id,
                task_id=None,
                title="Session ended",
                summary=f"Session {session_id} completed",
                tags=["session_end"],
                data={"session_id": session_id},
            )

            # Final consolidation
            await self.consolidate(
                session_id=session_id,
                dry_run=False,
                episode_limit=int(getattr(settings, "auto_consolidate_episode_limit", 50)),
                max_lessons=int(getattr(settings, "auto_consolidate_max_lessons", 10)),
                include_preferences=bool(getattr(settings, "auto_consolidate_include_preferences", True)),
                preferences_scope=str(getattr(settings, "auto_consolidate_preferences_scope", "session")),
            )

            # Trigger memory consolidation (decay/merge/prune)
            await self.consolidate_memory(dry_run=False)
        except Exception:
            # Never block session ending
            pass

    async def ensure_indexed(self, root_path: str) -> Dict[str, Any]:
        """Incrementally index the workspace into SQLite FTS.

        This is called from retrieval on-demand, so it must be fast and safe:
        - Uses DB-stored file metadata to skip unchanged files
        - Throttled to avoid scanning on every request
        """
        run_at = datetime.now(timezone.utc).isoformat()
        try:
            res = await memory_sql.index_workspace_incremental(
                root=root_path,
                max_file_bytes=int(getattr(settings, "workspace_fts_max_file_bytes", 512_000)),
                max_files=int(getattr(settings, "workspace_fts_autorefresh_max_files", 500)),
                max_seconds=float(getattr(settings, "workspace_fts_autorefresh_max_seconds", 1.5)),
                min_interval_seconds=float(
                    getattr(settings, "workspace_fts_min_interval_seconds", 120.0)
                ),
                force=False,
                prune_missing=False,
            )

            # Attach lightweight counts for debugging/UI.
            try:
                st = await memory_sql.workspace_status()
                res["doc_count"] = st.get("doc_count", 0)
                res["meta_count"] = st.get("meta_count", 0)
            except Exception:
                pass

            already = bool(res.get("throttled")) or int(res.get("indexed", 0) or 0) == 0
            res["already_indexed"] = already
            res["status"] = "ok"

            # Best-effort observability: record last run (compact result) for UI/debugging.
            try:
                compact = {
                    "indexed": res.get("indexed"),
                    "changed": res.get("changed"),
                    "removed": res.get("removed"),
                    "errors": res.get("errors"),
                    "throttled": res.get("throttled"),
                    "already_indexed": res.get("already_indexed"),
                    "doc_count": res.get("doc_count"),
                    "meta_count": res.get("meta_count"),
                }
                await memory_sql.set_maintenance_state(
                    "workspace_fts:last", {"run_at": run_at, "ok": True, "result": compact}
                )
            except Exception:
                pass

            return res
        except Exception as e:
            try:
                await memory_sql.set_maintenance_state(
                    "workspace_fts:last", {"run_at": run_at, "ok": False, "error": str(e)}
                )
                await memory_sql.set_maintenance_state(
                    "workspace_fts:last_error", {"run_at": run_at, "error": str(e)}
                )
            except Exception:
                pass
            raise

    async def index_project(self, root_path: str) -> Dict[str, Any]:
        return await memory_sql.index_workspace(root=root_path)

    async def index_project_incremental(
        self, root_path: str, *, force: bool = False
    ) -> Dict[str, Any]:
        return await memory_sql.index_workspace_incremental(
            root=root_path, force=force, prune_missing=bool(force)
        )

    async def workspace_status(self) -> Dict[str, Any]:
        return await memory_sql.workspace_status()

    async def clear_workspace_index(self) -> Dict[str, Any]:
        return await memory_sql.clear_workspace_index()

    async def upsert_doc(self, path: str, content: str) -> Dict[str, Any]:
        return await memory_sql.upsert_doc(path=path, content=content)

    async def get_doc(self, path: str) -> Optional[Dict[str, Any]]:
        return await memory_sql.get_doc(path=path)

    async def get_docs(self, paths: List[str], limit_chars: int = 12000) -> List[Dict[str, Any]]:
        return await memory_sql.get_docs(paths=paths, limit_chars=limit_chars)

    async def semantic_search(self, query: str, limit: int = 5) -> List[MemoryHit]:
        hits: List[VectorHit] = await vector_memory.semantic_search(query, limit=limit)
        out: List[MemoryHit] = []
        for h in hits:
            out.append(
                MemoryHit(
                    source="vector",
                    text=h.snippet,
                    score=h.score,
                    meta={"path": h.path, "chunk_index": h.chunk_index},
                )
            )
        return out

    async def search(self, query: str, limit: int = 5) -> List[MemoryHit]:
        hits: List[MemorySearchHit] = await memory_sql.search(query, limit=limit)
        out: List[MemoryHit] = []
        for h in hits:
            out.append(
                MemoryHit(source="fts", text=h.snippet, score=-h.rank, meta={"path": h.path})
            )
        return out

    async def search_hybrid(
        self,
        query: str,
        limit: int = 8,
        fts_limit: int = 24,
        vec_limit: int = 24,
    ) -> List[MemoryHit]:
        """Hybrid search combining FTS, BM25, and vector search.

        Uses BM25 for better ranking, query expansion, and optional LLM re-ranking.
        """
        if not query:
            return []

        fts_limit = max(1, int(fts_limit))
        vec_limit = max(1, int(vec_limit))
        limit = max(1, int(limit))

        fts_hits: List[MemorySearchHit] = await memory_sql.search(query, limit=fts_limit)
        fts_results = []
        for h in fts_hits:
            fts_results.append(
                {
                    "path": h.path,
                    "text": h.snippet,
                    "score": -h.rank,
                    "meta": {"path": h.path},
                }
            )

        vec_hits: List[VectorHit] = []
        if search_settings.vector_enabled:
            try:
                vec_hits = await vector_memory.semantic_search(query, limit=vec_limit)
            except EmbeddingsError:
                vec_hits = []
            except Exception:
                vec_hits = []

        vec_results = []
        for h in vec_hits:
            vec_results.append(
                {
                    "path": h.path,
                    "text": h.snippet,
                    "score": h.score,
                    "meta": {"path": h.path, "chunk_index": h.chunk_index},
                }
            )

        results = await hs.search(
            query=query,
            fts_results=fts_results,
            vector_results=vec_results,
            limit=limit,
        )

        out: List[MemoryHit] = []
        for r in results:
            out.append(
                MemoryHit(
                    source=r.get("src", "hybrid"),
                    text=r.get("text", ""),
                    score=r.get("score", 0),
                    meta=r.get("meta", {}),
                )
            )

        return out

    async def lessons_stats(self) -> Dict[str, Any]:
        return await memory_sql.lessons_stats()

    async def lessons_maintenance(
        self, *, dry_run: bool = True, mode: str = "strict"
    ) -> Dict[str, Any]:
        return await memory_sql.lessons_maintenance(dry_run=dry_run, mode=mode)

    async def memory_stats(self) -> Dict[str, Any]:
        return await memory_sql.memory_stats()

    async def health(self, session_id: str | None = None) -> Dict[str, Any]:
        """Lightweight memory health snapshot for UI/debugging.

        Best-effort: never throws.
        """
        out: Dict[str, Any] = {"ok": True}
        try:
            out["workspace"] = await self.workspace_status()
        except Exception as e:
            out["workspace_error"] = str(e)
            out["ok"] = False
        try:
            out["vectors"] = await self.vectors_status()
        except Exception as e:
            out["vectors_error"] = str(e)
        try:
            out["lessons"] = await self.lessons_stats()
        except Exception as e:
            out["lessons_error"] = str(e)
        try:
            out["db"] = await self.memory_stats()
        except Exception as e:
            out["db_error"] = str(e)
        try:
            out["db_backend"] = db_backend_info()
        except Exception:
            pass
        try:
            out["runtime_dependencies"] = await health_monitor.snapshot()
        except Exception as e:
            out["runtime_dependencies_error"] = str(e)
        # Consolidation last-run (per session)
        try:
            sid = (session_id or "").strip() if session_id is not None else ""
            if sid:
                st = await memory_sql.get_maintenance_state(f"consolidate:last:{sid}")
                if st is not None:
                    out["consolidation_last"] = st
        except Exception as e:
            out["consolidation_error"] = str(e)

        # Indexing last-run (global)
        try:
            st = await memory_sql.get_maintenance_state("workspace_fts:last")
            if st is not None:
                out["workspace_fts_last"] = st
        except Exception as e:
            out["workspace_fts_last_error"] = str(e)
        try:
            st = await memory_sql.get_maintenance_state("vectors:last")
            if st is not None:
                out["vectors_last"] = st
        except Exception as e:
            out["vectors_last_error"] = str(e)

        return out

    async def maintenance_status(
        self, session_id: str | None = None, limit: int = 25
    ) -> Dict[str, Any]:
        """Return recent maintenance/housekeeping state for UI/debug (best-effort)."""
        sid = (session_id or "").strip() if session_id else ""
        try:
            limit_i = max(1, int(limit))
        except Exception:
            limit_i = 25

        out: Dict[str, Any] = {"ok": True}
        try:
            out["housekeep_global"] = await memory_sql.get_maintenance_state("housekeep:last")
        except Exception as e:
            out["housekeep_global_error"] = str(e)
        # Indexing/refresh bookkeeping (global)
        try:
            out["workspace_fts_last"] = await memory_sql.get_maintenance_state("workspace_fts:last")
        except Exception as e:
            out["workspace_fts_last_error"] = str(e)
        try:
            out["vectors_last"] = await memory_sql.get_maintenance_state("vectors:last")
        except Exception as e:
            out["vectors_last_error"] = str(e)

        # Browser artifacts housekeeping (filesystem)
        try:
            out["browser_artifacts_last"] = await memory_sql.get_maintenance_state(
                "browser_artifacts:last"
            )
        except Exception as e:
            out["browser_artifacts_last_error"] = str(e)

        if sid:
            try:
                out["housekeep_session"] = await memory_sql.get_maintenance_state(
                    f"housekeep:last:{sid}"
                )
            except Exception as e:
                out["housekeep_session_error"] = str(e)
            try:
                out["consolidate_last"] = await memory_sql.get_maintenance_state(
                    f"consolidate:last:{sid}"
                )
            except Exception as e:
                out["consolidate_last_error"] = str(e)

            try:
                out["browser_artifacts_session"] = await memory_sql.get_maintenance_state(
                    f"browser_artifacts:last:{sid}"
                )
            except Exception as e:
                out["browser_artifacts_session_error"] = str(e)

        # Small 'recent' list (can be filtered by prefix on the API side if needed).
        try:
            out["recent"] = await memory_sql.list_maintenance_state("", limit=limit_i)
        except Exception:
            out["recent"] = []

        return out

    async def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 8,
        fts_limit: int = 24,
        vec_limit: int = 24,
        per_file_cap: int = 2,
        fts_weight: float = 1.0,
        vec_weight: float = 1.0,
    ) -> List[MemoryHit]:
        """Hybrid search: combine FTS keyword hits and vector semantic hits.

        Notes:
        - Scores are normalized per-source to 0..1 then combined.
        - Results are capped per file to avoid 1 giant file dominating.
        """
        q = (query or "").strip()
        if not q:
            return []

        # Pull a slightly larger pool and then trim.
        fts_limit = max(1, int(fts_limit))
        vec_limit = max(1, int(vec_limit))
        limit = max(1, int(limit))
        per_file_cap = max(1, int(per_file_cap))

        fts_hits = await self.search(q, limit=fts_limit)
        try:
            vec_hits = await self.semantic_search(q, limit=vec_limit)
        except EmbeddingsError:
            vec_hits = []
        except Exception:
            # Any vector failure should not break hybrid search; fall back to FTS only.
            vec_hits = []

        def _norm_scores(vals: List[float]) -> Dict[float, float]:
            if not vals:
                return {}
            mn = min(vals)
            mx = max(vals)
            if mx == mn:
                return {v: 1.0 for v in vals}
            out: Dict[float, float] = {}
            for v in vals:
                out[v] = (float(v) - float(mn)) / (float(mx) - float(mn))
            return out

        fts_norm = _norm_scores([float(h.score) for h in fts_hits])
        vec_norm = _norm_scores([float(h.score) for h in vec_hits])

        # Query tokens for small filename/path bonus.
        import re

        # Unicode-friendly tokenization (language-agnostic).
        # NOTE: `[^\W_]` ~= unicode "word" chars excluding underscore.
        # This covers Latin/Cyrillic/Greek/Arabic/Hebrew/etc. much better than
        # a hard-coded alphabet range.
        toks = re.findall(r"[^\W_]{3,}", q.lower(), flags=re.UNICODE)
        # Clamp to avoid pathological token sets on long prompts.
        if len(toks) > 40:
            toks = toks[:40]
        tokset = set(toks)

        candidates: List[Dict[str, Any]] = []
        for h in fts_hits:
            candidates.append(
                {
                    "path": str(h.meta.get("path") or ""),
                    "text": h.text,
                    "fts_score": float(h.score),
                    "vec_score": None,
                    "chunk_index": None,
                    "src": "fts",
                }
            )
        for h in vec_hits:
            candidates.append(
                {
                    "path": str(h.meta.get("path") or ""),
                    "text": h.text,
                    "fts_score": None,
                    "vec_score": float(h.score),
                    "chunk_index": int(h.meta.get("chunk_index") or 0),
                    "src": "vector",
                }
            )

        # Optional: recency bonus based on workspace file mtimes (if available).
        recency_enabled = bool(getattr(settings, "retrieval_hybrid_recency_enabled", True))
        recency_half_life_days = float(
            getattr(settings, "retrieval_hybrid_recency_half_life_days", 30.0) or 30.0
        )
        recency_max_bonus = float(
            getattr(settings, "retrieval_hybrid_recency_max_bonus", 0.12) or 0.12
        )
        recency_max_age_days = float(
            getattr(settings, "retrieval_hybrid_recency_max_age_days", 365.0) or 365.0
        )
        recency_half_life_days = max(0.0, recency_half_life_days)
        recency_max_bonus = max(0.0, recency_max_bonus)
        recency_max_age_days = max(0.0, recency_max_age_days)

        mtime_map: Dict[str, float] = {}
        now_ts = datetime.now(timezone.utc).timestamp()
        import math

        if recency_enabled and recency_max_bonus > 0.0 and recency_half_life_days > 0.0:
            try:
                paths2 = [
                    str(c.get("path") or "").strip()
                    for c in candidates
                    if str(c.get("path") or "").strip()
                ]
                mtime_map = await memory_sql.get_workspace_file_mtimes(paths2)
            except Exception:
                mtime_map = {}
        out: List[MemoryHit] = []
        # Compute combined scores.
        scored: List[Dict[str, Any]] = []
        for c in candidates:
            path = c.get("path") or ""
            if not path:
                continue
            fts_s = c.get("fts_score")
            vec_s = c.get("vec_score")
            fn = 0.0
            vn = 0.0
            if isinstance(fts_s, (int, float)):
                fn = float(fts_norm.get(float(fts_s), 0.0))
            if isinstance(vec_s, (int, float)):
                vn = float(vec_norm.get(float(vec_s), 0.0))

            bonus = 0.0
            pl = path.lower()
            # Tiny bonus if path contains query-ish tokens.
            if tokset and any(t in pl for t in tokset):
                bonus += 0.15

            # Recency bias: prefer recently changed workspace files (best-effort).
            recency_bonus = 0.0
            age_days = None
            try:
                mt = float(mtime_map.get(str(path), 0.0) or 0.0)
                if mt > 0.0:
                    age_days = max(0.0, (float(now_ts) - mt) / 86400.0)
                    if age_days <= float(recency_max_age_days):
                        recency_bonus = float(recency_max_bonus) * math.exp(
                            -age_days / float(recency_half_life_days)
                        )
                        bonus += recency_bonus
            except Exception:
                pass

            score = float(fts_weight) * fn + float(vec_weight) * vn + bonus
            scored.append(
                {
                    **c,
                    "score": score,
                    "fts_norm": fn,
                    "vec_norm": vn,
                    "bonus": bonus,
                    "recency_bonus": recency_bonus,
                    "age_days": age_days,
                }
            )

        scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        # Optional: filter out extremely low-score candidates to reduce noisy context.
        # This is conservative: if filtering would drop everything, we keep the original list.
        min_abs = float(getattr(settings, "retrieval_hybrid_min_score", 0.0) or 0.0)
        min_rel = float(getattr(settings, "retrieval_hybrid_min_rel_score", 0.0) or 0.0)
        min_abs = max(0.0, float(min_abs))
        min_rel = max(0.0, float(min_rel))
        if scored and (min_abs > 0.0 or min_rel > 0.0):
            try:
                top_score = float(scored[0].get("score") or 0.0)
            except Exception:
                top_score = 0.0
            filtered = []
            for c2 in scored:
                try:
                    s2 = float(c2.get("score") or 0.0)
                except Exception:
                    s2 = 0.0
                if min_abs > 0.0 and s2 < min_abs:
                    continue
                if min_rel > 0.0 and top_score > 0.0 and s2 < (top_score * min_rel):
                    continue
                filtered.append(c2)
            if filtered:
                scored = filtered

        # Optional: diversity-aware re-ranking (MMR-like).
        # This reduces near-duplicate snippets from the same or similar files,
        # making retrieval more robust on large repos.
        use_mmr = bool(getattr(settings, "retrieval_hybrid_use_mmr", True))
        mmr_lambda = float(getattr(settings, "retrieval_hybrid_mmr_lambda", 0.70) or 0.70)
        mmr_lambda = max(0.0, min(1.0, mmr_lambda))
        max_candidates = int(getattr(settings, "retrieval_hybrid_mmr_max_candidates", 120) or 120)

        import math

        def _tokset(s2: str) -> set[str]:
            # Keep it cheap: tokenise a bounded prefix.
            ss = (s2 or "")[:800].lower()
            toks2 = re.findall(r"[^\W_]{2,}", ss, flags=re.UNICODE)
            # Clamp to avoid pathological memory use.
            if len(toks2) > 120:
                toks2 = toks2[:120]
            return set(toks2)

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            if inter <= 0:
                return 0.0
            uni = len(a | b)
            if uni <= 0:
                return 0.0
            return float(inter) / float(uni)

        # Precompute token sets for MMR similarity.
        if use_mmr:
            for c in scored[: (max_candidates if max_candidates > 0 else len(scored))]:
                c["_tok"] = _tokset(str(c.get("text") or "") + " " + str(c.get("path") or ""))

        def _select(scored_list: list[dict]) -> list[dict]:
            # Enforce per-file cap during selection.
            if (not use_mmr) or mmr_lambda >= 0.999:
                chosen2: list[dict] = []
                per_file2: Dict[str, int] = {}
                for c2 in scored_list:
                    if len(chosen2) >= limit:
                        break
                    path2 = str(c2.get("path") or "")
                    if not path2:
                        continue
                    n2 = per_file2.get(path2, 0)
                    if n2 >= per_file_cap:
                        continue
                    per_file2[path2] = n2 + 1
                    chosen2.append(c2)
                return chosen2

            cand2 = scored_list[: (max_candidates if max_candidates > 0 else len(scored_list))]
            chosen2: list[dict] = []
            chosen_toks: list[set[str]] = []
            per_file2: Dict[str, int] = {}

            # Seed with the top-scoring candidate.
            for c2 in cand2:
                path2 = str(c2.get("path") or "")
                if not path2:
                    continue
                n2 = per_file2.get(path2, 0)
                if n2 >= per_file_cap:
                    continue
                chosen2.append(c2)
                chosen_toks.append(c2.get("_tok") or set())
                per_file2[path2] = n2 + 1
                break

            # Greedy MMR selection.
            while len(chosen2) < limit:
                best = None
                best_val = -math.inf
                for c2 in cand2:
                    if c2 in chosen2:
                        continue
                    path2 = str(c2.get("path") or "")
                    if not path2:
                        continue
                    n2 = per_file2.get(path2, 0)
                    if n2 >= per_file_cap:
                        continue

                    rel = float(c2.get("score") or 0.0)
                    toks2 = c2.get("_tok") or set()
                    max_sim = 0.0
                    for st in chosen_toks:
                        sim = _jaccard(toks2, st)
                        if sim > max_sim:
                            max_sim = sim
                    val = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                    if val > best_val:
                        best_val = val
                        best = c2

                if best is None:
                    break
                chosen2.append(best)
                chosen_toks.append(best.get("_tok") or set())
                bp = str(best.get("path") or "")
                per_file2[bp] = per_file2.get(bp, 0) + 1

            return chosen2

        selected = _select(scored)

        # Post-filter: dedupe near-identical snippets.
        # Even with per-file caps + MMR, overlaps between FTS and vector chunks
        # can still leak through (especially after excerpt expansion).
        dedupe_enabled = bool(getattr(settings, "retrieval_hybrid_dedupe_enabled", True))
        dedupe_thr = float(getattr(settings, "retrieval_hybrid_dedupe_sim_threshold", 0.92) or 0.92)
        dedupe_thr = max(0.0, min(0.999, dedupe_thr))
        dedupe_max_compare = int(getattr(settings, "retrieval_hybrid_dedupe_max_compare", 60) or 60)
        dedupe_max_compare = max(1, min(500, dedupe_max_compare))

        import hashlib

        def _norm_text(s3: str) -> str:
            ss3 = (s3 or "").strip().lower()
            # Collapse whitespace and clamp.
            ss3 = re.sub(r"\s+", " ", ss3)
            return ss3[:1200]

        def _sig_hash(s3: str) -> str:
            return hashlib.sha1(_norm_text(s3).encode("utf-8", errors="ignore")).hexdigest()

        def _dedupe_fill(pool_primary: list[dict], pool_fallback: list[dict]) -> list[dict]:
            if not dedupe_enabled:
                return pool_primary

            out2: list[dict] = []
            seen_hash: set[str] = set()
            seen_toks2: list[set[str]] = []
            per_file2: Dict[str, int] = {}
            seen_key: set[tuple] = set()

            def _try_add(c2: dict) -> bool:
                if len(out2) >= limit:
                    return False
                path2 = str(c2.get("path") or "")
                if not path2:
                    return False
                n2 = per_file2.get(path2, 0)
                if n2 >= per_file_cap:
                    return False

                ck = (
                    path2,
                    int(c2.get("chunk_index") or 0),
                    str(c2.get("src") or ""),
                    _sig_hash(str(c2.get("text") or "")),
                )
                if ck in seen_key:
                    return False

                txt = str(c2.get("text") or "")
                h = _sig_hash(txt)
                if h in seen_hash:
                    return False

                toks2 = c2.get("_tok")
                if toks2 is None:
                    toks2 = _tokset(txt + " " + path2)

                # Near-duplicate check against a bounded recent set.
                if seen_toks2:
                    start = max(0, len(seen_toks2) - dedupe_max_compare)
                    for st in seen_toks2[start:]:
                        if _jaccard(toks2, st) >= dedupe_thr:
                            return False

                seen_key.add(ck)
                seen_hash.add(h)
                seen_toks2.append(toks2)
                per_file2[path2] = n2 + 1
                out2.append(c2)
                return True

            for c2 in pool_primary:
                _try_add(c2)
                if len(out2) >= limit:
                    return out2

            for c2 in pool_fallback:
                # Skip items already present by identity to keep ordering stable.
                if c2 in pool_primary:
                    continue
                _try_add(c2)
                if len(out2) >= limit:
                    break

            return out2

        # Ensure token sets exist for selected if we later do near-dup checks.
        if dedupe_enabled:
            for c in selected:
                if c.get("_tok") is None:
                    c["_tok"] = _tokset(str(c.get("text") or "") + " " + str(c.get("path") or ""))

        selected = _dedupe_fill(selected, scored)

        for c in selected:
            path = str(c.get("path") or "")
            if not path:
                continue

            meta: Dict[str, Any] = {
                "path": path,
                "src": c.get("src"),
                "fts_score": c.get("fts_score"),
                "vec_score": c.get("vec_score"),
                "fts_norm": c.get("fts_norm"),
                "vec_norm": c.get("vec_norm"),
                "bonus": c.get("bonus"),
                "recency_bonus": c.get("recency_bonus"),
                "age_days": c.get("age_days"),
            }
            if c.get("chunk_index") is not None:
                meta["chunk_index"] = c.get("chunk_index")

            out.append(
                MemoryHit(
                    source="hybrid",
                    text=str(c.get("text") or ""),
                    score=float(c.get("score") or 0.0),
                    meta=meta,
                )
            )

        return out

    async def index_project_vectors(self, root_path: str) -> Dict[str, Any]:
        return await vector_memory.index_workspace(root_path=root_path)

    async def index_project_vectors_incremental(
        self, root_path: str, *, force: bool = False
    ) -> Dict[str, Any]:
        run_at = datetime.now(timezone.utc).isoformat()
        try:
            res = await vector_memory.index_workspace_incremental(
                root_path=root_path, force=force, prune_missing=bool(force)
            )
            try:
                compact = {
                    "indexed": res.get("indexed"),
                    "changed": res.get("changed"),
                    "removed": res.get("removed"),
                    "errors": res.get("errors"),
                    "throttled": res.get("throttled"),
                }
                await memory_sql.set_maintenance_state(
                    "vectors:last", {"run_at": run_at, "ok": True, "result": compact}
                )
            except Exception:
                pass
            return res
        except Exception as e:
            try:
                await memory_sql.set_maintenance_state(
                    "vectors:last", {"run_at": run_at, "ok": False, "error": str(e)}
                )
                await memory_sql.set_maintenance_state(
                    "vectors:last_error", {"run_at": run_at, "error": str(e)}
                )
            except Exception:
                pass
            raise

    async def vectors_status(self) -> Dict[str, Any]:
        return await vector_memory.status()

    async def clear_vectors_index(self) -> Dict[str, Any]:
        return await vector_memory.clear_index()

    async def set_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        await memory_sql.save_session_snapshot(session_id, snapshot)

    async def get_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await memory_sql.get_session_snapshot(session_id)

    # Working memory (mutable scratchpad)
    async def set_working_memory(self, session_id: str, content: str) -> None:
        await memory_sql.set_working_memory(session_id, content)

    async def get_working_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await memory_sql.get_working_memory(session_id)

    async def append_working_memory(
        self, session_id: str, text: str, *, max_chars: int = 12000
    ) -> Dict[str, Any]:
        return await memory_sql.append_working_memory(session_id, text, max_chars=max_chars)

    async def clear_working_memory(self, session_id: str) -> None:
        await memory_sql.clear_working_memory(session_id)

    async def add_lesson(
        self, key: str, lesson: str, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        await memory_sql.add_lesson(key, lesson, meta=meta)

    async def list_lessons(self, limit: int = 50) -> List[Dict[str, Any]]:
        return await memory_sql.list_lessons(limit=limit)

    async def search_lessons(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        return await memory_sql.search_lessons(query=query, limit=limit)

    # Procedural Memory (how to do X)
    async def add_procedure(
        self,
        key: str,
        title: str,
        steps: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a procedure (how-to)."""
        await memory_sql.add_procedure(key, title, steps, metadata=metadata)

    async def get_procedure(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a procedure by key."""
        return await memory_sql.get_procedure(key)

    async def search_procedures(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search procedures by title or content."""
        return await memory_sql.search_procedures(query, limit=limit)

    async def list_procedures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all procedures."""
        return await memory_sql.list_procedures(limit=limit)

    async def delete_procedure(self, key: str) -> Dict[str, Any]:
        """Delete a procedure."""
        return await memory_sql.delete_procedure(key)

    # Semantic Memory (entities and relations)
    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add or update an entity."""
        return await memory_sql.add_entity(name, entity_type, properties=properties)

    async def get_entity(self, id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        return await memory_sql.get_entity(id)

    async def search_entities(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search entities by name or type."""
        return await memory_sql.search_entities(query, limit=limit)

    async def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a relation between entities."""
        return await memory_sql.add_relation(
            subject_id, predicate, object_id, properties=properties
        )

    async def get_relations(self, entity_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all relations for an entity."""
        return await memory_sql.get_relations(entity_id, limit=limit)

    # Preferences (durable key/value)
    async def set_preference(
        self,
        *,
        scope: str,
        session_id: Optional[str],
        key: str,
        value: Any,
        source: str = "auto",
        is_locked: Optional[bool] = None,
        updated_by: str = "auto",
    ) -> Dict[str, Any]:
        return await memory_sql.set_preference(
            scope=scope,
            session_id=session_id,
            key=key,
            value=value,
            source=source,
            is_locked=is_locked,
            updated_by=updated_by,
        )

    async def get_preference(
        self, *, scope: str, session_id: Optional[str], key: str
    ) -> Optional[Dict[str, Any]]:
        return await memory_sql.get_preference(scope=scope, session_id=session_id, key=key)

    async def list_preferences(
        self, *, scope: str, session_id: Optional[str], prefix: str = "", limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await memory_sql.list_preferences(
            scope=scope, session_id=session_id, prefix=prefix, limit=limit
        )

    async def delete_preference(
        self, *, scope: str, session_id: Optional[str], key: str
    ) -> Dict[str, Any]:
        return await memory_sql.delete_preference(scope=scope, session_id=session_id, key=key)

    async def auto_capture_preferences(
        self, *, session_id: str, text: str, scope: str = "global"
    ) -> Dict[str, Any]:
        """Best-effort capture of durable preferences from a user prompt.

        This is conservative and allow-list based (see prefs_infer.py). It should never
        throw, and it should not block task creation.
        """
        applied: List[Dict[str, Any]] = []
        try:
            pairs = infer_preferences_from_text(text)
            for key, value in pairs:
                sid = session_id if (scope or "").strip().lower() == "session" else None
                res = await self.set_preference(
                    scope=scope,
                    session_id=sid,
                    key=key,
                    value=value,
                    source="auto",
                    is_locked=False,
                    updated_by="auto_capture",
                )
                if res.get("ok"):
                    applied.append({"key": key, "value": value, "scope": scope})
        except Exception:
            # Swallow errors by design; caller may log.
            return {"ok": False, "applied": []}
        return {"ok": True, "applied": applied}

    async def maybe_capture_preferences_from_text(
        self, *, session_id: str, text: str
    ) -> Dict[str, Any]:
        """Best-effort capture of durable user preferences.

        This is intentionally conservative and uses a small allow-list heuristic.
        It never throws and returns a summary of applied prefs.
        """
        try:
            inferred = infer_preferences_from_text(text)
            if not inferred:
                return {"ok": True, "applied": 0, "keys": []}
            applied_keys: List[str] = []
            for k, v in inferred:
                # Preferences are treated as global by default.
                res = await self.set_preference(
                    scope="global",
                    session_id=None,
                    key=k,
                    value=v,
                    source="auto",
                    is_locked=False,
                    updated_by="maybe_capture",
                )
                if res.get("ok"):
                    applied_keys.append(k)
            return {"ok": True, "applied": len(applied_keys), "keys": applied_keys}
        except Exception:
            return {"ok": False, "applied": 0, "keys": []}

    async def add_episode(
        self,
        *,
        session_id: str,
        task_id: Optional[str],
        title: str,
        summary: str,
        tags: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
        index_into_fts: bool = True,
    ) -> Dict[str, Any]:
        return await memory_sql.add_episode(
            session_id=session_id,
            task_id=task_id,
            title=title,
            summary=summary,
            tags=tags,
            data=data,
            index_into_fts=index_into_fts,
        )

    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        return await memory_sql.get_episode(episode_id)

    async def list_episodes(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return await memory_sql.list_episodes(session_id, limit=limit)

    async def search_episodes(
        self, session_id: str, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        return await memory_sql.search_episodes(session_id, query, limit=limit)

    async def save_session_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        await memory_sql.save_session_snapshot(session_id, snapshot)

    async def get_session_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await memory_sql.get_session_snapshot(session_id)

    async def housekeep(self, *, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Best-effort DB housekeeping.

        This should never throw or block critical flows.
        """
        out: Dict[str, Any] = {}
        try:
            out.update(await memory_sql.housekeep(session_id=session_id))
        except Exception as e:
            out["ok"] = False
            out["db_housekeep_error"] = str(e)

        # Filesystem housekeeping (browser artifacts) is optional and best-effort.
        try:
            from .browser_artifacts import browser_artifacts

            out["browser_artifacts"] = await browser_artifacts.housekeep(session_id=session_id)
        except Exception as e:
            out["browser_artifacts_error"] = str(e)

        # Keep the top-level ok True if DB housekeep ok or skipped.
        if "ok" not in out:
            out["ok"] = True
        return out

    async def consolidate(
        self,
        *,
        session_id: str,
        dry_run: bool = True,
        episode_limit: int = 50,
        max_lessons: int = 10,
        include_preferences: bool = True,
        preferences_scope: str = "global",
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Turn recent episodes into durable Lessons/Preferences (best-effort).

        This is a lightweight, deterministic consolidation step to keep memory useful.
        It never throws.

        If use_llm=True and LLM is available, uses LLM for consolidation.
        Otherwise falls back to simple heuristics.
        """
        sid = (session_id or "").strip()
        if not sid:
            return {"ok": False, "error": "missing session_id"}

        try:
            eps = await memory_sql.list_episodes(sid, limit=int(episode_limit))
        except Exception as e:
            return {"ok": False, "error": f"failed to list episodes: {e}"}

        if not eps:
            return {"ok": True, "lessons": [], "preferences": [], "message": "no episodes"}

        # Try LLM consolidation first
        llm_result = None
        if use_llm and llm_settings.auto_consolidate:
            try:
                llm_result = await llm_client.consolidate(eps, session_id=sid)
            except Exception:
                pass

        # Use LLM result or fallback
        if llm_result and llm_result.get("ok"):
            proposal = {
                "proposals": {
                    "lessons": llm_result.get("lessons", []),
                    "preferences": llm_result.get("preferences", []),
                },
                "summary": llm_result.get("summary", ""),
                "llm": True,
            }
        else:
            # Fallback to simple consolidation
            try:
                proposal = propose_from_episodes(
                    eps,
                    session_id=sid,
                    episode_limit=int(episode_limit),
                    max_lessons=int(max_lessons),
                    include_preferences=bool(include_preferences),
                )
            except Exception as e:
                return {"ok": False, "error": f"failed to propose consolidation: {e}"}

        proposal["dry_run"] = bool(dry_run)
        proposal["preferences_scope"] = str(preferences_scope or "global")

        if dry_run:
            return proposal

        applied = {"lessons": 0, "preferences": 0, "preference_skipped_locked": 0}

        # Apply lessons.
        for lesson_item in (proposal.get("proposals") or {}).get("lessons", []) or []:
            try:
                await self.add_lesson(
                    str(lesson_item.get("key") or ""),
                    str(lesson_item.get("value") or ""),
                    meta=lesson_item.get("meta") or {},
                )
                applied["lessons"] += 1
            except Exception:
                # Keep going.
                pass

        # Apply preferences (auto writes obey locks in memory_sql).
        if include_preferences:
            scope = (preferences_scope or "global").strip().lower() or "global"
            if scope not in ("global", "session"):
                scope = "global"
            sid_for_scope = sid if scope == "session" else None

            for p in (proposal.get("proposals") or {}).get("preferences", []) or []:
                try:
                    res = await self.set_preference(
                        scope=scope,
                        session_id=sid_for_scope,
                        key=str(p.get("key") or ""),
                        value=p.get("value"),
                        source="auto",
                        is_locked=False,
                        updated_by="consolidate",
                    )
                    if res.get("skipped"):
                        applied["preference_skipped_locked"] += 1
                    else:
                        applied["preferences"] += 1
                except Exception:
                    pass

        proposal["applied"] = applied
        return proposal

    async def export_session_memory(
        self,
        *,
        session_id: str,
        include_global_preferences: bool = True,
        include_session_preferences: bool = True,
        include_lessons: bool = True,
        include_working_memory: bool = True,
        include_snapshot: bool = True,
        include_episodes: bool = True,
        redact_secrets: bool = True,
        limit_preferences: int = 10000,
        limit_lessons: int = 10000,
        limit_episodes: int = 5000,
    ) -> Dict[str, Any]:
        """Export a compact memory snapshot for backup/migration.

        This intentionally does NOT export workspace FTS/vectors (they are derivable).
        """
        sid = (session_id or "").strip()
        if not sid:
            return {"ok": False, "error": "missing session_id"}

        exported_at = datetime.now(timezone.utc).isoformat()
        version = (getattr(settings, "version", "") or "").strip()
        if not version:
            # VERSION.txt is present in project root; backend may not know it.
            version = "unknown"

        out: Dict[str, Any] = {
            "format": "omnimind_memory_export",
            "schema": 1,
            "schema_version": 1,
            "version": version,
            "exported_at": exported_at,
            "session_id": sid,
        }

        if include_lessons:
            out["lessons"] = await memory_sql.export_lessons(limit=int(limit_lessons))
        if include_global_preferences:
            out["global_preferences"] = await memory_sql.export_preferences(
                scope="global", session_id=None, limit=int(limit_preferences)
            )
        if include_session_preferences:
            out["session_preferences"] = await memory_sql.export_preferences(
                scope="session", session_id=sid, limit=int(limit_preferences)
            )
        if include_working_memory:
            out["working_memory"] = await memory_sql.get_working_memory(sid) or {}
        if include_snapshot:
            out["snapshot"] = await memory_sql.get_session_snapshot(sid) or {}
        if include_episodes:
            out["episodes"] = await memory_sql.export_episodes(sid, limit=int(limit_episodes))

        # Safety: optionally redact common secret patterns.
        try:
            do_redact = bool(redact_secrets)
            if do_redact and bool(getattr(settings, "memory_export_redact_secrets", True)):
                out = redact_dict(out)
        except Exception:
            pass

        out.setdefault("ok", True)
        return out

    async def import_session_memory(
        self,
        *,
        session_id: str,
        export: Dict[str, Any],
        dry_run: bool = True,
        mode: str = "merge",
        allow_override_locked: bool = False,
        redact_secrets: bool = True,
    ) -> Dict[str, Any]:
        """Import a memory snapshot.

        mode:
          - merge: only add missing keys; never overwrite existing locked values
          - replace: overwrite existing values (except locked unless allow_override_locked)
        """
        sid = (session_id or "").strip()
        if not sid:
            return {"ok": False, "error": "missing session_id"}
        if not isinstance(export, dict):
            return {"ok": False, "error": "invalid export"}

        mode = (mode or "merge").strip().lower()
        if mode not in ("merge", "replace"):
            mode = "merge"

        payload = dict(export)

        warnings: List[str] = []
        # Schema validation (best-effort; we keep backward compatibility)
        schema = payload.get("schema_version")
        if schema is None:
            schema = payload.get("schema")
        try:
            schema_i = int(schema) if schema is not None else None
        except Exception:
            schema_i = None
        if schema_i is None:
            warnings.append("missing schema; assuming v1")
        elif schema_i != 1:
            warnings.append(f"unsupported schema {schema_i}; attempting best-effort import")

        try:
            if bool(redact_secrets) and bool(
                getattr(settings, "memory_import_redact_secrets", True)
            ):
                payload = redact_dict(payload)
        except Exception:
            pass

        actions = {
            "lessons_added": 0,
            "lessons_updated": 0,
            "lessons_skipped": 0,
            "prefs_added": 0,
            "prefs_updated": 0,
            "prefs_skipped": 0,
            "prefs_skipped_locked": 0,
            "wm_set": 0,
            "wm_skipped": 0,
            "snapshot_set": 0,
            "snapshot_skipped": 0,
            "episodes_added": 0,
            "episodes_skipped": 0,
        }

        # Safety: hard caps to avoid accidental huge imports
        try:
            max_lessons = int(getattr(settings, "memory_import_max_lessons", 10000) or 10000)
        except Exception:
            max_lessons = 10000
        try:
            max_prefs = int(getattr(settings, "memory_import_max_preferences", 10000) or 10000)
        except Exception:
            max_prefs = 10000
        try:
            max_eps = int(getattr(settings, "memory_import_max_episodes", 5000) or 5000)
        except Exception:
            max_eps = 5000

        # --- Lessons ---
        lessons = payload.get("lessons")
        if isinstance(lessons, list):
            for i, lesson_item in enumerate(lessons):
                if i >= max_lessons:
                    warnings.append(f"lessons truncated to {max_lessons}")
                    break
                if not isinstance(lesson_item, dict):
                    continue
                key = str(lesson_item.get("key") or "").strip()
                lesson = str(lesson_item.get("lesson") or "")
                meta = lesson_item.get("meta") if isinstance(lesson_item.get("meta"), dict) else {}
                if not key or not lesson:
                    continue

                existing = await memory_sql.get_lesson(key)
                if existing and mode == "merge":
                    actions["lessons_skipped"] += 1
                    continue
                if dry_run:
                    if existing:
                        actions["lessons_updated"] += 1
                    else:
                        actions["lessons_added"] += 1
                    continue

                try:
                    meta2 = dict(meta or {})
                    meta2.setdefault("imported", True)
                    meta2.setdefault("imported_from", payload.get("version") or "unknown")
                    await memory_sql.add_lesson(key, lesson, meta=meta2)
                    if existing:
                        actions["lessons_updated"] += 1
                    else:
                        actions["lessons_added"] += 1
                except Exception:
                    actions["lessons_skipped"] += 1

        # --- Preferences ---
        for pref_list, scope, sid_for_scope in (
            (payload.get("global_preferences"), "global", None),
            (payload.get("session_preferences"), "session", sid),
        ):
            if not isinstance(pref_list, list):
                continue
            for i, p in enumerate(pref_list):
                if i >= max_prefs:
                    warnings.append(f"preferences for scope {scope} truncated to {max_prefs}")
                    break
                if not isinstance(p, dict):
                    continue
                key = str(p.get("key") or "").strip()
                if not key:
                    continue
                value = p.get("value")
                src = str(p.get("source") or "system").strip().lower() or "system"
                if src not in ("auto", "manual", "system"):
                    src = "system"
                locked = bool(p.get("is_locked"))

                existing = await memory_sql.get_preference(
                    scope=scope, session_id=sid_for_scope, key=key
                )
                if existing:
                    if bool(existing.get("is_locked")) and not allow_override_locked:
                        actions["prefs_skipped_locked"] += 1
                        continue
                    if mode == "merge":
                        actions["prefs_skipped"] += 1
                        continue

                if dry_run:
                    if existing:
                        actions["prefs_updated"] += 1
                    else:
                        actions["prefs_added"] += 1
                    continue

                try:
                    await memory_sql.set_preference(
                        scope=scope,
                        session_id=sid_for_scope,
                        key=key,
                        value=value,
                        source=src,
                        is_locked=locked,
                        updated_by="import",
                    )
                    if existing:
                        actions["prefs_updated"] += 1
                    else:
                        actions["prefs_added"] += 1
                except Exception:
                    actions["prefs_skipped"] += 1

        # --- Working memory ---
        wm = payload.get("working_memory")
        if isinstance(wm, dict):
            content = str(wm.get("content") or "")
            existing_wm = await memory_sql.get_working_memory(sid)
            if existing_wm and mode == "merge":
                actions["wm_skipped"] += 1
            else:
                if dry_run:
                    actions["wm_set"] += 1
                else:
                    try:
                        await memory_sql.set_working_memory(sid, content)
                        actions["wm_set"] += 1
                    except Exception:
                        actions["wm_skipped"] += 1

        # --- Snapshot ---
        snap = payload.get("snapshot")
        if isinstance(snap, dict) and snap:
            existing_snap = await memory_sql.get_session_snapshot(sid)
            if existing_snap and mode == "merge":
                actions["snapshot_skipped"] += 1
            else:
                if dry_run:
                    actions["snapshot_set"] += 1
                else:
                    try:
                        await memory_sql.save_session_snapshot(sid, snap)
                        actions["snapshot_set"] += 1
                    except Exception:
                        actions["snapshot_skipped"] += 1

        # --- Episodes ---
        eps = payload.get("episodes")
        if isinstance(eps, list):
            # We never overwrite episodes; we append new "imported" ones.
            for i, e in enumerate(eps):
                if i >= max_eps:
                    warnings.append(f"episodes truncated to {max_eps}")
                    break
                if not isinstance(e, dict):
                    continue
                title = str(e.get("title") or "").strip() or "Imported episode"
                summary = str(e.get("summary") or "").strip()
                if not summary:
                    continue

                # Dedup imported episodes (avoid duplicates on repeated imports)
                try:
                    import hashlib

                    norm = (title.strip().lower() + "\n" + summary.strip()).encode(
                        "utf-8", "ignore"
                    )
                    fp = hashlib.sha256(norm).hexdigest()
                except Exception:
                    fp = ""
                if fp:
                    try:
                        exists = await memory_sql.episode_exists_by_fingerprint(
                            session_id=sid, fingerprint=fp
                        )
                    except Exception:
                        exists = False
                    if exists:
                        actions["episodes_skipped"] += 1
                        continue

                if dry_run:
                    actions["episodes_added"] += 1
                    continue

                try:
                    data = e.get("data") if isinstance(e.get("data"), dict) else {}
                    data = dict(data)
                    data.setdefault("imported", True)
                    data.setdefault("imported_from", payload.get("version") or "unknown")
                    data.setdefault("original_episode_id", e.get("id"))
                    data.setdefault("original_created_at", e.get("created_at"))
                    tags = e.get("tags") if isinstance(e.get("tags"), list) else []
                    tags2 = [str(t) for t in tags if str(t).strip()]
                    if "imported" not in tags2:
                        tags2.append("imported")
                    await memory_sql.add_episode(
                        session_id=sid,
                        task_id=e.get("task_id"),
                        title=title,
                        summary=summary,
                        tags=tags2,
                        data=data,
                        index_into_fts=False,
                    )
                    actions["episodes_added"] += 1
                except Exception:
                    actions["episodes_skipped"] += 1

        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "mode": mode,
            "actions": actions,
            "warnings": warnings,
        }

    # --- Cross-session memory ---

    async def cross_session_start(self, session_id: str, user_prompt: str = "") -> dict[str, Any]:
        """Start a new cross-session with context injection."""
        manager = cross_session_manager()
        return await manager.start_session(session_id, user_prompt)

    async def cross_session_record_message(
        self, session_id: str, content: str, role: str = "user"
    ) -> dict[str, Any]:
        """Record a message event in cross-session."""
        manager = cross_session_manager()
        return await manager.record_message(session_id, content, role)

    async def cross_session_record_tool_use(
        self, session_id: str, tool_name: str, tool_input: Any, tool_output: Any
    ) -> dict[str, Any]:
        """Record a tool use event in cross-session."""
        manager = cross_session_manager()
        return await manager.record_tool_use(session_id, tool_name, tool_input, tool_output)

    async def cross_session_stop(self, session_id: str) -> dict[str, Any]:
        """Finalize cross-session: extract observations and summary."""
        manager = cross_session_manager()
        report = await manager.stop_session(session_id)
        return {
            "ok": True,
            "session_id": report.session_id,
            "entries_stored": report.entries_stored,
            "observations_count": report.observations_count,
            "summary": report.summary,
        }

    async def cross_session_end(self, session_id: str) -> dict[str, Any]:
        """End cross-session."""
        manager = cross_session_manager()
        return await manager.end_session(session_id)

    async def cross_session_check_session_timeout(self, session_id: str) -> bool:
        """Check if session has exceeded inactivity timeout."""
        manager = cross_session_manager()
        return await manager.check_session_timeout(session_id)

    async def cross_session_get_context(
        self, user_prompt: str = "", max_tokens: Optional[int] = None
    ) -> dict[str, Any]:
        """Get token-budgeted context from previous sessions."""
        manager = cross_session_manager()
        bundle = await manager.get_context_for_prompt(user_prompt, max_tokens)
        return {
            "content": bundle.content,
            "tokens": bundle.tokens,
            "entries_count": bundle.entries_count,
            "sources": bundle.sources,
        }

    async def cross_session_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search across all session memories."""
        manager = cross_session_manager()
        return await manager.search(query, limit)

    async def cross_session_stats(self) -> dict[str, Any]:
        """Get cross-session memory statistics."""
        manager = cross_session_manager()
        return await manager.get_stats()

    # --- Memory consolidation (decay/merge/prune) ---

    async def consolidate_memory(self, dry_run: bool = True) -> dict[str, Any]:
        """Run memory consolidation: decay, merge, prune.

        Maintains memory quality by:
        - Decay: reducing importance of old memories
        - Merge: combining similar memories
        - Prune: removing low-importance memories
        """
        consolidator = memory_consolidator()
        return await consolidator.consolidate_all(dry_run=dry_run)

    async def get_consolidation_status(self) -> dict[str, Any]:
        """Get consolidation settings and status."""
        consolidator = memory_consolidator()
        return await consolidator.get_consolidation_status()

    # --- Conversations ---

    async def add_conversation_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Add a message to conversation history.

        Args:
            session_id: Session ID
            role: "user", "assistant", or "system"
            content: Message content
            model: Optional model name
            tokens: Optional token count
            metadata: Optional additional metadata
        """
        store = conversation_store()
        return await store.add_message(session_id, role, content, model, tokens, metadata)

    async def get_conversation_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get conversation messages (newest first)."""
        store = conversation_store()
        return await store.get_messages(session_id, limit, offset)

    async def get_conversation_messages_asc(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get conversation messages (oldest first - for context)."""
        store = conversation_store()
        return await store.get_messages_asc(session_id, limit)

    async def search_conversation(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search conversation messages."""
        store = conversation_store()
        return await store.search_messages(session_id, query, limit)

    async def get_conversation_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        store = conversation_store()
        return await store.get_stats()

    # --- Knowledge Base ---

    async def kb_add_document(
        self,
        title: str,
        content: str,
        source_type: str,
        source_url: Optional[str] = None,
        source_path: Optional[str] = None,
        format: str = "markdown",
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Add a document to knowledge base."""
        kb = knowledge_base()
        return await kb.add_document(
            title, content, source_type, source_url, source_path, format, session_id, metadata
        )

    async def kb_add_document_from_file(
        self,
        file_path: str,
        source_type: str = "file",
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a document by parsing a file."""
        kb = knowledge_base()
        return await kb.add_document_from_file(file_path, source_type, session_id)

    async def kb_add_document_from_url(
        self,
        url: str,
        content: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a document from URL."""
        kb = knowledge_base()
        return await kb.add_document_from_url(url, content, session_id)

    async def kb_get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Get a document by ID."""
        kb = knowledge_base()
        return await kb.get_document(doc_id)

    async def kb_list_documents(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List documents."""
        kb = knowledge_base()
        return await kb.list_documents(session_id, limit, offset)

    async def kb_search_documents(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents."""
        kb = knowledge_base()
        return await kb.search_documents(query, session_id, limit)

    async def kb_delete_document(self, doc_id: str) -> dict[str, Any]:
        """Delete a document."""
        kb = knowledge_base()
        return await kb.delete_document(doc_id)

    async def kb_get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        kb = knowledge_base()
        return await kb.get_stats()

    # --- Knowledge Graph ---

    async def kg_add_triple(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        confidence: float = 1.0,
        source_type: str = "text",
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a semantic triple to knowledge graph."""
        kg = knowledge_graph()
        return await kg.add_triple(
            subject, predicate, object_name, confidence, source_type, source_id, session_id
        )

    async def kg_get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query triples from knowledge graph."""
        kg = knowledge_graph()
        return await kg.get_triples(subject, predicate, object_name, session_id, limit)

    async def kg_get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        depth: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighboring entities in the graph."""
        kg = knowledge_graph()
        return await kg.get_neighbors(entity, direction, depth, limit)

    async def kg_find_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = 3,
    ) -> Optional[dict[str, Any]]:
        """Find a path between two entities."""
        kg = knowledge_graph()
        path = await kg.find_path(from_entity, to_entity, max_depth)
        if path:
            return {
                "nodes": path.nodes,
                "edges": path.edges,
                "length": path.length,
                "confidence": path.confidence,
            }
        return None

    async def kg_search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search entities in knowledge graph."""
        kg = knowledge_graph()
        return await kg.search_entities(query, entity_type, limit)

    async def kg_get_entity_facts(
        self,
        entity: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get all facts about an entity."""
        kg = knowledge_graph()
        return await kg.get_entity_facts(entity, limit)

    async def kg_get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        kg = knowledge_graph()
        return await kg.get_stats()

    async def kg_upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        action: str = "assert",
        confidence: float = 1.0,
        source_type: str = "text",
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        observed_at: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upsert temporal fact state in knowledge graph (assert/retract)."""
        kg = knowledge_graph()
        return await kg.upsert_fact(
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            action=action,
            confidence=confidence,
            source_type=source_type,
            source_id=source_id,
            session_id=session_id,
            metadata=metadata,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_to=valid_to,
        )

    async def kg_get_triples_as_of(
        self,
        as_of: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get triples valid at a given timestamp."""
        kg = knowledge_graph()
        return await kg.get_triples_as_of(
            as_of=as_of,
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            session_id=session_id,
            limit=limit,
        )

    async def kg_get_fact_history(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get chronological event history for KG facts."""
        kg = knowledge_graph()
        return await kg.get_fact_history(
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            session_id=session_id,
            limit=limit,
        )

    async def kg_find_path_as_of(
        self,
        from_entity: str,
        to_entity: str,
        as_of: Optional[str] = None,
        max_depth: int = 3,
    ) -> Optional[dict[str, Any]]:
        """Find path in graph as-of a timestamp."""
        kg = knowledge_graph()
        path = await kg.find_path_as_of(
            from_entity=from_entity,
            to_entity=to_entity,
            as_of=as_of,
            max_depth=max_depth,
        )
        if path:
            return {
                "nodes": path.nodes,
                "edges": path.edges,
                "length": path.length,
                "confidence": path.confidence,
            }
        return None

    async def kg_get_entity_timeline_summary(
        self,
        entity: str,
        predicate: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get aggregated temporal timeline summary for an entity."""
        kg = knowledge_graph()
        return await kg.get_entity_timeline_summary(
            entity=entity,
            predicate=predicate,
            session_id=session_id,
            limit=limit,
        )

    async def kg_clear(self, session_id: Optional[str] = None) -> dict[str, Any]:
        """Clear knowledge graph data."""
        kg = knowledge_graph()
        return await kg.clear(session_id)

    # --- Memory Extraction ---

    async def extract_memories(
        self,
        text: str,
        entity_id: Optional[str] = None,
        session_id: Optional[str] = None,
        extract_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Extract memories and triples from text automatically."""
        extractor = memory_extractor()
        result = await extractor.extract(text, entity_id, session_id, extract_types)
        return {
            "ok": True,
            "memories_count": result.stats["memories_stored"],
            "triples_count": result.stats["triples_extracted"],
            "types": list(set(m.memory_type for m in result.memories)),
        }

    async def get_extracted_memories(
        self,
        entity_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get extracted memories."""
        extractor = memory_extractor()
        return await extractor.get_memories(entity_id, memory_type, session_id, limit)

    async def search_extracted_memories(
        self,
        query: str,
        entity_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search extracted memories."""
        extractor = memory_extractor()
        return await extractor.search_memories(query, entity_id, limit)

    async def get_extraction_stats(self) -> dict[str, Any]:
        """Get extraction statistics."""
        extractor = memory_extractor()
        return await extractor.get_stats()

    # --- Memory Feedback & Correction ---

    async def memory_correct(
        self,
        key: str,
        value: Any,
        memory_type: str = "preference",
        scope: str = "global",
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Direct correction of a memory entry by key.

        Args:
            key: The memory key to correct
            new_value: The new value to set
            memory_type: 'preference' or 'lesson'
            scope: 'global' or 'session' (for preferences)
            session_id: Required for session-scoped preferences
        """
        k = (key or "").strip()
        if not k:
            return {"ok": False, "error": "missing key"}

        memory_type = (memory_type or "").lower()
        if memory_type not in ("preference", "lesson"):
            return {"ok": False, "error": "invalid memory_type, must be 'preference' or 'lesson'"}

        if memory_type == "preference":
            scope = (scope or "").strip().lower() or "global"
            if scope not in ("global", "session"):
                return {"ok": False, "error": "invalid scope, must be 'global' or 'session'"}

            result = await memory_sql.set_preference(
                scope=scope,
                session_id=session_id if scope == "session" else None,
                key=k,
                value=value,
                source="manual",
                is_locked=False,
                updated_by="memory_correct",
            )
            return {"ok": True, "type": "preference", "key": k, "result": result}

        else:  # lesson
            existing = await memory_sql.get_lesson(k)
            if existing:
                await memory_sql.delete_lesson(k)

            meta = {"corrected": True, "corrected_at": datetime.now(timezone.utc).isoformat()}
            await memory_sql.add_lesson(k, str(value), meta=meta)
            return {"ok": True, "type": "lesson", "key": k, "action": "replaced"}

    async def memory_feedback(
        self,
        feedback: str,
        session_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """Process natural language feedback to correct/update memory.

        Uses LLM to parse feedback and determine what needs to be changed.
        Falls back to rule-based parsing if LLM unavailable.

        Args:
            feedback: Natural language feedback (e.g., "I don't like coffee, I prefer tea")
            session_id: Optional session for session-scoped preferences
            use_llm: Whether to use LLM for parsing (default True)
        """
        fb = (feedback or "").strip()
        if not fb:
            return {"ok": False, "error": "missing feedback text"}

        if use_llm and llm_settings.auto_consolidate:
            try:
                result = await self._llm_feedback(fb, session_id)
                if result.get("ok"):
                    return result
            except Exception:
                pass

        return await self._rule_based_feedback(fb, session_id)

    async def _llm_feedback(self, feedback: str, session_id: Optional[str]) -> dict[str, Any]:
        """LLM-driven feedback parsing and application."""
        system_prompt = """You are a memory feedback parser. Parse the user's feedback and determine what memory action to take.

Respond with JSON only (no other text):
{
  "action": "correct | add | delete",
  "memory_type": "preference | lesson",
  "key": "the memory key",
  "value": "the new value (for correct/add)",
  "scope": "global | session",
  "reason": "why this change should be made"
}

Rules:
- If feedback says "don't like X" or "not Y" or "wrong" -> action: "correct"
- If feedback says "I am X" or "I work at Y" -> action: "add"
- If feedback says "forget" or "delete" -> action: "delete"
- Preferences are typically personal facts (likes, dislikes, facts about user)
- Lessons are general knowledge or instructions
- Default scope is "global" unless user mentions "in this session"
"""

        try:
            response = await llm_client.complete(
                prompt=feedback,
                system=system_prompt,
                max_tokens=300,
                temperature=0.3,
            )

            import json

            parsed = json.loads(response)

            action = parsed.get("action", "")
            memory_type = parsed.get("memory_type", "preference")
            key = parsed.get("key", "")
            value = parsed.get("value", "")
            scope = parsed.get("scope", "global")

            if not key:
                return {"ok": False, "error": "LLM could not determine key"}

            if action == "delete":
                if memory_type == "preference":
                    await memory_sql.delete_preference(
                        scope=scope, session_id=session_id if scope == "session" else None, key=key
                    )
                else:
                    await memory_sql.delete_lesson(key)
                return {"ok": True, "action": "deleted", "key": key, "memory_type": memory_type}

            return await self.memory_correct(
                key=key,
                value=value,
                memory_type=memory_type,
                scope=scope,
                session_id=session_id,
            )

        except Exception as e:
            return {"ok": False, "error": f"LLM feedback failed: {e}"}

    async def _rule_based_feedback(
        self, feedback: str, session_id: Optional[str]
    ) -> dict[str, Any]:
        """Simple rule-based feedback parsing (fallback when LLM unavailable)."""
        fb_lower = feedback.lower()

        corrections = [
            # EN
            ("not ", "not "),
            ("cannot", "cannot"),
            ("won't", "won't"),
            ("don't want", "don't want"),
            ("don't like", "don't like"),
            ("this is not", "this is not"),
            ("changed my mind", "changed my mind"),
            # RU
            ("не ", "not "),
            ("нельзя", "cannot"),
            ("не буду", "won't"),
            ("не хочу", "don't want"),
            ("не люблю", "don't like"),
            ("это не", "this is not"),
            ("передумал", "changed my mind"),
            ("передумала", "changed my mind"),
            # UK
            ("не ", "not "),
            ("не можна", "cannot"),
            ("не буду", "won't"),
            ("не хочу", "don't want"),
            ("не люблю", "don't like"),
            ("це не", "this is not"),
            ("передумав", "changed my mind"),
            ("передумала", "changed my mind"),
        ]

        for source_phrase, normalized_phrase in corrections:
            if source_phrase in fb_lower:
                parts = fb_lower.split(source_phrase)
                if len(parts) > 1:
                    key_part = parts[0].strip()
                    value_part = normalized_phrase.join(parts[1:]).strip()

                    key = self._extract_key_from_text(key_part)
                    if key:
                        return await self.memory_correct(
                            key=key,
                            value=self._negate_value(value_part),
                            memory_type="preference",
                            scope="global",
                        )

        add_patterns = [
            # EN
            "i ",
            "my ",
            "i work",
            "i live",
            # RU
            "я ",
            "мой ",
            "моя ",
            "моё ",
            "я работаю",
            "я живу",
            # UK
            "я ",
            "мій ",
            "моя ",
            "моє ",
            "я працюю",
            "я живу",
        ]
        for pattern in add_patterns:
            if pattern in fb_lower:
                idx = fb_lower.find(pattern)
                after = feedback[idx + len(pattern) :].strip() if idx >= 0 else ""
                if after:
                    key = self._extract_key_from_text(after)
                    if key:
                        value = after.split(",")[0].strip() if "," in after else after.strip()
                        return await self.memory_correct(
                            key=key,
                            value=value,
                            memory_type="preference",
                            scope="global",
                        )

        return {"ok": False, "error": "Could not parse feedback. Try: memory_correct(key, value)"}

    def _extract_key_from_text(self, text: str) -> Optional[str]:
        """Extract a preference key from natural text."""
        text = text.strip().lower()
        text = text.replace("?", "").replace("!", "").replace(".", "")

        replacements = {
            # EN
            "like": "like",
            "love": "like",
            "hate": "hate",
            "prefer": "prefer",
            "work at": "work_at",
            "live in": "live_in",
            # RU
            "люблю": "like",
            "нравится": "like",
            "ненавижу": "hate",
            "предпочитаю": "prefer",
            "работаю в": "work_at",
            "живу в": "live_in",
            # UK
            "подобається": "like",
            "ненавиджу": "hate",
            "віддаю перевагу": "prefer",
            "працюю в": "work_at",
        }

        for source_phrase, mapped in replacements.items():
            text = text.replace(source_phrase, mapped)

        words = text.split()
        if words:
            key = "_".join(words[:3])
            return key[:50]

        return None

    def _negate_value(self, value: str) -> Any:
        """Negate a value based on context."""
        value = value.strip().lower()

        positive = [
            "coffee",
            "tea",
            "yes",
            "true",
            "кофе",
            "чай",
            "да",
            "правда",
            "кава",
            "так",
        ]
        negative = ["no", "false", "нет", "ложь", "ні"]

        for p in positive:
            if p in value:
                return False

        for n in negative:
            if n in value:
                return True

        return value if value else None


memory = MemoryStore()
