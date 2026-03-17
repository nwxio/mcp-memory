from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_file() -> str | None:
    # Allow explicit path override
    env_file = os.getenv("OMNIMIND_ENV_FILE", ".env")
    env_file = (env_file or "").strip()
    if not env_file:
        return None
    return env_file if os.path.exists(env_file) else None


class Settings(BaseSettings):
    # Read .env automatically if present, otherwise rely on environment variables.
    model_config = SettingsConfigDict(
        env_prefix="OMNIMIND_",
        extra="ignore",
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
    )

    # Core
    api_key: str = "devkey"
    # Legacy DB selector (sqlite | postgres). New deployments should prefer
    # OMNIMIND_POSTGRES_ENABLED / OMNIMIND_SQLITE_ENABLED flags.
    db_type: str = "sqlite"
    postgres_enabled: bool | None = None
    sqlite_enabled: bool | None = None
    # If true, backend mismatch is fatal. Example: requested=postgres but
    # effective=sqlite will raise at startup instead of silent fallback.
    db_strict_backend: bool = False
    db_path: str = "/data/omnimind.db"

    # PostgreSQL (optional backend)
    postgres_host: str = "localhost"
    postgres_port: int = 5442
    postgres_db: str = "memory"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Database pragmas
    # busy_timeout reduces SQLITE_BUSY errors under concurrent access
    db_busy_timeout_ms: int = 5000
    # If enabled, OmniMind will checkpoint WAL during housekeeping to keep
    # -wal files from growing without bound on long-running deployments.
    db_wal_checkpoint_on_housekeep: bool = True
    # WAL checkpoint mode: PASSIVE | FULL | RESTART | TRUNCATE
    db_wal_checkpoint_mode: str = "TRUNCATE"
    workspace: str = "/workspace"

    # Attachments (per-chat)
    # Stored outside workspace to avoid accidental indexing/leakage.
    attachments_dir: str = "/data/omnimind_attachments"
    # Hard caps (bytes/files) to prevent abuse.
    attachments_max_bytes: int = 20_000_000
    attachments_max_per_message: int = 10
    # LLM ingestion caps for text/* attachments (prompt/context injection).
    # These are char-based limits after decoding (utf-8 with replacement).
    attachments_text_max_chars_total: int = 60_000
    attachments_text_max_chars_per_file: int = 20_000

    # Multimodal LLM: optional inlining of image/* attachments (data-URI).
    # Disabled by default to avoid regressions / payload bloat.
    attachments_images_to_llm_enabled: bool = False
    attachments_images_max_files: int = 4
    attachments_images_max_bytes_total: int = 1_500_000
    attachments_images_max_bytes_per_file: int = 800_000
    attachments_images_allowed_mime: str = "image/png,image/jpeg,image/webp"
    # Token TTL for browser-friendly preview URLs.
    attachments_token_ttl_s: int = 900
    plugin_dir: str = "/app/omnimind/plugins_ext"
    sandbox_image: str = "omnimind-sandbox:latest"

    # CORS
    # Comma/semicolon-separated origins. Use "*" for any origin.
    # NOTE: If "*" is present, credentials will be forced off in main.py
    # because browsers disallow allow-credentials with wildcard origins.
    cors_allow_origins: str = "*"
    # Whether to allow cookies/authorization in cross-origin requests.
    # If cors_allow_origins contains "*", this will be ignored (forced False).
    cors_allow_credentials: bool = False
    # Comma/semicolon-separated methods/headers. Use "*" to allow all.
    cors_allow_methods: str = "*"
    cors_allow_headers: str = "*"

    # Workspace indexing safety
    # If enabled, OmniMind will redact common secret patterns (API keys, bearer
    # tokens, passwords, private keys) before storing file contents in the
    # workspace indexes (FTS and vectors). This reduces accidental secret
    # leakage via retrieval/UI previews.
    workspace_redact_secrets: bool = True

    # Audit safety
    # If enabled, OmniMind will redact common secret patterns before persisting
    # audit payloads (prompts, plans, tool IO snippets) into the DB.
    # This reduces accidental secret leakage via the UI and exported logs.
    audit_redact_secrets: bool = True

    # Patch preview safety
    # If enabled, OmniMind will redact common secret patterns in patch previews
    # (diffs shown in UI/approvals). Full patch contents remain intact so apply/
    # rollback stay correct.
    patch_preview_redact_secrets: bool = True

    # Memory safety
    # If enabled, OmniMind will redact common secret patterns before persisting
    # Working Memory (WM) and Lessons. This reduces accidental secret leakage
    # via the UI and retrieval context.
    wm_redact_secrets: bool = True
    lessons_redact_secrets: bool = True

    # Durable memory consolidation
    # Optional: after each task, deterministically promote useful signals from recent
    # episodes into Lessons/Preferences. Enabled to auto-save lessons/preferences after tasks.
    auto_consolidate_on_task_complete: bool = True
    # Minimum seconds between auto-consolidations for the same session.
    auto_consolidate_min_interval_s: int = 1800
    auto_consolidate_episode_limit: int = 50
    auto_consolidate_max_lessons: int = 10
    auto_consolidate_include_preferences: bool = True
    # 'session' is safer by default to avoid polluting global prefs.
    auto_consolidate_preferences_scope: str = "session"

    # Session snapshot (STM) can contain tool params/results. Redact secrets before
    # persisting to DB to avoid accidental leakage via the UI.
    snapshot_redact_secrets: bool = True

    # Consolidation safety
    # If enabled, OmniMind will redact common secret patterns in consolidation
    # proposals (episodes -> lessons/preferences) before persisting or showing them.
    consolidate_redact_secrets: bool = True

    # Memory export/import safety
    # If enabled, OmniMind will redact common secret patterns in exported memory snapshots.
    memory_export_redact_secrets: bool = True
    # If enabled, OmniMind will redact common secret patterns in imported memory snapshots before writing.
    memory_import_redact_secrets: bool = True

    # Memory import hard caps (safety against huge payloads)
    # These are independent from export limits because imports can come from outside.
    memory_import_max_preferences: int = 10000
    memory_import_max_lessons: int = 10000
    memory_import_max_episodes: int = 5000

    # Memory TTL (days) - auto-expiry for lessons and episodes
    # Set to 0 to disable auto-expiry
    memory_lessons_ttl_days: int = 90
    memory_episodes_ttl_days: int = 60
    memory_preferences_ttl_days: int = 180

    # Conversation settings
    conversation_redact_secrets: bool = True
    conversation_max_messages: int = 1000

    # Cross-session memory settings
    cross_session_enabled: bool = True
    cross_session_max_context_tokens: int = 2000
    cross_session_redact_secrets: bool = True

    # Memory consolidation (decay/merge/prune)
    memory_decay_enabled: bool = True
    memory_decay_factor: float = 0.9
    memory_decay_period_days: int = 30
    memory_decay_min_importance: float = 0.1

    memory_merge_enabled: bool = True
    memory_merge_similarity_threshold: float = 0.85

    memory_prune_enabled: bool = True
    memory_prune_max_age_days: int = 180
    memory_prune_min_score: float = 0.05

    # Vector memory settings
    vector_memory_enabled: bool = True
    vector_memory_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual: RU, EN, DE, FR + 50+

    # Neo4j graph database backend (optional)
    # Set neo4j_enabled=true and provide credentials to use Neo4j as knowledge graph backend
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    # Comma-separated predicates that should keep only one active object
    # per (subject, predicate) in temporal KG mode.
    kg_temporal_single_active_predicates: str = "works_for,belongs_to,prefers"
    vector_memory_dimensions: int = 384
    embeddings_provider: str = "fastembed"  # fastembed, openai, cohere

    # Preferences sometimes intentionally store operational secrets (e.g. API
    # keys for external services). For that reason, preference redaction is
    # opt-in.
    preferences_redact_secrets: bool = False

    # Optional PATH passed into sandbox containers (terminal tool).
    # Can be useful for deterministic command resolution. Example:
    # OMNIMIND_SANDBOX_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    sandbox_path: str = ""

    # Terminal execution
    # terminal_executor: docker | local | auto
    # - docker: always run commands inside the Docker sandbox (recommended)
    # - local: run commands directly on the backend host/container (no Docker)
    # - auto: try docker, fall back to local if docker is unavailable
    terminal_executor: str = "auto"
    terminal_local_fallback: bool = True

    # Security / sandbox
    allow_paths: str = "/workspace;/data"
    deny_paths: str = "/;/etc;/proc;/sys;/dev;/root"
    enable_terminal: bool = True
    enable_ssh: bool = True
    enable_docker: bool = True

    # SSH defaults (used by ssh tool)
    ssh_default_host: str = ""
    ssh_default_user: str = ""
    ssh_default_port: int = 22
    ssh_identity_file: str = ""  # e.g. /data/id_ed25519
    ssh_known_hosts_file: str = ""  # e.g. /data/known_hosts
    ssh_strict_host_key_checking: bool = True
    ssh_always_require_approval: bool = True

    # Web search (SearxNG)
    searxng_base_url: str = "https://s.netwize.work"
    searxng_timeout_s: float = 20.0
    # Cap the number of results pulled from SearxNG (the tool may request fewer).
    # This prevents accidental huge payloads on misconfigured instances.
    searxng_max_results: int = 25

    # App time
    app_timezone: str = "Europe/Kyiv"

    # Agent tone / prompts
    # agent_persona: strict | friendly
    # - strict: more "tool-like" structured outputs
    # - friendly: more human, conversational outputs (still safe)
    agent_persona: str = "friendly"

    # Default chat system prompt (used by chat tool when caller did not provide one).
    # Can be overridden via OMNIMIND_CHAT_DEFAULT_SYSTEM in .env.
    chat_default_system: str = (
        "You are a friendly, conversational AI assistant. Answer naturally and clearly, "
        "you may use light humor but avoid fluff. If the user asks for code, provide precise, "
        "working code. If there are risks or uncertainty, explain them in simple terms."
    )

    # Optional higher-priority chat system prompt. If set, it overrides chat_default_system.
    # Env: OMNIMIND_CHAT_SYSTEM_PROMPT
    chat_system_prompt: str = ""

    # Optional extra system prompts for different LLM phases.
    # These are appended to the built-in system prompts for safety/format contracts.
    # Env: OMNIMIND_THINK_SYSTEM_PROMPT / OMNIMIND_PLAN_SYSTEM_PROMPT / OMNIMIND_VERIFY_SYSTEM_PROMPT
    think_system_prompt: str = ""
    plan_system_prompt: str = ""
    verify_system_prompt: str = ""

    # Auto-solve loop (agent retries before asking the user)
    # The agent will try to re-plan / re-run steps up to auto_solve_max_cycles.
    # It will only ask the user for guidance after auto_solve_ask_after failed cycles.
    # Env: OMNIMIND_AUTO_SOLVE_ENABLED / OMNIMIND_AUTO_SOLVE_MAX_CYCLES / OMNIMIND_AUTO_SOLVE_ASK_AFTER
    auto_solve_enabled: bool = True
    auto_solve_max_cycles: int = 7
    auto_solve_ask_after: int = 7

    # Web cache
    web_cache_ttl_hours: int = 72
    web_cache_max_rows: int = 2000

    # Background maintenance loop
    # The backend runs a best-effort maintenance loop (never fatal) that keeps
    # caches and indexes reasonably fresh on long-running deployments.
    maintenance_enabled: bool = True
    # Web cache cleanup interval (seconds)
    maintenance_web_cache_interval_s: int = 3600
    # Workspace FTS incremental refresh interval (seconds)
    maintenance_workspace_fts_enabled: bool = True
    maintenance_workspace_fts_interval_s: int = 180
    # Vector index refresh interval (seconds). Disabled by default because
    # embeddings might be unavailable in some deployments.
    maintenance_vectors_enabled: bool = False
    maintenance_vectors_interval_s: int = 900
    # Housekeeping interval (episodes/audit/monitor results). This is also
    # internally throttled by housekeeping_min_interval_s.
    maintenance_housekeep_enabled: bool = True
    maintenance_housekeep_interval_s: int = 900

    # Working Memory (WM)
    # If enabled, OmniMind will append a compact, non-sensitive summary of each
    # completed task into the session WM. This improves continuity even when LLM
    # planning occasionally falls back.
    working_memory_auto_update: bool = True
    # Global max size of WM for a session (chars). The append operation will clamp.
    working_memory_max_chars: int = 12000
    # Max chars of the auto-appended delta for a single task.
    working_memory_auto_update_max_delta_chars: int = 2500

    # Episodic memory retention
    # Keep episodic logs bounded to prevent DB bloat on long-running deployments.
    # Set to 0 to disable time-based pruning.
    episode_retention_days: int = 180
    # Keep at most N newest episodes per session. Set to 0 to disable count-based pruning.
    episode_max_per_session: int = 500

    # Audit log retention
    # Audit logs are useful, but can grow quickly with streaming UIs and monitor ticks.
    # Set to 0 to disable time-based pruning.
    audit_retention_days: int = 60
    # Hard cap for audit_events rows (best-effort). Set to 0 to disable.
    audit_max_rows: int = 50_000

    # Monitor results retention
    # MonitorRunner appends a row for each rule evaluation (history),
    # so we keep it bounded.
    monitor_results_retention_days: int = 14
    monitor_results_max_rows: int = 50_000
    monitor_results_max_rows_per_session: int = 10_000

    # Autonomy actions retention
    # autonomy_actions grows with tool calls in autonomous runs (and even more in A3/A4),
    # so we keep it bounded via housekeeping.
    autonomy_actions_retention_days: int = 14
    autonomy_actions_max_rows: int = 50_000
    autonomy_actions_max_rows_per_session: int = 10_000

    # Background housekeeping throttle (seconds)
    # Avoid doing prune work on every completed task.
    housekeeping_min_interval_s: int = 30

    # Web research source filtering
    # Comma-separated list of blocked TLDs.
    # Default blocks Russian TLDs (including punycode form).
    # Accepts values with or without a leading dot (e.g. "ru" or ".ru").
    web_blocked_tlds: str = ".ru,.рф,xn--p1ai"

    # Autonomy level (A0..A4)
    # A0: observe-only (no clicks/types)
    # A1: safe navigation (low-risk clicks only)
    # A2: assist (medium actions require approvals; high always)
    # A3: power-user (medium actions may be auto-executed on allowlist)
    # A4: YOLO (almost all actions auto; still blocks "red" actions like payment/delete/password)
    autonomy_level: str = "A2"

    # Browser allowlist/safety keywords (comma-separated)
    # Domains can be given as example.com (suffix match) or full host.
    browser_domain_allowlist: str = ""

    # Additional sensitive keywords for risk classification (comma-separated).
    # Merge-only with preferences.
    browser_sensitive_keywords: str = (
        "delete,remove,confirm,save,submit,pay,checkout,buy,order,sign in,log in,login,logout,sign out,transfer,send,withdraw,"
        "delete account,confirm payment,purchase,order now,sign in,log out,transfer funds,send money,"
        "удал,подтверд,оплат,куп,заказ,войти,вход,выйти,перевод,отправ,"
        "видал,підтверд,оплат,куп,замов,увійти,вхід,вийти,переказ,надісл"
    )

    # Hard approval keywords (cannot be auto-executed even in A4) (comma-separated).
    browser_hard_approval_keywords: str = "pay,checkout,buy,order,purchase,subscribe,delete,remove,erase,format,withdraw,transfer,send,money,card,cvv,cvc,password,2fa,otp"

    # Browser automation tool (Playwright) — optional.
    # Even when enabled, risky interactions are gated via approvals.
    browser_enabled: bool = False
    browser_headless_default: bool = True
    browser_session_ttl_minutes: int = 30
    browser_max_sessions: int = 3
    # Where screenshots/traces/downloads are stored.
    # In docker-compose, /data is a persisted volume.
    browser_artifacts_dir: str = "/data/omnimind_browser"
    # Max size (bytes) allowed for serving or persisting browser artifacts (downloads/traces).
    # Prevents accidental huge files from filling disk.
    browser_artifact_max_bytes: int = 50 * 1024 * 1024
    # Additional blocked TLDs for BrowserTool (comma-separated). Merge-only with preferences.
    browser_blocked_tlds: str = ""

    # Optional: path to system Chromium/Chrome executable (used when Playwright browsers are not installed).
    # Examples: /usr/bin/chromium, /usr/bin/google-chrome
    browser_chromium_executable_path: str = ""

    # Live browser (B3) — optional: a separate container runs a headful Chromium with VNC/noVNC,
    # and exposes Chrome DevTools Protocol (CDP). BrowserTool can attach to it for a real-time view.
    browser_live_enabled: bool = False

    # CDP endpoint for the live browser container (Playwright connect_over_cdp expects an HTTP URL).
    browser_live_cdp_endpoint: str = "http://browser_live:9222"

    # noVNC URL that humans can open (UI will show it when configured).
    browser_live_novnc_url: str = ""

    # Connect timeout when attaching to the live browser (seconds).
    browser_live_connect_timeout_s: float = 8.0

    # Browser artifacts housekeeping (filesystem)
    # Downloads/traces/screenshots are stored on disk and can grow without bound.
    # These settings keep them bounded in long-running deployments.
    browser_artifacts_housekeep_enabled: bool = True
    # Minimum seconds between filesystem scans (separate from DB housekeeping throttles).
    browser_artifacts_housekeep_min_interval_s: int = 1800
    # Retention in days for artifacts (0 disables time-based pruning).
    browser_artifacts_retention_days: int = 7
    # Hard cap for total bytes kept per scan (0 disables size cap). Oldest files are deleted first.
    browser_artifacts_max_total_bytes: int = 2 * 1024 * 1024 * 1024
    # Safety caps to keep scans bounded.
    browser_artifacts_housekeep_max_files: int = 20000
    browser_artifacts_housekeep_max_sessions: int = 250

    # Vector search performance
    vector_search_candidate_limit: int = 2500

    # LLM configuration (optional)
    # llm_provider: none | ollama | openai_compatible | openai | deepseek | gemini
    llm_provider: str = "none"
    # Comma-separated list of fallback providers to try if the primary provider fails.
    # Example: "ollama,deepseek,openai,gemini" (order matters).
    llm_provider_fallbacks: str = ""
    # Circuit-breaker cooldown (seconds). If a provider fails, we temporarily
    # skip it to avoid hammering a dead endpoint.
    llm_provider_cooldown_s: float = 30.0

    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    # Provider-specific API keys (optional). If set, they override llm_api_key
    # for the corresponding provider.
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "llama3.1:8b"
    # Comma-separated list of model names to try if llm_model is missing in Ollama.
    # Example: "gemma2,llama3.1:8b,llama3:8b"
    llm_model_fallbacks: str = "gemma2,llama3.1:8b,llama3:8b"
    # If True and provider=ollama, OmniMind will auto-select an available model
    # from llm_model_fallbacks (or any non-embedding model) when the configured
    # llm_model is not present on the Ollama server.
    llm_auto_select_model: bool = True
    llm_temperature: float = 0.2
    # Default max tokens for general LLM calls (chat/tool-level generation).
    # Raise via OMNIMIND_LLM_MAX_TOKENS if you want longer responses.
    llm_max_tokens: int = 4096

    # Max tokens for the final user-facing report stream.
    # This is intentionally higher than llm_max_tokens to avoid silent truncation.
    llm_report_max_tokens: int = 8192

    llm_think_max_tokens: int = 1600
    llm_plan_max_tokens: int = 2200
    llm_verify_max_tokens: int = 1400
    # LLM HTTP client timeouts (seconds)
    # NOTE: If OmniMind backend runs in Docker and your LLM server runs on the host,
    # 'http://localhost:11434' from inside the container points to the container itself.
    # Use a host-reachable address (e.g. host.docker.internal) or the host IP.
    llm_timeout_s: float = 300.0
    llm_connect_timeout_s: float = 10.0
    llm_tags_timeout_s: float = 10.0

    # Verify gates
    verify_enabled: bool = True
    verify_strict: bool = True
    verify_timeout_s: int = 180
    verify_on_patch_apply: bool = True
    verify_on_commit_staged: bool = True

    # Embeddings / Vector memory (optional)
    # embeddings_provider: fastembed | ollama | openai | none
    embeddings_provider: str = "fastembed"
    # FastEmbed-specific model selector.
    # Kept separate from vector_memory_model for explicit env compatibility:
    # OMNIMIND_EMBEDDINGS_FASTEMBED_MODEL
    embeddings_fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Guardrails for offline/slow environments.
    embeddings_fastembed_timeout_s: float = 20.0
    embeddings_fastembed_allow_hash_fallback: bool = True
    embeddings_hash_fallback_dimensions: int = 384
    embeddings_base_url: str = "http://localhost:11434"
    embeddings_model: str = "nomic-embed-text"
    embeddings_pull_missing: bool = True
    embeddings_pull_timeout_s: int = 600

    # Vector memory indexing
    vector_chunk_chars: int = 1200
    vector_chunk_overlap: int = 150
    vector_max_files: int = 5000
    vector_max_file_bytes: int = 750_000
    vector_allow_ext: str = (
        ".py,.md,.txt,.json,.yml,.yaml,.toml,.js,.ts,.tsx,.css,.html,.sh,.env.example"
    )

    # Auto-build vectors on first use (to keep UX smooth)
    vector_autobuild_on_first_use: bool = True
    vector_autobuild_max_files_startup: int = 200

    # Vector workspace incremental indexing (budgeted, throttled)
    vector_ws_max_files: int = 5000
    vector_ws_max_seconds: float = 8.0
    # Budget for background incremental refresh (used during retrieval)
    vector_ws_autorefresh_max_files: int = 60
    vector_ws_autorefresh_max_seconds: float = 1.0
    vector_ws_min_interval_seconds: float = 300.0

    # Retrieval: prefer hybrid workspace search (FTS + vectors) for better relevance
    retrieval_use_hybrid_workspace: bool = True
    retrieval_hybrid_limit: int = 8
    retrieval_hybrid_fts_pool: int = 24
    retrieval_hybrid_vec_pool: int = 24
    retrieval_hybrid_per_file_cap: int = 2
    retrieval_hybrid_fts_weight: float = 1.0
    retrieval_hybrid_vec_weight: float = 1.0
    # Hybrid filter: drop extremely low-score candidates (keeps noise down).
    # Absolute score threshold (combined normalized score + bonuses). 0.0 disables.
    retrieval_hybrid_min_score: float = 0.18
    # Relative threshold vs top score in the candidate pool. 0.0 disables.
    retrieval_hybrid_min_rel_score: float = 0.06
    # Hybrid rerank: diversity-aware selection (MMR-like)
    retrieval_hybrid_use_mmr: bool = True
    retrieval_hybrid_mmr_lambda: float = 0.70
    retrieval_hybrid_mmr_max_candidates: int = 120

    # Hybrid post-filter: dedupe near-identical snippets.
    # Helps reduce repeated context when FTS + vectors return overlapping chunks.
    retrieval_hybrid_dedupe_enabled: bool = True
    retrieval_hybrid_dedupe_sim_threshold: float = 0.92
    retrieval_hybrid_dedupe_max_compare: int = 60

    # Hybrid: recency bias (favor recently changed workspace files)
    retrieval_hybrid_recency_enabled: bool = True
    retrieval_hybrid_recency_half_life_days: float = 30.0
    retrieval_hybrid_recency_max_bonus: float = 0.12
    retrieval_hybrid_recency_max_age_days: float = 365.0

    # Query rewrite for workspace retrieval (FTS/hybrid)
    retrieval_workspace_query_rewrite: bool = True
    retrieval_workspace_query_rewrite_min_tokens: int = 3
    retrieval_workspace_query_rewrite_max_tokens: int = 12

    # Context packing budgets (characters)
    retrieval_context_max_chars: int = 6000
    retrieval_context_split_budgets: bool = True
    retrieval_context_memory_chars: int = 2200
    retrieval_context_workspace_chars: int = 3800

    # Packing policy: diversify retrieved context to reduce repetition and over-focus on one file.
    retrieval_pack_round_robin: bool = True
    retrieval_pack_per_path_cap: int = 2
    retrieval_pack_per_source_cap: int = 0

    # Retrieval context packing
    # When enabled, workspace hits (FTS/vector) are expanded into larger excerpts
    # with surrounding context (and optional line numbers for code-ish files).
    retrieval_expand_workspace_hits: bool = True
    retrieval_workspace_max_files: int = 4
    retrieval_workspace_excerpt_chars: int = 1200
    retrieval_workspace_excerpt_truncate: int = 1600
    retrieval_workspace_excerpt_max_lines: int = 80
    retrieval_workspace_line_numbers: bool = True
    # Hard clamp for very large docs pulled from memory_docs.
    retrieval_workspace_hard_max_chars: int = 350_000

    # Vector hit expansion (neighbors)
    retrieval_vector_neighbor_radius: int = 1
    retrieval_vector_excerpt_chars: int = 1400

    # Lessons retrieval
    retrieval_lessons_use_search: bool = True
    retrieval_lessons_limit: int = 8

    # Workspace FTS indexing (SQLite)
    # Incremental indexing is used during retrieval to keep the DB in sync
    # without forcing a heavy full scan on every request.
    workspace_fts_max_file_bytes: int = 512_000
    # Default budgets for explicit indexing calls (API/manual)
    workspace_fts_max_files: int = 6000
    workspace_fts_max_seconds: float = 2.0
    # Budget for background incremental refresh
    workspace_fts_autorefresh_max_files: int = 500
    workspace_fts_autorefresh_max_seconds: float = 1.5
    workspace_fts_min_interval_seconds: float = 120.0
    # Optional time budget for explicit full scans (0 disables time limit)
    workspace_fts_fullscan_max_seconds: float = 0.0

    # On-write indexing (keep memory fresh after file commits)
    # When FileTool commits staged changes, index the changed files into FTS
    # (and vectors) so retrieval sees updates immediately.
    workspace_index_on_write: bool = True
    workspace_index_on_write_vectors: bool = True
    # Safety clamps for post-write indexing bursts.
    workspace_index_on_write_max_files: int = 25
    workspace_index_on_write_max_file_bytes: int = 512_000
    workspace_index_on_write_max_seconds: float = 2.0
    workspace_index_on_write_vectors_max_seconds: float = 5.0

    # Graph memory (associative links between files/symbols)
    graph_memory_enabled: bool = True
    # Ingest graph links on write/commit (post-commit hook)
    graph_memory_on_write: bool = True
    graph_memory_on_write_max_files: int = 50
    graph_memory_on_write_max_seconds: float = 3.0
    # Include graph summaries in retrieval context
    graph_memory_retrieval: bool = True
    graph_memory_retrieval_limit: int = 6

    # Web research history (Stage 6)
    # Stores recent web-research answers in SQLite for quick recall in UI.
    research_history_enabled: bool = True
    research_history_default_limit: int = 50
    # Hard clamp to avoid DB bloat when sources/snippets are large.
    research_history_max_payload_chars: int = 200_000
    research_history_preview_chars: int = 280

    # UI streaming (frontend "tmux"/file write previews)
    ui_stream_file_writes: bool = True
    ui_stream_file_chunk_chars: int = 160
    ui_stream_file_max_bytes: int = 4000

    # Retrieval trace shown in UI (WS events)
    ui_retrieval_preview_limit: int = 12
    ui_retrieval_preview_chars: int = 220

    # Monitoring
    monitor_enabled: bool = True
    monitor_tick_s: float = 1.0
    monitor_global_min_interval_s: int = 5  # clamp per-rule intervals to avoid tight loops

    # Monitor auto-actions (disabled by default)
    # When enabled, monitor rules may run pre-defined tool actions via TaskManager
    # without invoking the LLM planner. Tool approvals and autonomy guardrails still apply.
    monitor_auto_enabled: bool = False
    # Comma-separated allowlist of tool.action entries permitted for monitor auto-actions.
    # Example: "terminal.run,file.read_file"
    monitor_auto_allowlist: str = ""
    # Hard cap for auto-action tasks created per day across all sessions (0 disables cap).
    monitor_auto_max_tasks_per_day: int = 500

    # Autonomy
    # autonomy_mode: propose | auto_low_risk
    autonomy_mode: str = "propose"
    # If enabled, some low-risk steps may auto-run without raising approvals (still respects tool policies).
    auto_approve_low_risk: bool = False

    # Autonomy guardrails (anti-loop / anti-spam)
    autonomy_dedupe_enabled: bool = True
    # If the same tool/action+params repeats faster than this, require approval.
    autonomy_min_tool_interval_s: float = 1.5
    # If too many tool calls happen in a short window, require approval.
    autonomy_burst_window_s: float = 10.0
    autonomy_burst_limit: int = 12

    @staticmethod
    def _normalize_provider_name(p: str) -> str:
        p = (p or "").strip().lower()
        if p in ("openai-compatible", "openai_compatible"):
            return "openai_compatible"
        if p in ("chatgpt", "openai"):
            return "openai"
        if p in ("google", "google_genai"):
            return "gemini"
        return p or "none"

    def llm_provider_chain(self) -> list[str]:
        """Return provider chain: primary + fallbacks (normalized, de-duped).

        Env: OMNIMIND_LLM_PROVIDER + OMNIMIND_LLM_PROVIDER_FALLBACKS
        """
        primary = self._normalize_provider_name(self.llm_provider)
        raw = (self.llm_provider_fallbacks or "").strip()
        chain: list[str] = [primary] if primary else ["none"]
        for part in raw.split(","):
            name = self._normalize_provider_name(part)
            if not name:
                continue
            if name not in chain:
                chain.append(name)
        # Always keep 'none' last if present.
        if "none" in chain and chain[-1] != "none":
            chain = [x for x in chain if x != "none"] + ["none"]
        return chain

    def llm_effective_api_key_for(self, provider: str) -> str:
        p = self._normalize_provider_name(provider)
        if p == "openai" and (self.openai_api_key or "").strip():
            return (self.openai_api_key or "").strip()
        if p == "deepseek" and (self.deepseek_api_key or "").strip():
            return (self.deepseek_api_key or "").strip()
        if p == "gemini" and (self.gemini_api_key or "").strip():
            return (self.gemini_api_key or "").strip()
        return (self.llm_api_key or "").strip()

    def llm_effective_model_for(self, provider: str) -> str:
        p = self._normalize_provider_name(provider)
        model = (self.llm_model or "").strip()
        default_models = {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-2.0-flash",
        }
        if p in default_models:
            looks_like_ollama = (":" in model) or model.lower().startswith(
                ("llama", "gemma", "mistral", "qwen", "phi")
            )
            if (not model) or looks_like_ollama:
                return default_models[p]
        return model

    def llm_effective_base_url_for(self, provider: str) -> str:
        p = self._normalize_provider_name(provider)
        base = (self.llm_base_url or "").strip()

        def _looks_like_ollama_base(url: str) -> bool:
            u = (url or "").strip().lower()
            if not u:
                return False
            # Common case: user previously configured Ollama and then switches provider.
            # Ollama default port is 11434; treat any base pointing to that port as ollama-like.
            return ":11434" in u

        local_defaults = {
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://host.docker.internal:11434",
        }

        # For hosted providers, if base is empty OR still pointing to an Ollama-style base,
        # swap to the provider's sensible default.
        if p == "openai" and (not base or base in local_defaults or _looks_like_ollama_base(base)):
            return "https://api.openai.com"
        if p == "deepseek" and (
            not base or base in local_defaults or _looks_like_ollama_base(base)
        ):
            return "https://api.deepseek.com"
        if p == "gemini" and (not base or base in local_defaults or _looks_like_ollama_base(base)):
            return "https://generativelanguage.googleapis.com"

        return base

    def llm_effective_provider(self) -> str:
        """Normalized provider name."""
        p = (self.llm_provider or "").strip().lower()
        if p in ("openai-compatible", "openai_compatible"):
            return "openai_compatible"
        if p in ("chatgpt", "openai"):
            return "openai"
        return p or "none"

    def llm_effective_api_key(self) -> str:
        """Pick the right API key for the selected provider."""
        p = self.llm_effective_provider()
        if p == "openai" and (self.openai_api_key or "").strip():
            return (self.openai_api_key or "").strip()
        if p == "deepseek" and (self.deepseek_api_key or "").strip():
            return (self.deepseek_api_key or "").strip()
        if p == "gemini" and (self.gemini_api_key or "").strip():
            return (self.gemini_api_key or "").strip()
        return (self.llm_api_key or "").strip()

    def llm_effective_model(self) -> str:
        """Provider-aware default model."""
        return self.llm_effective_model_for(self.llm_effective_provider())

    def llm_effective_base_url(self) -> str:
        """Provider-aware base URL defaulting."""
        return self.llm_effective_base_url_for(self.llm_effective_provider())


settings = Settings()


def _guess_project_root() -> Path:
    """Best-effort project root detection.

    Works both in editable repo layout and in packaged/container layouts.
    """
    here = Path(__file__).resolve()
    p = here
    for _ in range(8):
        if (p / "docker-compose.yml").exists() or (p / "README.md").exists():
            return p
        p = p.parent
    # fallback: backend/omnimind/core/config.py -> ../../../../
    try:
        return here.parents[4]
    except Exception:
        return here.parent


def _postprocess_settings() -> None:
    """Harden defaults for non-docker / local runs.

    Common failure mode: OMNIMIND_WORKSPACE is not set and /workspace does not exist.
    In that case, UI features like file browsing + workspace indexing break.

    Docker deployments are unaffected because docker-compose sets OMNIMIND_WORKSPACE.
    """
    # Backward compatibility with older prompt/docs examples that used
    # AUTO_CONSOLIDATE_ON_TASK_COMPLETE without OMNIMIND_ prefix.
    if (
        os.getenv("OMNIMIND_AUTO_CONSOLIDATE_ON_TASK_COMPLETE") is None
        and os.getenv("AUTO_CONSOLIDATE_ON_TASK_COMPLETE") is not None
    ):
        raw = (os.getenv("AUTO_CONSOLIDATE_ON_TASK_COMPLETE") or "").strip().lower()
        settings.auto_consolidate_on_task_complete = raw in ("1", "true", "yes", "on", "y")

    if os.getenv("OMNIMIND_WORKSPACE"):
        return

    ws = (settings.workspace or "").strip() or "/workspace"
    if ws == "/workspace" and not os.path.exists(ws):
        root = _guess_project_root()
        cand = root / "workspace"
        if cand.exists():
            settings.workspace = str(cand.resolve())
        else:
            # Local/dev fallback: use repository root when /workspace is absent.
            settings.workspace = str(root.resolve())

        # keep allow_paths coherent (used when creating new sessions)
        try:
            parts = [p.strip() for p in (settings.allow_paths or "").split(";") if p.strip()]
            parts = [settings.workspace if p == "/workspace" else p for p in parts]
            settings.allow_paths = ";".join(parts)
        except Exception:
            pass


_postprocess_settings()
