from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import httpx

from .config import settings


class EmbeddingsError(Exception):
    pass


@dataclass
class EmbeddingsResult:
    vector: List[float]
    dim: int


# FastEmbed lazy loading
_fastembed_model = None


def _get_fastembed_model():
    """Lazy load FastEmbed model."""
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
            # Default to multilingual model (supports 50+ languages including Russian)
            model_name = getattr(
                settings,
                "embeddings_fastembed_model",
                getattr(
                    settings,
                    "vector_memory_model",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ),
            )
            _fastembed_model = TextEmbedding(model_name=model_name)
        except ImportError:
            raise EmbeddingsError("FastEmbed not installed. Run: pip install fastembed")
        except Exception as e:
            raise EmbeddingsError(f"FastEmbed initialization failed: {e}")
    return _fastembed_model


def _hash_fallback_embed(text: str) -> EmbeddingsResult:
    """Deterministic offline fallback embedding.

    Produces a stable normalized vector without external network/model downloads.
    This is a resilience fallback, not a quality replacement for real embeddings.
    """
    dim = int(
        getattr(
            settings,
            "embeddings_hash_fallback_dimensions",
            getattr(settings, "vector_memory_dimensions", 384),
        )
        or 384
    )
    if dim <= 0:
        dim = 384

    vec = [0.0] * dim
    text_n = (text or "").strip().lower()
    if not text_n:
        return EmbeddingsResult(vector=vec, dim=dim)

    toks = re.findall(r"[^\W_]{2,}", text_n, flags=re.UNICODE)
    if not toks:
        toks = [text_n[i : i + 3] for i in range(max(1, len(text_n) - 2))]

    for i, tok in enumerate(toks[:4096]):
        h = hashlib.blake2b(f"{tok}:{i}".encode("utf-8", errors="ignore"), digest_size=16).digest()
        idx = int.from_bytes(h[:4], "big", signed=False) % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        weight = 0.5 + (h[5] / 255.0)
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return EmbeddingsResult(vector=vec, dim=dim)


async def _fastembed_encode(text: str) -> EmbeddingsResult:
    """Encode text using FastEmbed (local CPU)."""
    timeout_s = float(getattr(settings, "embeddings_fastembed_timeout_s", 20.0) or 20.0)
    loop = asyncio.get_running_loop()

    # FastEmbed can block while trying to download a model; keep it bounded.
    try:
        model = await asyncio.wait_for(loop.run_in_executor(None, _get_fastembed_model), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise EmbeddingsError(f"FastEmbed initialization timed out after {timeout_s:.1f}s") from e

    try:
        vectors = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: list(model.embed([text]))),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as e:
        raise EmbeddingsError(f"FastEmbed embedding timed out after {timeout_s:.1f}s") from e

    if not vectors:
        raise EmbeddingsError("FastEmbed returned no vectors")
    vec = vectors[0]
    return EmbeddingsResult(vector=list(vec), dim=len(vec))


async def _ollama_has_model(client: httpx.AsyncClient, base: str, model: str) -> bool:
    try:
        r = await client.get(f"{base}/api/tags")
        if r.status_code >= 400:
            return True
        data = r.json()
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return True
        for m in models:
            if isinstance(m, dict) and str(m.get("name", "")).strip() == model:
                return True
        return False
    except Exception:
        return True


async def _ollama_pull_model(client: httpx.AsyncClient, base: str, model: str, timeout_s: int) -> None:
    endpoints = [
        (f"{base}/api/pull", {"name": model, "stream": True}),
        (f"{base}/api/pull", {"model": model, "stream": True}),
        (f"{base}/api/pull", {"name": model}),
        (f"{base}/api/pull", {"model": model}),
    ]

    last_err: Optional[str] = None
    for url, payload in endpoints:
        try:
            r = await client.post(url, json=payload, timeout=httpx.Timeout(float(timeout_s), connect=10.0))
            if r.status_code >= 400:
                last_err = f"{r.status_code}: {r.text}"
                continue

            success = False
            async for line in r.aiter_lines():
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    last_err = str(obj.get("error"))
                    break
                status = str(obj.get("status", "")).lower()
                if status == "success":
                    success = True
                    break

            if success:
                return

            try:
                obj = r.json()
                if isinstance(obj, dict) and str(obj.get("status", "")).lower() == "success":
                    return
                if isinstance(obj, dict) and obj.get("error"):
                    last_err = str(obj.get("error"))
                    continue
            except Exception:
                pass
        except Exception as e:
            last_err = str(e)

    raise EmbeddingsError(f"Failed to pull Ollama model '{model}': {last_err or 'unknown error'}")


async def _ensure_ollama_model_available(client: httpx.AsyncClient, base: str, model: str) -> None:
    if await _ollama_has_model(client, base, model):
        return
    if not bool(getattr(settings, "embeddings_pull_missing", True)):
        raise EmbeddingsError(f"Ollama model '{model}' is missing and auto-pull is disabled.")
    await _ollama_pull_model(client, base, model, int(getattr(settings, "embeddings_pull_timeout_s", 600)))


async def _ollama_embed_text(text: str) -> EmbeddingsResult:
    """Encode text using Ollama API."""
    base = (getattr(settings, "embeddings_base_url", "") or getattr(settings, "llm_base_url", "") or "http://localhost:11434").rstrip("/")
    model = (getattr(settings, "embeddings_model", "") or "nomic-embed-text").strip()
    payload = {"model": model, "prompt": text}

    urls = [f"{base}/api/embeddings", f"{base}/api/embed"]

    last_err: Optional[str] = None
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await _ensure_ollama_model_available(client, base, model)

        for url in urls:
            try:
                r = await client.post(url, json=payload)
                if r.status_code >= 400:
                    last_err = f"{r.status_code}: {r.text}"
                    continue
                data = r.json()
                if isinstance(data, dict) and isinstance(data.get("embedding"), list):
                    vec = [float(x) for x in data["embedding"]]
                    return EmbeddingsResult(vector=vec, dim=len(vec))
                emb = data.get("embeddings") if isinstance(data, dict) else None
                if isinstance(emb, list):
                    if len(emb) == 0:
                        last_err = "Empty embeddings array"
                        continue
                    if isinstance(emb[0], list):
                        vec = [float(x) for x in emb[0]]
                        return EmbeddingsResult(vector=vec, dim=len(vec))
                    vec = [float(x) for x in emb]
                    return EmbeddingsResult(vector=vec, dim=len(vec))
                last_err = f"Unexpected embeddings response shape: {data!r}"
            except Exception as e:
                last_err = str(e)

    raise EmbeddingsError(f"Ollama embeddings failed: {last_err or 'unknown error'}")


async def _openai_embed_text(text: str) -> EmbeddingsResult:
    """Encode text using OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EmbeddingsError("OpenAI API key not set (OPENAI_API_KEY)")
    
    base = (getattr(settings, "embeddings_base_url", "") or "https://api.openai.com").rstrip("/")
    model = (getattr(settings, "embeddings_model", "") or "text-embedding-3-small").strip()
    
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(
                f"{base}/v1/embeddings",
                json={"input": text, "model": model},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if r.status_code >= 400:
                raise EmbeddingsError(f"OpenAI API error: {r.status_code}: {r.text}")
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("data"), list) and len(data["data"]) > 0:
                vec = data["data"][0].get("embedding", [])
                return EmbeddingsResult(vector=vec, dim=len(vec))
            raise EmbeddingsError(f"Unexpected OpenAI response: {data}")
        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"OpenAI embeddings failed: {e}")


async def _cohere_embed_text(text: str) -> EmbeddingsResult:
    """Encode text using Cohere API."""
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        raise EmbeddingsError("Cohere API key not set (COHERE_API_KEY)")
    
    model = (getattr(settings, "embeddings_model", "") or "embed-multilingual-v3.0").strip()
    
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(
                "https://api.cohere.ai/v1/embed",
                json={"texts": [text], "model": model, "input_type": "search_document"},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if r.status_code >= 400:
                raise EmbeddingsError(f"Cohere API error: {r.status_code}: {r.text}")
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("embeddings"), list) and len(data["embeddings"]) > 0:
                vec = data["embeddings"][0]
                return EmbeddingsResult(vector=vec, dim=len(vec))
            raise EmbeddingsError(f"Unexpected Cohere response: {data}")
        except EmbeddingsError:
            raise
        except Exception as e:
            raise EmbeddingsError(f"Cohere embeddings failed: {e}")


async def embed_text(text: str) -> EmbeddingsResult:
    """Return an embedding vector for text.

    Supported providers:
      - fastembed: Local CPU embeddings using FastEmbed (no external API) [DEFAULT]
      - ollama: Uses Ollama HTTP API
      - openai: Uses OpenAI API
      - cohere: Uses Cohere API
      - none: Disabled
    """
    provider = (getattr(settings, "embeddings_provider", "fastembed") or "fastembed").strip().lower()
    
    if provider in ("none", "", "disabled", "off"):
        raise EmbeddingsError("Embeddings provider is disabled (MEMORY_EMBEDDINGS_PROVIDER=none).")

    # Try FastEmbed first (local, no API key needed)
    if provider == "fastembed":
        try:
            return await _fastembed_encode(text)
        except EmbeddingsError:
            if bool(getattr(settings, "embeddings_fastembed_allow_hash_fallback", True)):
                return _hash_fallback_embed(text)
            raise
        except Exception as e:
            if bool(getattr(settings, "embeddings_fastembed_allow_hash_fallback", True)):
                return _hash_fallback_embed(text)
            raise EmbeddingsError(f"FastEmbed failed: {e}")

    if provider == "ollama":
        return await _ollama_embed_text(text)
    
    if provider == "openai":
        return await _openai_embed_text(text)
    
    if provider == "cohere":
        return await _cohere_embed_text(text)

    raise EmbeddingsError(f"Unsupported embeddings provider: {provider}")
