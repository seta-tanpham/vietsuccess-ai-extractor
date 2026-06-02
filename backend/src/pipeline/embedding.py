"""
Phase 3 — Embedding engine.

Backends:
  openai  → text-embedding-3-small (1536-dim), API-based, costs money
  bge-m3  → local BAAI/bge-m3 (1024-dim), free, needs pip install FlagEmbedding

Set EMBEDDING_BACKEND in .env to switch.
NOTE: switching backend requires re-embed (different dim / model).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.config import settings

log = logging.getLogger(__name__)

# OpenAI pricing: $0.02 per 1M tokens (text-embedding-3-small)
_OPENAI_COST_PER_TOKEN = 0.00000002


def embed_texts(texts: list[str], batch_size: int = 32) -> tuple[list[list[float]], dict]:
    """
    Embed a list of texts. Returns (vectors, cost_info).

    cost_info = {
        "backend": str,
        "model": str,
        "total_tokens": int,          # OpenAI only
        "estimated_cost_usd": float,  # OpenAI only
        "elapsed_s": float,
    }
    """
    if not texts:
        return [], {"backend": settings.embedding_backend, "total_tokens": 0, "elapsed_s": 0}

    backend = settings.embedding_backend
    start = time.perf_counter()

    if backend == "openai":
        vectors, token_count = _embed_openai(texts, batch_size)
        elapsed = time.perf_counter() - start
        cost_info = {
            "backend": "openai",
            "model": settings.openai_embedding_model,
            "total_tokens": token_count,
            "estimated_cost_usd": round(token_count * _OPENAI_COST_PER_TOKEN, 6),
            "elapsed_s": round(elapsed, 2),
        }
    elif backend == "bge-m3":
        vectors = _embed_bge_m3(texts, batch_size)
        elapsed = time.perf_counter() - start
        cost_info = {
            "backend": "bge-m3",
            "model": "BAAI/bge-m3",
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "elapsed_s": round(elapsed, 2),
        }
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}. Use 'openai' or 'bge-m3'")

    log.info(
        "Embedded %d texts via %s in %.1fs (cost: $%.4f)",
        len(texts), backend, elapsed, cost_info.get("estimated_cost_usd", 0),
    )
    return vectors, cost_info


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Used at search time."""
    vectors, _ = embed_texts([text], batch_size=1)
    return vectors[0]


# ── OpenAI backend ─────────────────────────────────────────────────────────────

def _embed_openai(texts: list[str], batch_size: int) -> tuple[list[list[float]], int]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openai_embedding_model

    all_vectors: list[list[float]] = []
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace empty strings (API rejects them)
        batch = [t if t.strip() else "." for t in batch]

        response = client.embeddings.create(input=batch, model=model)
        all_vectors.extend([item.embedding for item in response.data])
        total_tokens += response.usage.total_tokens

        log.debug("Embedded batch %d/%d (%d tokens)", i // batch_size + 1, (len(texts) - 1) // batch_size + 1, response.usage.total_tokens)

    return all_vectors, total_tokens


# ── bge-m3 backend (local, Apple Silicon friendly) ────────────────────────────

_bge_model = None


def _embed_bge_m3(texts: list[str], batch_size: int) -> list[list[float]]:
    global _bge_model
    if _bge_model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            _bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        except ImportError:
            raise ImportError("Install FlagEmbedding: pip install FlagEmbedding")

    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = _bge_model.encode(batch, batch_size=batch_size, max_length=512)
        all_vectors.extend(result["dense_vecs"].tolist())

    return all_vectors
