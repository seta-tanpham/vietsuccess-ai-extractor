"""
Phase 3 orchestrator: Quality-passed semantic chunks → embeddings → pgvector HNSW index.

Steps:
  1. Filter: only chunk_type=semantic AND is_low_quality=False
  2. Batch embed search_text via configured backend
  3. Persist to chunk_embeddings (idempotent — skip already-embedded chunks)
  4. Create HNSW index on chunk_embeddings.embedding
  5. Create metadata indexes (video_id, speaker_id, chunk_type)
  6. Log cost per video
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.chunk import Chunk
from src.models.embedding import ChunkEmbedding
from src.models.video import Video
from src.pipeline.embedding import embed_texts

log = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32


def run_phase3(video_id: uuid.UUID, db: Session) -> dict:
    """
    Embed all quality-passed semantic chunks for a video.
    Returns cost + coverage report.
    """
    video = db.get(Video, video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    video.status = "embedding"
    db.commit()

    try:
        result = _embed_video_chunks(video_id, db)
        _ensure_hnsw_index(db)
        # Phase 4 (summary) advances to `ready`; mark the embedding stage done here.
        video.status = "ready_for_summary"
        db.commit()
        log.info("[%s] Phase 3 complete ✓ — %d chunks embedded", video_id, result["embedded_count"])
        return result
    except Exception as exc:
        db.rollback()
        log.exception("[%s] Phase 3 failed: %s", video_id, exc)
        video.status = "error"
        db.commit()
        raise


def _embed_video_chunks(video_id: uuid.UUID, db: Session) -> dict:
    # Only semantic chunks that passed QC
    eligible = (
        db.query(Chunk)
        .filter_by(video_id=video_id, chunk_type="semantic", is_low_quality=False)
        .order_by(Chunk.start_ms)
        .all()
    )

    if not eligible:
        log.warning("[%s] No eligible semantic chunks to embed", video_id)
        return {"video_id": str(video_id), "eligible": 0, "embedded_count": 0, "skipped": 0}

    # Idempotent: skip chunks that already have an embedding
    already_embedded = {
        row.chunk_id
        for row in db.query(ChunkEmbedding.chunk_id).filter(
            ChunkEmbedding.chunk_id.in_([c.id for c in eligible])
        ).all()
    }

    to_embed = [c for c in eligible if c.id not in already_embedded]
    skipped = len(eligible) - len(to_embed)

    if not to_embed:
        log.info("[%s] All %d chunks already embedded", video_id, len(eligible))
        return {"video_id": str(video_id), "eligible": len(eligible), "embedded_count": 0, "skipped": skipped}

    log.info("[%s] Embedding %d chunks (skipping %d already done)...", video_id, len(to_embed), skipped)

    # Batch embed
    texts = [c.search_text or c.original_transcript or "" for c in to_embed]
    from src.config import settings
    vectors, cost_info = embed_texts(texts, batch_size=_EMBED_BATCH_SIZE)

    # Persist embeddings
    model_name = cost_info["model"]
    for chunk, vector in zip(to_embed, vectors):
        db.add(ChunkEmbedding(
            chunk_id=chunk.id,
            embedding=vector,
            model_name=model_name,
        ))

    db.commit()

    log.info(
        "[%s] Embedded %d chunks | backend=%s | tokens=%d | cost=$%.4f | %.1fs",
        video_id,
        len(to_embed),
        cost_info["backend"],
        cost_info.get("total_tokens", 0),
        cost_info.get("estimated_cost_usd", 0.0),
        cost_info["elapsed_s"],
    )

    return {
        "video_id": str(video_id),
        "eligible": len(eligible),
        "embedded_count": len(to_embed),
        "skipped": skipped,
        **cost_info,
    }


def _ensure_hnsw_index(db: Session) -> None:
    """
    Create HNSW index on chunk_embeddings if it doesn't exist.
    HNSW is preferred over ivfflat for small-medium scale (faster query, no train step).
    """
    db.commit()
    with db.get_bind().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ce_embedding_hnsw
            ON chunk_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))

    # Metadata pre-filter indexes
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks (video_id)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks (chunk_type)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_is_low_quality ON chunks (is_low_quality)"))
    db.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_speaker_id ON chunks (speaker_id) WHERE speaker_id IS NOT NULL"))

    db.commit()
    log.info("HNSW index and metadata indexes ensured")


def re_embed_video(video_id: uuid.UUID, db: Session) -> dict:
    """
    Delete existing embeddings for this video and re-run phase3.
    Use when switching embedding model.
    """
    chunks = db.query(Chunk).filter_by(video_id=video_id).all()
    chunk_ids = [c.id for c in chunks]

    if chunk_ids:
        deleted = db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id.in_(chunk_ids)).delete(synchronize_session=False)
        db.commit()
        log.info("[%s] Deleted %d embeddings for re-embed", video_id, deleted)

    return run_phase3(video_id, db)
