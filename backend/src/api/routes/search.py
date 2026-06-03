"""
Vector search API — Phase 3 output.

GET /search?q=text&video_id=X&speaker_id=Y&limit=10

Returns ranked chunks by cosine similarity. No LLM — raw vector search only.
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database import get_db
from src.pipeline.embedding import embed_query

router = APIRouter()


def _scope_filter(scope: str) -> str:
    """Build the chunk_type/quality filter for a given search scope."""
    if scope == "summaries":
        return "c.chunk_type IN ('topic_segment', 'video_summary')"
    if scope == "all":
        return (
            "((c.chunk_type = 'semantic' AND c.is_low_quality = false) "
            "OR c.chunk_type IN ('topic_segment', 'video_summary'))"
        )
    # default: 'chunks' — quality-passed semantic chunks only
    return "c.chunk_type = 'semantic' AND c.is_low_quality = false"


@router.get("")
def search(
    q: str = Query(..., min_length=1, description="Search query in Vietnamese or English"),
    video_id: Optional[uuid.UUID] = Query(None, description="Limit to a specific video"),
    speaker_id: Optional[uuid.UUID] = Query(None, description="Limit to a specific speaker"),
    scope: str = Query("chunks", pattern="^(chunks|summaries|all)$",
                       description="chunks = transcript chunks; summaries = topic/video summaries; all = both"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """
    Vector search. Each transcript chunk result is enriched with the summary of its
    topic segment (topic_summary) and the whole-video summary (video_summary), so a
    retrieved chunk always carries both layers of context.
    """
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Embed the query
    try:
        query_vector = embed_query(q.strip())
    except Exception as exc:
        raise HTTPException(500, f"Embedding failed: {exc}")

    # Build SQL with optional pre-filters (reduces search space before cosine scan)
    filters = [_scope_filter(scope), "ce.embedding IS NOT NULL"]
    params: dict = {"query_vec": str(query_vector), "limit": limit}

    if video_id:
        filters.append("c.video_id = :video_id")
        params["video_id"] = str(video_id)

    if speaker_id:
        filters.append("c.speaker_id = :speaker_id")
        params["speaker_id"] = str(speaker_id)

    where_clause = " AND ".join(filters)

    sql = text(f"""
        SELECT
            c.id::text              AS chunk_id,
            c.video_id::text        AS video_id,
            c.chunk_type,
            c.start_ms,
            c.end_ms,
            c.speaker_id::text      AS speaker_id,
            c.original_transcript,
            c.search_text,
            c.summary               AS chunk_summary,
            c.topic_label,
            c.segment_path,
            c.chapter_index,
            s.display_name          AS speaker_name,
            v.title                 AS video_title,
            v.original_filename,
            tc.summary              AS topic_summary,
            tc.topic_label          AS topic_label_parent,
            vsum.short_summary      AS video_summary,
            1 - (ce.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM chunks c
        JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        JOIN videos v ON v.id = c.video_id
        LEFT JOIN speakers s ON s.id = c.speaker_id
        LEFT JOIN chunks tc ON tc.id = c.parent_chunk_id AND tc.chunk_type = 'topic_segment'
        LEFT JOIN video_summaries vsum ON vsum.video_id = c.video_id
        WHERE {where_clause}
        ORDER BY ce.embedding <=> CAST(:query_vec AS vector)
        LIMIT :limit
    """)

    rows = db.execute(sql, params).mappings().all()

    results = [
        {
            "chunk_id": row["chunk_id"],
            "video_id": row["video_id"],
            "video_title": row["video_title"] or row["original_filename"],
            "type": row["chunk_type"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "duration_ms": row["end_ms"] - row["start_ms"],
            "speaker_id": row["speaker_id"],
            "speaker_name": row["speaker_name"],
            # For summary rows original_transcript is null → fall back to the summary text.
            "transcript": row["original_transcript"] or row["chunk_summary"],
            "topic_label": row["topic_label"] or row["topic_label_parent"],
            # Two summary layers attached to every retrieved chunk:
            "topic_summary": row["topic_summary"],
            "video_summary": row["video_summary"],
            "segment_path": row["segment_path"],
            "chapter_index": row["chapter_index"],
            "score": round(float(row["score"]), 4),
        }
        for row in rows
    ]

    return {
        "query": q,
        "scope": scope,
        "total": len(results),
        "results": results,
    }


@router.get("/videos/{video_id}/coverage")
def embedding_coverage(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Check how many semantic chunks have embeddings vs total eligible."""
    sql = text("""
        SELECT
            COUNT(*) FILTER (WHERE c.is_low_quality = false AND c.chunk_type = 'semantic') AS eligible,
            COUNT(ce.id) FILTER (WHERE c.is_low_quality = false AND c.chunk_type = 'semantic') AS embedded
        FROM chunks c
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        WHERE c.video_id = :video_id
    """)
    row = db.execute(sql, {"video_id": str(video_id)}).mappings().first()

    eligible = row["eligible"] or 0
    embedded = row["embedded"] or 0
    missing = eligible - embedded

    return {
        "video_id": str(video_id),
        "eligible": eligible,
        "embedded": embedded,
        "missing": missing,
        "coverage_pct": round(embedded / eligible * 100, 1) if eligible else 0,
        "ready": missing == 0 and eligible > 0,
    }
