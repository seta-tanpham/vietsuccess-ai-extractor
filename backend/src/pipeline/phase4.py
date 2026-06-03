"""
Phase 4 orchestrator: aligned transcript → LLM summary / topics → video_summaries.

Runs after Phase 3 (embedding) for BOTH upload and YouTube videos. Produces a
single VideoSummary row per video: short/long summary, key takeaways, topics,
mentioned entities. Non-fatal by design — a summary failure must not block a
video from reaching `ready`.

Status transitions: ready_for_summary → summarizing → ready
"""
from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy.orm import Session

from src.config import settings
from src.models.summary import VideoSummary
from src.models.transcript import Transcript
from src.models.video import Video
from src.pipeline.video_deduplication import canonical_video_id

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You analyze Vietnamese interview/podcast transcripts and produce a concise editorial summary.
Output ONLY valid JSON. Write summaries and topics in Vietnamese."""

_USER_TEMPLATE = """VIDEO TITLE: {title}

TRANSCRIPT:
{transcript}

Produce ONLY this JSON structure:
{{
  "short_summary": "<2-3 câu tóm tắt nhanh>",
  "long_summary": "<1-2 đoạn tóm tắt đầy đủ>",
  "key_takeaways": ["<bài học/ý chính>", "..."],
  "main_topics": ["<chủ đề chính>", "..."],
  "sub_topics": ["<chủ đề phụ>", "..."],
  "mentioned_entities": ["<người/công ty/sách/thuật ngữ được nhắc tới>", "..."]
}}"""


def run_phase4(video_id: uuid.UUID, db: Session) -> dict:
    """Generate and persist a VideoSummary. Sets status to `ready` when done."""
    video = db.get(Video, video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    if not settings.summary_enabled:
        log.info("[%s] Phase 4 skipped (summary_enabled=false)", video_id)
        video.status = "ready"
        db.commit()
        return {"video_id": str(video_id), "skipped": True}

    video.status = "summarizing"
    db.commit()

    try:
        # ── 1. Per-topic-segment summaries (hierarchical layer) ───────────────
        from src.pipeline.topic_summary import summarize_topics
        n_topics = summarize_topics(video_id, db)

        # ── 2. Whole-video summary ────────────────────────────────────────────
        data = None
        transcript_text = _gather_transcript_text(video_id, db)
        if transcript_text.strip():
            data = _summarize(video.title or "", transcript_text)
            _upsert_summary(video_id, data, db)
        else:
            log.warning("[%s] Phase 4: no transcript text, skipping video summary", video_id)

        # ── 3. Embed topic + video summaries into RAG ─────────────────────────
        n_embedded = 0
        if settings.summary_embed_enabled:
            n_embedded = _embed_summaries(video_id, db, data)

        video.status = "ready"
        db.commit()
        log.info("[%s] Phase 4 complete ✓ — %d topic summaries, %d summary embeddings",
                 video_id, n_topics, n_embedded)
        return {"video_id": str(video_id), "topic_summaries": n_topics,
                "summary_embeddings": n_embedded, "summary": data}

    except Exception as exc:
        db.rollback()
        # Summary is non-essential — don't strand the video in `error`; mark it ready.
        log.warning("[%s] Phase 4 summary failed (non-fatal): %s", video_id, exc)
        video = db.get(Video, video_id)
        if video:
            video.status = "ready"
            db.commit()
        return {"video_id": str(video_id), "skipped": True, "reason": "summary_error"}


def _embed_summaries(video_id: uuid.UUID, db: Session, video_summary_data: Optional[dict]) -> int:
    """Embed topic-segment summaries + the video summary into chunk_embeddings so
    they are retrievable in RAG. Idempotent (clears prior summary embeddings first).
    The video summary is stored on a synthetic chunk_type='video_summary' chunk."""
    from src.models.chunk import Chunk
    from src.models.embedding import ChunkEmbedding
    from src.pipeline.embedding import embed_texts

    video = db.get(Video, video_id)

    # Ensure a single video_summary chunk holding the whole-video summary text.
    vs_text = ""
    if video_summary_data:
        vs_text = (video_summary_data.get("long_summary")
                   or video_summary_data.get("short_summary") or "")
    vs_chunk = db.query(Chunk).filter_by(video_id=video_id, chunk_type="video_summary").first()
    if vs_text:
        if not vs_chunk:
            vs_chunk = Chunk(
                video_id=video_id, chunk_type="video_summary",
                start_ms=0, end_ms=(video.duration_ms or 0),
            )
            db.add(vs_chunk)
            db.flush()
        vs_chunk.summary = vs_text
        vs_chunk.search_text = vs_text
        db.commit()

    # Collect embedding targets: (chunk_id, text)
    topics = db.query(Chunk).filter_by(video_id=video_id, chunk_type="topic_segment").all()
    targets: list[tuple] = [(t.id, t.summary) for t in topics if t.summary]
    if vs_chunk and vs_text:
        targets.append((vs_chunk.id, vs_text))
    if not targets:
        return 0

    # Idempotent: drop any existing embeddings for these summary chunks.
    ids = [cid for cid, _ in targets]
    db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id.in_(ids)).delete(synchronize_session=False)
    db.commit()

    vectors, cost_info = embed_texts([txt for _, txt in targets])
    for (cid, _), vector in zip(targets, vectors):
        db.add(ChunkEmbedding(chunk_id=cid, embedding=vector, model_name=cost_info["model"]))
    db.commit()
    return len(targets)


def _gather_transcript_text(video_id: uuid.UUID, db: Session) -> str:
    """Join aligned-transcript turns into a single speaker-labelled text block,
    capped at summary_max_input_chars."""
    video = db.get(Video, video_id)
    canonical_id = canonical_video_id(video) if video else video_id
    transcript = db.query(Transcript).filter_by(video_id=canonical_id).first()
    if not transcript or not transcript.aligned_transcript:
        return ""

    lines = []
    for turn in transcript.aligned_transcript:
        speaker = turn.get("speaker", "")
        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}" if speaker else text)
    full = "\n".join(lines)
    return full[: settings.summary_max_input_chars]


def _summarize(title: str, transcript_text: str) -> dict:
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = _USER_TEMPLATE.format(title=title or "VietSuccess interview", transcript=transcript_text)
    response = client.chat.completions.create(
        model=settings.summary_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def _upsert_summary(video_id: uuid.UUID, data: dict, db: Session) -> None:
    summary = db.query(VideoSummary).filter_by(video_id=video_id).first()
    if summary is None:
        summary = VideoSummary(video_id=video_id)
        db.add(summary)

    summary.short_summary = data.get("short_summary")
    summary.long_summary = data.get("long_summary")
    summary.key_takeaways = data.get("key_takeaways")
    summary.main_topics = data.get("main_topics")
    summary.sub_topics = data.get("sub_topics")
    summary.mentioned_entities = data.get("mentioned_entities")
    summary.model_version = settings.summary_model
    db.commit()
