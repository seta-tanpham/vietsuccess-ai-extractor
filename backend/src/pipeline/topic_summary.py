"""
Topic-segment summarization — Phase 4 (hierarchical layer).

For each `topic_segment` chunk of a video, summarize the text of its child
semantic chunks into: summary + keywords + topic_label. The topic_label is
propagated down to the child semantic chunks so every retrievable (semantic)
chunk carries the name of the topic it belongs to.

Stored on the existing `chunks` columns (summary / keywords / topic_label) —
no schema change.
"""
from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy.orm import Session

from src.config import settings
from src.models.chunk import Chunk

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Bạn tóm tắt một đoạn chủ đề (topic segment) trong một video phỏng vấn/podcast tiếng Việt.
Chỉ xuất JSON hợp lệ. Viết bằng tiếng Việt."""

_USER_TEMPLATE = """NỘI DUNG ĐOẠN CHỦ ĐỀ:
{segment_text}

Xuất ra DUY NHẤT cấu trúc JSON sau:
{{
  "topic_label": "<nhãn chủ đề ngắn gọn, 3-6 từ>",
  "summary": "<2-3 câu tóm tắt nội dung đoạn này>",
  "keywords": ["<từ khóa>", "..."]
}}"""


def summarize_topics(video_id: uuid.UUID, db: Session) -> int:
    """Summarize every topic_segment chunk of a video. Returns count summarized."""
    if not settings.topic_summary_enabled:
        log.info("[%s] topic summarization disabled", video_id)
        return 0

    topics = (
        db.query(Chunk)
        .filter_by(video_id=video_id, chunk_type="topic_segment")
        .order_by(Chunk.start_ms)
        .all()
    )
    if not topics:
        return 0

    from openai import OpenAI

    if not settings.openai_api_key:
        log.warning("[%s] OPENAI_API_KEY not set — skipping topic summaries", video_id)
        return 0

    client = OpenAI(api_key=settings.openai_api_key)
    summarized = 0

    for topic in topics:
        children = (
            db.query(Chunk)
            .filter_by(parent_chunk_id=topic.id, chunk_type="semantic")
            .order_by(Chunk.start_ms)
            .all()
        )
        seg_text = " ".join((c.original_transcript or "") for c in children).strip()
        if not seg_text:
            continue
        seg_text = seg_text[: settings.summary_max_input_chars]

        try:
            data = _summarize_one(client, seg_text)
        except Exception as exc:
            log.warning("[%s] topic summary failed for %s (non-fatal): %s", video_id, topic.id, exc)
            continue

        topic.summary = data.get("summary")
        topic.keywords = data.get("keywords")
        topic.topic_label = (data.get("topic_label") or "")[:200] or None

        # Propagate topic_label down so each semantic chunk knows its topic.
        for child in children:
            child.topic_label = topic.topic_label

        summarized += 1

    db.commit()
    log.info("[%s] summarized %d/%d topic segments", video_id, summarized, len(topics))
    return summarized


def _summarize_one(client, segment_text: str) -> dict:
    response = client.chat.completions.create(
        model=settings.summary_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(segment_text=segment_text)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")
