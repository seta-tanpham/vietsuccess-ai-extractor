"""
Phase 2 orchestrator: Clean aligned transcript → Chunked + Quality-checked chunks.

Steps:
  1. Atomic chunking  (rule 3.4)
  2. Semantic chunking with overlap (rule 3.5)
  3. Topic segment grouping (chapter heuristic)
  4. segment_path generation (ltree format)
  5. Quality check gate (6 conditions)
  6. Speaker confirm flow — flag semantics with > 30% null speaker
  7. Persist all chunks to DB
  8. Idempotent: deletes existing chunks for the video before re-running
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from src.config import settings
from src.models.chunk import Chunk
from src.models.speaker import Speaker
from src.models.transcript import Transcript
from src.models.video import Video
from src.pipeline.chunking.atomic import build_atomic_chunks
from src.pipeline.chunking.quality import (
    check_semantic_chunk,
    check_timestamp_integrity,
    generate_qc_report,
)
from src.pipeline.chunking.semantic import (
    assign_segment_paths,
    build_semantic_chunks,
)
from src.pipeline.chunking.topic import group_into_topics

log = logging.getLogger(__name__)


def run_phase2(video_id: uuid.UUID, db: Session) -> dict:
    """
    Run Phase 2 for a video. Returns QC report dict.
    Idempotent: deletes existing chunks first so it can be re-run safely.
    """
    video = db.get(Video, video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    transcript = db.query(Transcript).filter_by(video_id=video_id).first()
    if not transcript or not transcript.aligned_transcript:
        raise RuntimeError("Phase 1 not complete — no aligned_transcript found")

    # Clean state: delete existing chunks for this video (re-run safety)
    existing_count = db.query(Chunk).filter_by(video_id=video_id).count()
    if existing_count:
        db.query(Chunk).filter_by(video_id=video_id).delete()
        db.commit()
        log.info("[%s] Deleted %d existing chunks for re-run", video_id, existing_count)

    video.status = "chunking"
    db.commit()

    # ── Step 1: Build atomic chunks ───────────────────────────────────────────
    log.info("[%s] Building atomic chunks...", video_id)
    atomic_data = build_atomic_chunks(
        aligned_turns=transcript.aligned_transcript,
        whisper_raw_json=transcript.raw_json or {},
        atomic_max_ms=settings.atomic_max_duration_ms,
        atomic_min_ms=settings.atomic_min_duration_ms,
    )
    log.info("[%s] %d atomic chunks", video_id, len(atomic_data))

    # ── Step 2: Build semantic chunks ─────────────────────────────────────────
    log.info("[%s] Building semantic chunks...", video_id)
    semantic_data = build_semantic_chunks(
        atomics=atomic_data,
        target_min_ms=settings.semantic_target_min_ms,
        target_max_ms=settings.semantic_target_max_ms,
        overlap_ms=settings.semantic_overlap_ms,
    )
    log.info("[%s] %d semantic chunks", video_id, len(semantic_data))

    # ── Step 3: Group into topic segments ────────────────────────────────────
    atomic_indices_per_semantic = [s.atomic_indices for s in semantic_data]
    topic_groups = group_into_topics(semantic_data, atomic_data, atomic_indices_per_semantic)
    log.info("[%s] %d topic segments", video_id, len(topic_groups))

    # ── Step 4: Build speaker lookup ─────────────────────────────────────────
    speakers = db.query(Speaker).filter_by(video_id=video_id).all()
    speaker_map = {s.diarization_label: s.id for s in speakers}

    # ── Step 5: Persist to DB ─────────────────────────────────────────────────
    semantic_chunk_ids: list[uuid.UUID] = []
    atomic_chunk_objs: list[Chunk] = []
    semantic_chunk_objs: list[Chunk] = []

    for topic_idx, sem_indices in enumerate(topic_groups, start=1):
        topic_sem_data = [semantic_data[i] for i in sem_indices]

        # Create topic_segment chunk
        topic_start = atomic_data[topic_sem_data[0].atomic_indices[0]].start_ms
        topic_end = atomic_data[topic_sem_data[-1].atomic_indices[-1]].end_ms
        seg_path_topic = f"topic_{topic_idx}"

        topic_chunk = Chunk(
            video_id=video_id,
            chunk_type="topic_segment",
            segment_path=seg_path_topic,
            start_ms=topic_start,
            end_ms=topic_end,
            chapter_index=topic_idx,
        )
        db.add(topic_chunk)
        db.flush()  # get topic_chunk.id

        # Create semantic + atomic chunks for this topic
        sem_paths = assign_segment_paths(topic_idx, topic_sem_data)

        for sem_local_idx, (sem, sem_path) in enumerate(zip(topic_sem_data, sem_paths), start=1):
            # Combined text for semantic chunk = all core atomic texts
            core_atomics = [atomic_data[i] for i in sem.atomic_indices]
            sem_text = " ".join(a.original_transcript for a in core_atomics).strip()
            sem_search = " ".join(a.search_text for a in core_atomics).strip()

            # Find dominant speaker for this semantic chunk
            sem_speakers = [a.speaker for a in core_atomics if getattr(a, 'speaker', None)]
            dominant_speaker_id = None
            if sem_speakers:
                from collections import Counter
                dominant_label = Counter(sem_speakers).most_common(1)[0][0]
                dominant_speaker_id = speaker_map.get(dominant_label)

            sem_chunk = Chunk(
                video_id=video_id,
                parent_chunk_id=topic_chunk.id,
                speaker_id=dominant_speaker_id,
                chunk_type="semantic",
                segment_path=sem_path,
                start_ms=sem.start_ms,
                end_ms=sem.end_ms,
                original_transcript=sem_text,
                search_text=sem_search,
                chapter_index=topic_idx,
            )
            db.add(sem_chunk)
            db.flush()  # get sem_chunk.id
            semantic_chunk_ids.append(sem_chunk.id)
            semantic_chunk_objs.append(sem_chunk)

            # Create atomic chunks, assign parent
            for atom_local_idx, atom_idx in enumerate(sem.atomic_indices, start=1):
                a = atomic_data[atom_idx]
                atom_path = f"{sem_path}.atomic_{atom_local_idx}"
                speaker_id = speaker_map.get(a.speaker)

                atom_chunk = Chunk(
                    video_id=video_id,
                    parent_chunk_id=sem_chunk.id,
                    speaker_id=speaker_id,
                    chunk_type="atomic",
                    segment_path=atom_path,
                    start_ms=a.start_ms,
                    end_ms=a.end_ms,
                    original_transcript=a.original_transcript,
                    search_text=a.search_text,
                    chapter_index=topic_idx,
                )
                db.add(atom_chunk)
                atomic_chunk_objs.append(atom_chunk)

    db.flush()

    # ── Step 6: Quality check ─────────────────────────────────────────────────
    log.info("[%s] Running quality check...", video_id)

    for sem_chunk in semantic_chunk_objs:
        # Get atomic children
        child_atomics = [c for c in atomic_chunk_objs if c.parent_chunk_id == sem_chunk.id]

        # Enrich atomics with speaker label for null-speaker check
        for a in child_atomics:
            if a.speaker_id is None:
                a.__dict__["speaker"] = "SPEAKER_UNKNOWN"

        qc = check_semantic_chunk(
            semantic=sem_chunk,
            atomic_chunks=child_atomics,
            min_confidence=settings.quality_min_confidence,
            min_duration_ms=settings.quality_min_duration_ms,
            max_duration_ms=settings.quality_max_duration_ms,
            max_null_speaker_ratio=settings.quality_max_null_speaker_ratio,
        )
        sem_chunk.is_low_quality = qc.is_low_quality
        sem_chunk.quality_fail_reasons = qc.fail_reasons if qc.fail_reasons else None

    # Timestamp integrity check (check 6)
    bad_indices = check_timestamp_integrity(semantic_chunk_objs)
    if bad_indices:
        log.error("[%s] Timestamp overlap detected in %d semantic chunks — possible diarization corruption", video_id, len(bad_indices))
        for idx in bad_indices:
            sem = semantic_chunk_objs[idx]
            reasons = (sem.quality_fail_reasons or []) + ["timestamp_overlap"]
            sem.is_low_quality = True
            sem.quality_fail_reasons = reasons

    # ── Step 7: Speaker confirm flow ─────────────────────────────────────────
    _flag_semantics_needing_speaker_confirm(semantic_chunk_objs, atomic_chunk_objs, db)

    video.status = "quality_check"
    db.commit()

    # ── Step 8: QC Report ────────────────────────────────────────────────────
    report = generate_qc_report(semantic_chunk_objs)
    log.info(
        "[%s] QC: %d/%d semantic chunks low-quality (%.1f%%). Reasons: %s",
        video_id,
        report["low_quality_count"],
        report["total"],
        report["low_quality_pct"],
        report["fail_breakdown"],
    )

    if report["above_threshold"]:
        log.warning("[%s] Low quality rate %.1f%% > 8%% — investigate before embedding", video_id, report["low_quality_pct"])

    video.status = "ready_for_embedding"
    db.commit()

    return {
        "video_id": str(video_id),
        "atomic_count": len(atomic_chunk_objs),
        "semantic_count": len(semantic_chunk_objs),
        "topic_count": len(topic_groups),
        **report,
    }


def _flag_semantics_needing_speaker_confirm(
    semantics: list[Chunk],
    atomics: list[Chunk],
    db: Session,
) -> None:
    """
    If a semantic chunk has > 30% null-speaker atomics, mark it is_low_quality
    and add 'needs_speaker_confirm' reason.
    """
    for sem in semantics:
        children = [a for a in atomics if a.parent_chunk_id == sem.id]
        if not children:
            continue
        null_count = sum(1 for a in children if a.speaker_id is None)
        ratio = null_count / len(children)
        if ratio > 0.30 and "too_many_null_speakers" not in (sem.quality_fail_reasons or []):
            reasons = (sem.quality_fail_reasons or []) + ["needs_speaker_confirm"]
            sem.is_low_quality = True
            sem.quality_fail_reasons = reasons
