"""
Chunk API routes:
  POST /chunks/{video_id}/run        — trigger Phase 2
  GET  /chunks/{video_id}            — list chunks (filterable)
  GET  /chunks/{video_id}/qc         — QC dashboard per video
  POST /chunks/{video_id}/rechunk    — re-run Phase 2 (wipes existing chunks)
"""
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.chunk import Chunk
from src.models.video import Video
from src.pipeline.chunking.quality import generate_qc_report

router = APIRouter()


@router.post("/{video_id}/run", status_code=202)
def run_phase2(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Trigger Phase 2 chunking in background."""
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    if video.status not in ("ready_for_chunking", "quality_check", "ready_for_embedding", "error"):
        raise HTTPException(400, f"Video status '{video.status}' not ready for chunking. Need: ready_for_chunking")

    _run_background(video_id)
    return {"video_id": str(video_id), "message": "Phase 2 started in background"}


@router.get("/{video_id}")
def list_chunks(
    video_id: uuid.UUID,
    chunk_type: str | None = Query(None, description="atomic | semantic | topic_segment"),
    only_good: bool = Query(False, description="Only return is_low_quality=false chunks"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
):
    """List chunks for a video with optional filters."""
    q = db.query(Chunk).filter_by(video_id=video_id)
    if chunk_type:
        q = q.filter_by(chunk_type=chunk_type)
    if only_good:
        q = q.filter_by(is_low_quality=False)
    q = q.order_by(Chunk.start_ms)

    total = q.count()
    chunks = q.offset(skip).limit(limit).all()

    return {
        "video_id": str(video_id),
        "total": total,
        "items": [_serialize_chunk(c) for c in chunks],
    }


@router.get("/{video_id}/qc")
def qc_dashboard(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """QC dashboard: per-video quality report broken down by fail reason."""
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    semantic_chunks = (
        db.query(Chunk)
        .filter_by(video_id=video_id, chunk_type="semantic")
        .order_by(Chunk.start_ms)
        .all()
    )

    if not semantic_chunks:
        return {
            "video_id": str(video_id),
            "status": video.status,
            "message": "No semantic chunks yet — run Phase 2 first",
        }

    report = generate_qc_report(semantic_chunks)

    # Collect sample of failed chunks for debugging
    failed_samples = [
        _serialize_chunk(c)
        for c in semantic_chunks
        if c.is_low_quality
    ][:10]

    return {
        "video_id": str(video_id),
        "video_status": video.status,
        **report,
        "failed_samples": failed_samples,
        "recommendation": _qc_recommendation(report),
    }


@router.post("/{video_id}/rechunk", status_code=202)
def rechunk(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Re-run Phase 2 from scratch.
    Safe: wipes existing chunks but keeps transcript data intact.
    """
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    if not video.status or video.status == "uploading":
        raise HTTPException(400, "Video has no transcript yet")

    _run_background(video_id)
    return {"video_id": str(video_id), "message": "Re-chunking started — existing chunks will be replaced"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_background(video_id: uuid.UUID) -> None:
    from src.database import SessionLocal
    from src.pipeline.phase2 import run_phase2

    def _task():
        db = SessionLocal()
        try:
            run_phase2(video_id, db)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Phase 2 failed for %s: %s", video_id, exc)
        finally:
            db.close()

    threading.Thread(target=_task, daemon=True).start()


def _serialize_chunk(c: Chunk) -> dict:
    return {
        "id": str(c.id),
        "chunk_type": c.chunk_type,
        "segment_path": c.segment_path,
        "start_ms": c.start_ms,
        "end_ms": c.end_ms,
        "duration_ms": c.end_ms - c.start_ms,
        "speaker_id": str(c.speaker_id) if c.speaker_id else None,
        "is_low_quality": c.is_low_quality,
        "quality_fail_reasons": c.quality_fail_reasons,
        "original_transcript": (c.original_transcript or "")[:200],
        "chapter_index": c.chapter_index,
    }


def _qc_recommendation(report: dict) -> str:
    pct = report.get("low_quality_pct", 0)
    breakdown = report.get("fail_breakdown", {})

    if pct == 0:
        return "All chunks passed QC. Ready for embedding."
    if not report.get("above_threshold"):
        return f"{pct}% low quality (under 8% threshold). Acceptable — proceed to embedding."

    tips = []
    if "low_confidence" in breakdown:
        tips.append("Low confidence: check audio quality or try a larger Whisper model")
    if "too_many_null_speakers" in breakdown or "needs_speaker_confirm" in breakdown:
        tips.append("Null speakers: confirm speaker names in the speaker mapping UI")
    if "too_short" in breakdown:
        tips.append("Too short: decrease ATOMIC_MIN_DURATION_MS or check diarization")
    if "timestamp_overlap" in breakdown:
        tips.append("Timestamp overlap: diarization pipeline may be corrupted — re-run Phase 1")

    return f"{pct}% low quality (> 8% threshold). Action needed: " + "; ".join(tips)
