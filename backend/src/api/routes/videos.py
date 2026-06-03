import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db
from src.models.transcript import Transcript
from src.models.video import Video, VideoAsset
from src.models.speaker import Speaker
from src.pipeline.phase1 import run_phase1
from src.pipeline.video_deduplication import (
    canonical_video_id,
    create_duplicate_video_record,
    find_processed_duplicate,
    sha256_bytes,
)
from src.storage.minio_client import ensure_buckets, upload_file

router = APIRouter()

_ALLOWED_MIME = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"}
_MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


@router.get("")
def list_videos(skip: int = Query(0, ge=0), limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    from src.models.chunk import Chunk
    from src.models.embedding import ChunkEmbedding
    videos = db.query(Video).order_by(Video.created_at.desc()).offset(skip).limit(limit).all()
    result = []
    for v in videos:
        n_speakers = db.query(Speaker).filter_by(video_id=v.id).count()
        eligible = db.query(Chunk).filter_by(video_id=v.id, chunk_type="semantic", is_low_quality=False).count()
        embedded = (
            db.query(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .filter(Chunk.video_id == v.id)
            .count()
        )
        result.append({
            "id": str(v.id),
            "title": v.title,
            "original_filename": v.original_filename,
            "status": v.status,
            "language": v.language,
            "duration_ms": v.duration_ms,
            "created_at": v.created_at.isoformat(),
            "n_speakers": n_speakers,
            "embedded": embedded,
            "eligible": eligible,
        })
    return {"total": len(result), "items": result}


@router.post("/upload", status_code=201)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload video → create Video record + raw_video asset → kick off Phase 1."""

    # Validate mime type
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Allowed: {_ALLOWED_MIME}")

    content = await file.read()
    if len(content) > _MAX_SIZE_BYTES:
        raise HTTPException(400, "File exceeds 2 GB limit")

    video_hash = sha256_bytes(content)
    duplicate = find_processed_duplicate(db, video_hash)
    if duplicate:
        video_id = uuid.uuid4()
        video = create_duplicate_video_record(
            db,
            video_id=video_id,
            duplicate=duplicate,
            video_sha256=video_hash,
            original_filename=file.filename or "upload",
        )
        return {
            "video_id": str(video_id),
            "status": video.status,
            "filename": video.original_filename,
            "video_sha256": video.video_sha256,
            "duplicate_of_video_id": str(duplicate.id),
            "message": "Duplicate video detected; transcript and speakers copied, Phase 1 processing skipped.",
        }

    ensure_buckets()

    video_id = uuid.uuid4()
    object_key = f"{video_id}/raw{Path(file.filename or 'video.mp4').suffix}"

    # Upload to MinIO
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or ".mp4").suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        upload_file(settings.minio_bucket_videos, object_key, tmp_path, file.content_type)
    finally:
        os.unlink(tmp_path)

    # Persist video + asset records
    video = Video(id=video_id, video_sha256=video_hash, original_filename=file.filename or "upload", status="uploaded")
    db.add(video)

    asset = VideoAsset(
        video_id=video_id,
        asset_type="raw_video",
        minio_bucket=settings.minio_bucket_videos,
        object_key=object_key,
        mime_type=file.content_type,
        size_bytes=len(content),
    )
    db.add(asset)
    db.commit()
    db.refresh(video)

    # Run Phase 1 in background thread (non-blocking)
    _run_background(video_id)

    return {
        "video_id": str(video_id),
        "status": video.status,
        "filename": video.original_filename,
        "video_sha256": video.video_sha256,
        "duplicate_of_video_id": None,
    }


@router.get("/{video_id}")
def get_video(video_id: uuid.UUID, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    canonical_id = canonical_video_id(video)
    transcript = db.query(Transcript).filter_by(video_id=canonical_id).first()
    speakers = db.query(Speaker).filter_by(video_id=canonical_id).all()

    video_url = None
    for asset in video.assets:
        if asset.asset_type == "raw_video":
            try:
                from src.storage.minio_client import get_presigned_url
                video_url = get_presigned_url(asset.minio_bucket, asset.object_key)
            except Exception:
                video_url = None

    return {
        "id": str(video.id),
        "canonical_video_id": str(canonical_id),
        "duplicate_of_video_id": str(video.duplicate_of_video_id) if video.duplicate_of_video_id else None,
        "video_sha256": video.video_sha256,
        "filename": video.original_filename,
        "status": video.status,
        "language": video.language,
        "created_at": video.created_at.isoformat(),
        "transcript_confidence": transcript.confidence if transcript else None,
        "video_url": video_url,
        "speakers": [
            {
                "id": str(s.id),
                "label": s.diarization_label,
                "display_name": s.display_name,
                "mapping_status": s.mapping_status,
            }
            for s in speakers
        ],
    }


@router.get("/{video_id}/transcript")
def get_transcript(video_id: uuid.UUID, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    canonical_id = canonical_video_id(video)
    transcript = db.query(Transcript).filter_by(video_id=canonical_id).first()
    if not transcript:
        raise HTTPException(404, "Transcript not found — Phase 1 not complete yet")

    return {
        "video_id": str(video_id),
        "canonical_video_id": str(canonical_id),
        "confidence": transcript.confidence,
        "language": transcript.language,
        "aligned_turns": transcript.aligned_transcript or [],
        "raw_segments": (transcript.raw_json or {}).get("segments", []),
    }


@router.patch("/{video_id}/speakers/{speaker_id}")
def confirm_speaker(video_id: uuid.UUID, speaker_id: uuid.UUID, display_name: str, role: str = "guest", db: Session = Depends(get_db)):
    speaker = db.get(Speaker, speaker_id)
    if not speaker or speaker.video_id != video_id:
        raise HTTPException(404, "Speaker not found")

    speaker.display_name = display_name
    speaker.role = role
    speaker.mapping_status = "confirmed"
    db.commit()
    return {"id": str(speaker.id), "display_name": display_name, "mapping_status": "confirmed"}


@router.post("/{video_id}/reprocess")
def reprocess(video_id: uuid.UUID, db: Session = Depends(get_db)):
    """Re-run Phase 1 from scratch (idempotent — skips already-done steps)."""
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    if video.duplicate_of_video_id:
        return {
            "video_id": str(video_id),
            "canonical_video_id": str(video.duplicate_of_video_id),
            "message": "Duplicate video; reprocess the canonical video instead.",
        }

    _run_background(video_id)
    return {"video_id": str(video_id), "message": "Phase 1 re-triggered"}


def _run_background(video_id: uuid.UUID) -> None:
    from src.database import SessionLocal
    from src.pipeline.phase2 import run_phase2
    from src.pipeline.phase3 import run_phase3
    from src.models.video import Video
    import logging

    def _task():
        db = SessionLocal()
        try:
            logging.getLogger(__name__).info("[%s] Starting background pipeline Phase 1", video_id)
            run_phase1(video_id, db)

            logging.getLogger(__name__).info("[%s] Starting background pipeline Phase 2", video_id)
            run_phase2(video_id, db)

            logging.getLogger(__name__).info("[%s] Starting background pipeline Phase 3", video_id)
            run_phase3(video_id, db)
        except Exception as exc:
            logging.getLogger(__name__).exception("Full background pipeline failed for %s: %s", video_id, exc)
            try:
                video = db.get(Video, video_id)
                if video:
                    video.status = "error"
                    db.commit()
            except Exception:
                db.rollback()
        finally:
            db.close()

    thread = threading.Thread(target=_task, daemon=True)
    thread.start()
