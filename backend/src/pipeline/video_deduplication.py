"""Video hash helpers for skipping duplicate Phase 1 work."""
import hashlib
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from src.models.speaker import Speaker
from src.models.transcript import Transcript
from src.models.video import Video

_PROCESSED_STATUSES = {"ready_for_chunking", "ready"}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_processed_duplicate(db: Session, video_sha256: str) -> Optional[Video]:
    candidates = (
        db.query(Video)
        .filter(Video.video_sha256 == video_sha256)
        .filter(Video.duplicate_of_video_id.is_(None))
        .filter(Video.status.in_(_PROCESSED_STATUSES))
        .order_by(Video.created_at.asc())
        .all()
    )
    for video in candidates:
        transcript = db.query(Transcript).filter_by(video_id=video.id).first()
        if transcript and transcript.raw_json:
            return video
    return None


def canonical_video_id(video: Video):
    return video.duplicate_of_video_id or video.id


def create_duplicate_video_record(
    db: Session,
    *,
    video_id: uuid.UUID,
    duplicate: Video,
    video_sha256: str,
    original_filename: str,
) -> Video:
    video = Video(
        id=video_id,
        duplicate_of_video_id=duplicate.id,
        video_sha256=video_sha256,
        original_filename=original_filename,
        duration_ms=duplicate.duration_ms,
        language=duplicate.language,
        status="duplicate",
        category=duplicate.category,
        recorded_at=duplicate.recorded_at,
        processed_at=duplicate.processed_at,
    )
    db.add(video)

    source_transcript = db.query(Transcript).filter_by(video_id=duplicate.id).first()
    if source_transcript:
        db.add(
            Transcript(
                video_id=video_id,
                provider=source_transcript.provider,
                raw_json=deepcopy(source_transcript.raw_json),
                aligned_transcript=deepcopy(source_transcript.aligned_transcript),
                language=source_transcript.language,
                confidence=source_transcript.confidence,
                model_version=source_transcript.model_version,
            )
        )

    source_speakers = db.query(Speaker).filter_by(video_id=duplicate.id).all()
    for speaker in source_speakers:
        db.add(
            Speaker(
                video_id=video_id,
                diarization_label=speaker.diarization_label,
                display_name=speaker.display_name,
                role=speaker.role,
                mapping_status=speaker.mapping_status,
                mapped_by=speaker.mapped_by,
                mapped_at=speaker.mapped_at,
            )
        )

    db.commit()
    db.refresh(video)
    return video
