"""
Full pipeline runner: video file → ready for search.

Usage (from code):
    from src.pipeline.runner import ingest_video
    result = ingest_video("/path/to/video.webm", db)

Usage (CLI):
    python scripts/run_pipeline.py video/myvideo.webm
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from src.config import settings
from src.models.video import Video, VideoAsset
from src.pipeline.phase1 import run_phase1
from src.pipeline.phase2 import run_phase2
from src.pipeline.phase3 import run_phase3
from src.storage.minio_client import ensure_buckets, upload_file

log = logging.getLogger(__name__)


def ingest_video(
    file_path: str,
    db: Session,
    title: str | None = None,
    run_phases: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """
    Full pipeline: register video → Phase 1 → Phase 2 → Phase 3.

    Args:
        file_path:   Local path to video file.
        db:          SQLAlchemy session.
        title:       Optional title. Defaults to filename.
        run_phases:  Which phases to run. Default (1, 2, 3) = all.

    Returns dict with video_id + per-phase timing + final QC report.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("INGEST START: %s", path.name)
    log.info("=" * 60)

    # ── Register video in DB + upload to MinIO ────────────────────────────────
    video_id = _register_video(path, db, title)
    timing: dict[str, float] = {}

    # ── Phase 1: Raw → Clean aligned transcript ───────────────────────────────
    if 1 in run_phases:
        t1 = time.perf_counter()
        log.info("[Phase 1] Starting...")
        run_phase1(video_id, db)
        timing["phase1_s"] = round(time.perf_counter() - t1, 1)
        log.info("[Phase 1] Done in %.1fs", timing["phase1_s"])

    # ── Phase 2: Chunking + Quality Check ────────────────────────────────────
    if 2 in run_phases:
        t2 = time.perf_counter()
        log.info("[Phase 2] Starting...")
        p2_result = run_phase2(video_id, db)
        timing["phase2_s"] = round(time.perf_counter() - t2, 1)
        log.info("[Phase 2] Done in %.1fs", timing["phase2_s"])
    else:
        p2_result = {}

    # ── Phase 3: Embedding + pgvector ────────────────────────────────────────
    if 3 in run_phases:
        t3 = time.perf_counter()
        log.info("[Phase 3] Starting...")
        p3_result = run_phase3(video_id, db)
        timing["phase3_s"] = round(time.perf_counter() - t3, 1)
        log.info("[Phase 3] Done in %.1fs", timing["phase3_s"])
    else:
        p3_result = {}

    total_s = round(time.perf_counter() - t0, 1)
    timing["total_s"] = total_s

    log.info("=" * 60)
    log.info("INGEST COMPLETE: %s in %.1fs", path.name, total_s)
    log.info("=" * 60)

    return {
        "video_id": str(video_id),
        "filename": path.name,
        "timing": timing,
        "phase2": p2_result,
        "phase3": p3_result,
    }


def _register_video(path: Path, db: Session, title: str | None) -> uuid.UUID:
    """Upload file to MinIO, create Video + VideoAsset records."""
    ensure_buckets()

    video_id = uuid.uuid4()
    mime = _guess_mime(path.suffix)
    object_key = f"{video_id}/raw{path.suffix}"

    log.info("[Register] Uploading %s to MinIO...", path.name)
    upload_file(settings.minio_bucket_videos, object_key, str(path), mime)

    video = Video(
        id=video_id,
        original_filename=path.name,
        title=title or path.stem,
        status="uploaded",
    )
    asset = VideoAsset(
        video_id=video_id,
        asset_type="raw_video",
        minio_bucket=settings.minio_bucket_videos,
        object_key=object_key,
        mime_type=mime,
        size_bytes=path.stat().st_size,
    )
    db.add(video)
    db.add(asset)
    db.commit()
    log.info("[Register] video_id=%s", video_id)
    return video_id


def _guess_mime(suffix: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(suffix.lower(), "video/octet-stream")
