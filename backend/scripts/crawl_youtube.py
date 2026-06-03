"""
Crawl a YouTube video and run the FULL pipeline in the FOREGROUND, printing
every step's log so you can watch it run live.

Usage:
    python scripts/crawl_youtube.py "<youtube_url>"
    python scripts/crawl_youtube.py "<youtube_url>" --fresh     # ignore dedup, crawl a new copy

If the video was already crawled, by default it re-runs the pipeline phases on the
existing record (so you still see the per-step logs) instead of downloading again.

Tip: point at the right DB explicitly, e.g.
    DATABASE_URL=postgresql://nguyenthituyetmay:iloveyou044@localhost:5433/vietsuccess \
        python scripts/crawl_youtube.py "<url>"
"""
import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# INFO logs to stdout so every pipeline step is visible; silence noisy HTTP libs.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.config import settings
from src.database import SessionLocal
from src.models.video import Video, VideoAsset
from src.pipeline.phase1 import run_phase1
from src.pipeline.phase2 import run_phase2
from src.pipeline.phase3 import run_phase3
from src.pipeline.phase4 import run_phase4
from src.pipeline.youtube_crawl import (
    YouTubeCrawlError,
    canonicalize_url,
    download_media,
    extract_video_id,
    map_metadata,
)
# Reuse the same helpers the /videos/crawl route uses.
from src.api.routes.videos import (
    _get_or_create_channel,
    _maybe_fetch_thumbnail,
    _upload_raw_metadata,
)
from src.storage.minio_client import ensure_buckets, upload_file

log = logging.getLogger("crawl")


def crawl_and_register(db, url: str) -> uuid.UUID:
    """Download via yt-dlp, upload to MinIO, create Video + assets. Returns video_id."""
    from datetime import datetime, timezone
    import os

    canonical = canonicalize_url(url)
    youtube_id = extract_video_id(url)
    log.info("[crawl] youtube_id=%s canonical=%s", youtube_id, canonical)

    ensure_buckets()
    log.info("[crawl] downloading media via yt-dlp ...")
    local_path, info = download_media(canonical)
    meta = map_metadata(info)
    log.info("[crawl] metadata: title=%r channel=%r duration=%ss",
             meta.get("title"), (meta.get("channel") or {}).get("title"),
             round((meta.get("duration_ms") or 0) / 1000))

    video_id = uuid.uuid4()
    object_key = f"{video_id}/raw.mp4"
    log.info("[crawl] uploading raw video to MinIO → %s", object_key)
    upload_file(settings.minio_bucket_videos, object_key, str(local_path), "video/mp4")
    raw_metadata_key = _upload_raw_metadata(video_id, info)
    log.info("[crawl] saved raw metadata json → %s", raw_metadata_key)
    thumbnail_key = _maybe_fetch_thumbnail(video_id, meta.get("thumbnail_url"))
    size_bytes = local_path.stat().st_size
    try:
        os.unlink(local_path)
    except OSError:
        pass

    channel = _get_or_create_channel(db, meta.get("channel") or {})
    log.info("[crawl] channel: %s", channel.title if channel else None)

    video = Video(
        id=video_id,
        source_type="youtube",
        youtube_video_id=youtube_id,
        youtube_url=url,
        canonical_url=canonical,
        crawl_status="completed",
        raw_metadata_key=raw_metadata_key,
        title=meta.get("title"),
        description=meta.get("description"),
        duration_ms=meta.get("duration_ms"),
        published_at=meta.get("published_at"),
        channel_id=channel.id if channel else None,
        tags=meta.get("tags"),
        youtube_category_id=meta.get("youtube_category_id"),
        default_language=meta.get("default_language"),
        language=meta.get("default_language"),
        view_count=meta.get("view_count"),
        like_count=meta.get("like_count"),
        comment_count=meta.get("comment_count"),
        statistics_crawled_at=datetime.now(timezone.utc),
        status="uploaded",
    )
    db.add(video)
    db.add(VideoAsset(
        video_id=video_id, asset_type="raw_video",
        minio_bucket=settings.minio_bucket_videos, object_key=object_key,
        mime_type="video/mp4", size_bytes=size_bytes,
    ))
    if thumbnail_key:
        db.add(VideoAsset(
            video_id=video_id, asset_type="thumbnail",
            minio_bucket=settings.minio_bucket_videos, object_key=thumbnail_key,
            mime_type="image/jpeg",
        ))
    db.commit()
    log.info("[crawl] created Video record video_id=%s", video_id)
    return video_id


def main():
    parser = argparse.ArgumentParser(description="Crawl a YouTube video + run pipeline with live logs")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--fresh", action="store_true", help="Ignore dedup; crawl a new copy")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-run the pipeline even if the video is already 'ready'")
    args = parser.parse_args()

    print("=" * 70)
    print("DB:", settings.database_url.split("@")[-1])
    print("=" * 70)

    db = SessionLocal()
    try:
        youtube_id = extract_video_id(args.url)
        existing = (
            db.query(Video).filter(Video.youtube_video_id == youtube_id)
            .order_by(Video.created_at.asc()).first()
        )

        if existing and not args.fresh:
            # Already crawled: skip processing if it's already done, unless --reprocess.
            if existing.status == "ready" and not args.reprocess:
                log.info("[crawl] already crawled and ready (%s) — skipping. Use --reprocess to force.",
                         existing.id)
                print(f"\nSKIPPED (already ready): video_id = {existing.id}")
                return
            video_id = existing.id
            log.info("[crawl] exists (status=%s) → re-running pipeline on %s", existing.status, video_id)
        else:
            video_id = crawl_and_register(db, args.url)

        t0 = time.perf_counter()
        log.info("==================== PHASE 1: transcript + speakers ====================")
        run_phase1(video_id, db)
        log.info("==================== PHASE 2: chunking + QC ====================")
        run_phase2(video_id, db)
        log.info("==================== PHASE 3: embedding ====================")
        run_phase3(video_id, db)
        log.info("==================== PHASE 4: summaries (topic + video) → RAG ====================")
        run_phase4(video_id, db)

        video = db.get(Video, video_id)
        print("\n" + "=" * 70)
        print(f"DONE in {time.perf_counter() - t0:.1f}s — status = {video.status}")
        print(f"video_id = {video_id}")
        print("=" * 70)
    except YouTubeCrawlError as exc:
        log.error("Crawl failed: %s", exc)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
