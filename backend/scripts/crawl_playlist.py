"""
Crawl an entire YouTube playlist and run the full pipeline for EACH video,
sequentially, with live per-step logs.

By default it SKIPS videos that are already fully processed (status='ready'),
so you can re-run the same playlist to pick up only new/incomplete videos.

Usage:
    python scripts/crawl_playlist.py "<playlist_url>"
    python scripts/crawl_playlist.py "<playlist_url>" --limit 5      # only first 5
    python scripts/crawl_playlist.py "<playlist_url>" --reprocess    # re-run even ready ones

Example (Kháng Thương playlist):
    python scripts/crawl_playlist.py \
      "https://www.youtube.com/playlist?list=PLWrhnsc6Cvco81VnlZJFOrmiEHrD9Phl-"

Tip: point at the right DB explicitly if needed:
    DATABASE_URL=postgresql://user:pass@host:5433/vietsuccess python scripts/crawl_playlist.py "<url>"
"""
import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # allow importing the sibling crawl_youtube.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.config import settings
from src.database import SessionLocal
from src.models.video import Video
from src.pipeline.phase1 import run_phase1
from src.pipeline.phase2 import run_phase2
from src.pipeline.phase3 import run_phase3
from src.pipeline.phase4 import run_phase4
from src.pipeline.youtube_crawl import (
    YouTubeCrawlError,
    extract_playlist_entries,
    extract_video_id,
)
from crawl_youtube import crawl_and_register  # reuse single-video download + register

log = logging.getLogger("playlist")


def process_one(db, url: str, reprocess: bool) -> str:
    """Returns 'skipped' | 'done' | 'failed'."""
    youtube_id = extract_video_id(url)
    existing = (
        db.query(Video).filter(Video.youtube_video_id == youtube_id)
        .order_by(Video.created_at.asc()).first()
    )

    if existing and existing.status == "ready" and not reprocess:
        log.info("  ↷ already ready (%s) — skipping", existing.id)
        return "skipped"

    if existing:
        video_id = existing.id
        log.info("  ↻ exists (status=%s) — re-running pipeline on %s", existing.status, video_id)
    else:
        video_id = crawl_and_register(db, url)

    run_phase1(video_id, db)
    run_phase2(video_id, db)
    run_phase3(video_id, db)
    run_phase4(video_id, db)

    video = db.get(Video, video_id)
    log.info("  ✓ done — status=%s video_id=%s", video.status, video_id)
    return "done"


def main():
    parser = argparse.ArgumentParser(description="Crawl a whole YouTube playlist + run pipeline per video")
    parser.add_argument("playlist_url", help="YouTube playlist URL (or any video URL with ?list=...)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos")
    parser.add_argument("--reprocess", action="store_true", help="Re-run pipeline even on already-ready videos")
    args = parser.parse_args()

    print("=" * 70)
    print("DB:", settings.database_url.split("@")[-1])
    print("=" * 70)

    try:
        pl = extract_playlist_entries(args.playlist_url)
    except YouTubeCrawlError as exc:
        log.error("%s", exc)
        sys.exit(1)

    videos = pl["videos"]
    if args.limit:
        videos = videos[: args.limit]

    log.info("Playlist: %r — %d videos to process", pl["title"], len(videos))

    stats = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.perf_counter()

    for i, v in enumerate(videos, 1):
        db = SessionLocal()
        log.info("=" * 70)
        log.info("[%d/%d] %s  (%s)", i, len(videos), (v.get("title") or "")[:60], v["id"])
        log.info("=" * 70)
        try:
            result = process_one(db, v["url"], args.reprocess)
            stats[result] += 1
        except Exception as exc:  # one bad video must not stop the whole playlist
            log.exception("  ✗ failed: %s", exc)
            stats["failed"] += 1
        finally:
            db.close()

    print("\n" + "=" * 70)
    print(f"PLAYLIST DONE in {time.perf_counter() - t0:.0f}s — "
          f"done={stats['done']} skipped={stats['skipped']} failed={stats['failed']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
