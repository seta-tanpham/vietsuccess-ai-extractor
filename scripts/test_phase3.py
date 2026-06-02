"""
Phase 3 test: embed + index existing video.

Usage:
    python scripts/test_phase3.py                  # list videos
    python scripts/test_phase3.py <video_id>        # run Phase 3
    python scripts/test_phase3.py <video_id> --re   # re-embed (delete old + re-run)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SessionLocal
from src.models.video import Video
from src.models.chunk import Chunk
from src.models.embedding import ChunkEmbedding


def list_videos(db):
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    if not videos:
        print("No videos yet. Run run_pipeline.py first.")
        return
    print(f"\n{'ID':<38}  {'Status':<25}  Filename")
    print("-" * 85)
    for v in videos:
        # Count embedded chunks
        q = (
            db.query(Chunk)
            .filter_by(video_id=v.id, chunk_type="semantic", is_low_quality=False)
            .count()
        )
        embedded = (
            db.query(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .filter(Chunk.video_id == v.id)
            .count()
        )
        print(f"{str(v.id):<38}  {v.status:<25}  {v.original_filename[:30]}  [{embedded}/{q} embedded]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", nargs="?")
    parser.add_argument("--re", action="store_true", help="Re-embed (delete old embeddings first)")
    args = parser.parse_args()

    db = SessionLocal()

    if not args.video_id:
        list_videos(db)
        print("\nUsage: python scripts/test_phase3.py <video_id>")
        db.close()
        return

    import uuid, time
    video_id = uuid.UUID(args.video_id)
    video = db.get(Video, video_id)
    if not video:
        print(f"Video {video_id} not found")
        db.close()
        return

    print(f"\n{'='*60}")
    print(f"Phase 3 — Embedding")
    print(f"Video : {video.original_filename[:50]}")
    print(f"Status: {video.status}")
    print(f"{'='*60}\n")

    if video.status not in ("ready_for_embedding", "quality_check", "embedding", "ready", "error"):
        print(f"Phase 2 not complete (status={video.status}). Run Phase 2 first.")
        db.close()
        return

    t = time.perf_counter()
    if args.re:
        from src.pipeline.phase3 import re_embed_video
        result = re_embed_video(video_id, db)
    else:
        from src.pipeline.phase3 import run_phase3
        result = run_phase3(video_id, db)

    elapsed = time.perf_counter() - t

    print(f"\nResult:")
    print(f"  Eligible  : {result.get('eligible', '-')}")
    print(f"  Embedded  : {result.get('embedded_count', '-')}")
    print(f"  Skipped   : {result.get('skipped', '-')}  (already done)")
    print(f"  Backend   : {result.get('backend', '-')}")
    if result.get("total_tokens"):
        print(f"  Tokens    : {result['total_tokens']:,}")
        print(f"  Cost      : ${result.get('estimated_cost_usd', 0):.4f}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"\nSearch: GET http://localhost:8000/search?q=your+query&video_id={video_id}")
    db.close()


if __name__ == "__main__":
    main()
