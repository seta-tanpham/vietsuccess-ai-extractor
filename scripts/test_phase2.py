"""
Smoke-test for Phase 2: Chunking + Quality Check.

Usage:
    python scripts/test_phase2.py <video_id>
    python scripts/test_phase2.py   # lists available video_ids from DB
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SessionLocal
from src.models.video import Video
from src.models.chunk import Chunk
from src.pipeline.phase2 import run_phase2


def list_videos(db):
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    if not videos:
        print("No videos in DB yet. Run test_phase1.py first.")
        return
    print("\nAvailable videos:")
    print(f"{'ID':<38} {'Status':<25} {'Filename'}")
    print("-" * 80)
    for v in videos:
        print(f"{str(v.id):<38} {v.status:<25} {v.original_filename[:30]}")


def main():
    db = SessionLocal()

    if len(sys.argv) < 2:
        list_videos(db)
        print("\nUsage: python scripts/test_phase2.py <video_id>")
        db.close()
        return

    import uuid
    video_id = uuid.UUID(sys.argv[1])
    video = db.get(Video, video_id)
    if not video:
        print(f"Video {video_id} not found")
        db.close()
        return

    print(f"\n{'='*60}")
    print(f"VietSuccess Phase 2 Test")
    print(f"Video: {video.original_filename[:50]}")
    print(f"Status: {video.status}")
    print(f"{'='*60}\n")

    if video.status not in ("ready_for_chunking", "quality_check", "ready_for_embedding", "error"):
        print(f"Video status is '{video.status}'. Phase 1 must be complete first.")
        db.close()
        return

    import time
    start = time.time()
    result = run_phase2(video_id, db)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"Phase 2 complete in {elapsed:.1f}s")
    print(f"  Atomic chunks  : {result['atomic_count']}")
    print(f"  Semantic chunks: {result['semantic_count']}")
    print(f"  Topic segments : {result['topic_count']}")
    print(f"\nQuality Check:")
    print(f"  Total semantic : {result['total']}")
    print(f"  Low quality    : {result['low_quality_count']} ({result['low_quality_pct']}%)")
    print(f"  Pass           : {result['pass_count']}")
    if result['fail_breakdown']:
        print(f"  Fail breakdown : {result['fail_breakdown']}")
    if result['above_threshold']:
        print(f"\n  ⚠ WARNING: > 8% low quality rate — investigate before embedding")

    # Show sample semantic chunks
    print(f"\nSample semantic chunks (first 3):")
    samples = (
        db.query(Chunk)
        .filter_by(video_id=video_id, chunk_type="semantic")
        .order_by(Chunk.start_ms)
        .limit(3)
        .all()
    )
    for c in samples:
        dur_s = (c.end_ms - c.start_ms) / 1000
        status = "FAIL" if c.is_low_quality else "PASS"
        print(f"  [{status}] {c.segment_path} | {dur_s:.1f}s | {(c.original_transcript or '')[:80]}")

    # Show atomic count per semantic
    print(f"\nAtomic chunks per semantic (first 5 semantics):")
    semantics = (
        db.query(Chunk)
        .filter_by(video_id=video_id, chunk_type="semantic")
        .order_by(Chunk.start_ms)
        .limit(5)
        .all()
    )
    for sem in semantics:
        atom_count = db.query(Chunk).filter_by(parent_chunk_id=sem.id, chunk_type="atomic").count()
        print(f"  {sem.segment_path}: {atom_count} atomic chunks, duration={(sem.end_ms-sem.start_ms)//1000}s")

    db.close()
    print(f"\n{'='*60}")
    print("Next: run Phase 3 (embedding)")
    print(f"API: POST http://localhost:8000/chunks/{video_id}/run")
    print(f"QC:  GET  http://localhost:8000/chunks/{video_id}/qc")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
