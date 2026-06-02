"""
VietSuccess pipeline CLI.

INGEST a new video (all 3 phases):
    python scripts/run_pipeline.py video/myvideo.webm

Run specific phases on NEW video:
    python scripts/run_pipeline.py video/myvideo.webm --phases 1 2

Run specific phases on EXISTING video (by video_id):
    python scripts/run_pipeline.py --id <video_id> --phases 3
    python scripts/run_pipeline.py --id <video_id> --phases 2 3

List all processed videos:
    python scripts/run_pipeline.py --list
"""
import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.database import SessionLocal
from src.models.video import Video


def cmd_list(db):
    from src.models.chunk import Chunk
    from src.models.embedding import ChunkEmbedding

    videos = db.query(Video).order_by(Video.created_at.desc()).limit(20).all()
    if not videos:
        print("No videos yet.")
        return
    print(f"\n{'ID':<38}  {'Status':<22}  Filename")
    print("-" * 85)
    for v in videos:
        eligible = db.query(Chunk).filter_by(video_id=v.id, chunk_type="semantic", is_low_quality=False).count()
        embedded = (
            db.query(ChunkEmbedding)
            .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
            .filter(Chunk.video_id == v.id)
            .count()
        )
        tag = f"[{embedded}/{eligible} embedded]" if eligible else ""
        print(f"{str(v.id):<38}  {v.status:<22}  {v.original_filename[:30]} {tag}")


def cmd_ingest(file_path: str, phases: list[int]):
    """Ingest a new video file through the specified phases."""
    from src.pipeline.runner import ingest_video

    db = SessionLocal()
    try:
        result = ingest_video(file_path, db, run_phases=tuple(phases))
        _print_result(result)
    finally:
        db.close()


def cmd_run_phases(video_id_str: str, phases: list[int]):
    """Run specific phases on an already-registered video."""
    db = SessionLocal()
    try:
        vid = uuid.UUID(video_id_str)
        video = db.get(Video, vid)
        if not video:
            print(f"Video {vid} not found. Use --list to see available videos.")
            return

        print(f"\nRunning phases {phases} on: {video.original_filename}")
        print(f"Current status: {video.status}\n")

        import time
        timing = {}
        p2_result = {}
        p3_result = {}

        if 1 in phases:
            from src.pipeline.phase1 import run_phase1
            t = time.perf_counter()
            run_phase1(vid, db)
            timing["phase1_s"] = round(time.perf_counter() - t, 1)

        if 2 in phases:
            from src.pipeline.phase2 import run_phase2
            t = time.perf_counter()
            p2_result = run_phase2(vid, db)
            timing["phase2_s"] = round(time.perf_counter() - t, 1)

        if 3 in phases:
            from src.pipeline.phase3 import run_phase3
            t = time.perf_counter()
            p3_result = run_phase3(vid, db)
            timing["phase3_s"] = round(time.perf_counter() - t, 1)

        timing["total_s"] = sum(timing.values())

        _print_result({
            "video_id": video_id_str,
            "filename": video.original_filename,
            "timing": timing,
            "phase2": p2_result,
            "phase3": p3_result,
        })
    finally:
        db.close()


def _print_result(result: dict):
    t = result["timing"]
    p2 = result.get("phase2", {})
    p3 = result.get("phase3", {})

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  video_id : {result['video_id']}")
    print(f"  file     : {result.get('filename', '')}")
    print(f"\nTiming:")
    if "phase1_s" in t: print(f"  Phase 1  : {t['phase1_s']}s")
    if "phase2_s" in t: print(f"  Phase 2  : {t['phase2_s']}s")
    if "phase3_s" in t: print(f"  Phase 3  : {t['phase3_s']}s")
    print(f"  Total    : {t.get('total_s', 0)}s")

    if p2:
        print(f"\nChunks:")
        print(f"  Atomic   : {p2.get('atomic_count', '-')}")
        print(f"  Semantic : {p2.get('semantic_count', '-')}")
        print(f"  Topics   : {p2.get('topic_count', '-')}")
        print(f"  QC pass  : {p2.get('pass_count', '-')} / {p2.get('total', '-')}")

    if p3:
        print(f"\nEmbedding:")
        print(f"  Embedded : {p3.get('embedded_count', '-')}")
        print(f"  Backend  : {p3.get('backend', '-')}")
        if p3.get("total_tokens"):
            print(f"  Tokens   : {p3['total_tokens']:,}")
            print(f"  Cost     : ${p3.get('estimated_cost_usd', 0):.4f}")

    print(f"\nSearch: GET http://localhost:8000/search?q=your+query&video_id={result['video_id']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="VietSuccess pipeline", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", help="Path to video file (for new ingestion)")
    parser.add_argument("--id", dest="video_id", metavar="VIDEO_ID", help="Existing video_id (skip upload, run phases only)")
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3], metavar="N", help="Phases to run: 1 2 3 (default: all)")
    parser.add_argument("--list", action="store_true", help="List all processed videos")

    args = parser.parse_args()

    if args.list:
        db = SessionLocal()
        cmd_list(db)
        db.close()
        return

    if args.video_id:
        # Run phases on existing video
        cmd_run_phases(args.video_id, args.phases)
        return

    if args.file:
        # Ingest new video
        if not Path(args.file).exists():
            print(f"File not found: {args.file}")
            sys.exit(1)
        cmd_ingest(args.file, args.phases)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
