"""
Quick smoke-test for Phase 1.
Usage:
    python scripts/test_phase1.py <path-to-video-file>
    python scripts/test_phase1.py video/yourfile.webm
"""
import sys
import time
import uuid
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import SessionLocal
from src.models.video import Video, VideoAsset
from src.pipeline.phase1 import run_phase1
from src.pipeline.video_deduplication import create_duplicate_video_record, find_processed_duplicate, sha256_file
from src.storage.minio_client import ensure_buckets, upload_file
from src.config import settings


def resolve_video_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    # Common shell typo: /video/foo.webm instead of video/foo.webm
    if raw_path.startswith("/video/"):
        project_path = PROJECT_ROOT / raw_path.lstrip("/")
        if project_path.exists():
            return project_path

    return path


def main():
    raw_video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not raw_video_path:
        print("Usage: python scripts/test_phase1.py <path-to-video>")
        sys.exit(1)

    video_path = resolve_video_path(raw_video_path)
    if not video_path.exists():
        print(f"Video file not found: {raw_video_path}")
        print("Usage: python scripts/test_phase1.py <path-to-video>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"VietSuccess Phase 1 Test")
    print(f"File: {video_path.name}")
    print(f"{'='*60}\n")

    # 1. Validate duplicate by content hash before touching MinIO
    db = SessionLocal()
    video_hash = sha256_file(video_path)
    duplicate = find_processed_duplicate(db, video_hash)
    video_id = uuid.uuid4()

    if duplicate:
        print("[1/5] Duplicate video detected by SHA-256.")
        create_duplicate_video_record(
            db,
            video_id=video_id,
            duplicate=duplicate,
            video_sha256=video_hash,
            original_filename=video_path.name,
        )
        print(f"      Duplicate of: {duplicate.id}")
        print(f"      Video ID    : {video_id}")
        print("      Copied transcript/speakers and skipped upload + Phase 1 processing.")
        db.close()
        return

    # 2. Setup MinIO buckets
    print("[2/5] Ensuring MinIO buckets exist...")
    ensure_buckets()
    print("      OK")

    # 3. Upload raw video to MinIO
    object_key = f"{video_id}/raw{video_path.suffix}"

    print(f"      SHA-256: {video_hash}")
    print(f"[3/5] Uploading {video_path.name} to MinIO...")
    upload_file(settings.minio_bucket_videos, object_key, str(video_path), "video/webm")
    print(f"      Uploaded → {object_key}")

    # 4. Create DB records
    print("[4/5] Creating DB records...")
    video = Video(id=video_id, video_sha256=video_hash, original_filename=video_path.name, status="uploaded")
    asset = VideoAsset(
        video_id=video_id,
        asset_type="raw_video",
        minio_bucket=settings.minio_bucket_videos,
        object_key=object_key,
        mime_type="video/webm",
        size_bytes=video_path.stat().st_size,
    )
    db.add(video)
    db.add(asset)
    db.commit()
    print(f"      Video ID: {video_id}")

    # 5. Run Phase 1
    print("\n[5/5] Running Phase 1 pipeline (this may take a while)...\n")
    start = time.time()
    run_phase1(video_id, db)
    elapsed = time.time() - start
    print(f"\n[5/5] Phase 1 done in {elapsed:.1f}s")

    # 6. Print results
    print("\n[6/6] Results:")
    db.refresh(video)
    print(f"      Status : {video.status}")

    from src.models.transcript import Transcript
    from src.models.speaker import Speaker

    transcript = db.query(Transcript).filter_by(video_id=video_id).first()
    if transcript:
        print(f"      Language   : {transcript.language}")
        print(f"      Confidence : {transcript.confidence:.2f}" if transcript.confidence else "      Confidence : N/A")
        turns = transcript.aligned_transcript or []
        print(f"      Aligned turns: {len(turns)}")
        print("\n  First 3 turns:")
        for t in turns[:3]:
            print(f"    [{t['speaker']}] {t['start_ms']}ms–{t['end_ms']}ms: {t['text'][:80]}")

    speakers = db.query(Speaker).filter_by(video_id=video_id).all()
    print(f"\n  Speakers detected: {[s.diarization_label for s in speakers]}")

    db.close()
    print(f"\n{'='*60}")
    print("Phase 1 test complete!")
    print(f"Run the API server: uvicorn src.api.main:app --reload")
    print(f"Then GET http://localhost:8000/videos/{video_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
