"""
Phase 1 orchestrator: Raw video → Clean aligned transcript.

Steps:
  1. Audio normalization (ffmpeg)
  2. Raw transcript (Whisper)
  3. Filler word cleaning
  4. Speaker diarization (pyannote) — skipped if no token
  5. Speaker turn merging
  6. Speaker records creation
  7. Status state machine updates
  8. Idempotent: any step can be re-run safely
"""
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from src.config import settings
from src.models.transcript import Transcript
from src.models.video import Video, VideoAsset
from src.models.speaker import Speaker
from src.pipeline.audio_normalization import normalize_audio
from src.pipeline.diarization import align_transcript_with_diarization, diarize
from src.pipeline.filler_cleaning import clean_text, compression_ratio
from src.pipeline.speaker_merging import merge_speaker_turns
from src.pipeline.transcription import transcribe
from src.storage.minio_client import get_client, upload_file

log = logging.getLogger(__name__)


def _set_status(db: Session, video: Video, status: str) -> None:
    video.status = status
    db.commit()
    log.info("[%s] status → %s", video.id, status)


def run_phase1(video_id: uuid.UUID, db: Session) -> None:
    """
    Run all Phase 1 steps for a video. Idempotent — skips already-done steps.
    Raises on unrecoverable error and sets video.status = 'error'.
    """
    video = db.get(Video, video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    try:
        _step1_normalize(video, db)
        _step2_transcribe(video, db)
        _step3_diarize_and_merge(video, db)
        _step4_create_speakers(video, db)

        _set_status(db, video, "ready_for_chunking")
        log.info("[%s] Phase 1 complete ✓", video.id)

    except Exception as exc:
        log.exception("[%s] Phase 1 failed: %s", video.id, exc)
        _set_status(db, video, "error")
        raise


# ── Step 1: Audio normalization ───────────────────────────────────────────────

def _step1_normalize(video: Video, db: Session) -> str:
    # Idempotent: skip if audio_wav asset already exists
    existing = _find_asset(video, "audio_wav")
    if existing:
        log.info("[%s] audio_wav already exists, skipping normalization", video.id)
        return existing.object_key

    _set_status(db, video, "normalizing")

    raw_asset = _find_asset(video, "raw_video")
    if not raw_asset:
        raise RuntimeError("No raw_video asset found — upload first")

    # Download raw video from MinIO to temp file
    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "input" + Path(raw_asset.object_key).suffix)
        wav_path = os.path.join(tmp, "audio.wav")

        get_client().fget_object(raw_asset.minio_bucket, raw_asset.object_key, input_path)
        normalize_audio(input_path, wav_path)

        object_key = f"{video.id}/audio.wav"
        upload_file(settings.minio_bucket_videos, object_key, wav_path, "audio/wav")

        asset = VideoAsset(
            video_id=video.id,
            asset_type="audio_wav",
            minio_bucket=settings.minio_bucket_videos,
            object_key=object_key,
            mime_type="audio/wav",
            size_bytes=Path(wav_path).stat().st_size,
        )
        db.add(asset)
        db.commit()
        log.info("[%s] audio normalized → %s", video.id, object_key)
        return object_key


# ── Step 2: Whisper transcription ────────────────────────────────────────────

def _step2_transcribe(video: Video, db: Session) -> Transcript:
    # Idempotent: skip if transcript already exists
    existing = db.query(Transcript).filter_by(video_id=video.id).first()
    if existing and existing.raw_json:
        log.info("[%s] transcript already exists, skipping", video.id)
        return existing

    _set_status(db, video, "transcribing")

    wav_asset = _find_asset(video, "audio_wav")
    if not wav_asset:
        raise RuntimeError("No audio_wav asset — run normalization first")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        get_client().fget_object(wav_asset.minio_bucket, wav_asset.object_key, wav_path)

        result = transcribe(wav_path)

    # Clean search_text for each segment
    segments = result["segments"]
    for seg in segments:
        seg["search_text"] = clean_text(seg.get("text", ""))

    transcript = existing or Transcript(video_id=video.id, provider=settings.whisper_model)
    transcript.raw_json = {"segments": segments, "language": result.get("language")}
    transcript.confidence = result.get("confidence")
    transcript.model_version = settings.whisper_model
    transcript.language = result.get("language")

    if not existing:
        db.add(transcript)
    db.commit()
    log.info("[%s] transcribed %d segments, confidence=%.2f", video.id, len(segments), transcript.confidence or 0)
    return transcript


# ── Step 3: Diarization + speaker turn merging ───────────────────────────────

def _step3_diarize_and_merge(video: Video, db: Session) -> None:
    transcript = db.query(Transcript).filter_by(video_id=video.id).first()
    if not transcript or not transcript.raw_json:
        raise RuntimeError("No transcript found — run transcription first")

    if transcript.aligned_transcript:
        log.info("[%s] aligned_transcript already exists, skipping diarization", video.id)
        return

    _set_status(db, video, "diarizing")

    segments = transcript.raw_json.get("segments", [])

    if settings.pyannote_auth_token:
        wav_asset = _find_asset(video, "audio_wav")
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "audio.wav")
            get_client().fget_object(wav_asset.minio_bucket, wav_asset.object_key, wav_path)
            raw_turns = diarize(wav_path)

        turns = align_transcript_with_diarization(segments, raw_turns)
        log.info("[%s] diarized %d raw turns", video.id, len(raw_turns))
    else:
        # No pyannote token — assign all to SPEAKER_00, still useful for transcript
        log.warning("[%s] No PYANNOTE_AUTH_TOKEN — assigning all segments to SPEAKER_00", video.id)
        turns = [
            {
                "speaker": "SPEAKER_00",
                "start_ms": int(seg["start"] * 1000),
                "end_ms": int(seg["end"] * 1000),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
        ]

    before_count = len(turns)
    merged = merge_speaker_turns(turns)
    log.info("[%s] turns merged: %d → %d", video.id, before_count, len(merged))

    transcript.aligned_transcript = merged
    db.commit()


# ── Step 4: Create speaker records ───────────────────────────────────────────

def _step4_create_speakers(video: Video, db: Session) -> None:
    transcript = db.query(Transcript).filter_by(video_id=video.id).first()
    if not transcript or not transcript.aligned_transcript:
        return

    existing_labels = {s.diarization_label for s in db.query(Speaker).filter_by(video_id=video.id).all()}

    new_labels = {t["speaker"] for t in transcript.aligned_transcript} - existing_labels
    for label in sorted(new_labels):
        db.add(Speaker(video_id=video.id, diarization_label=label, mapping_status="auto"))

    if new_labels:
        db.commit()
        log.info("[%s] created %d speaker records: %s", video.id, len(new_labels), sorted(new_labels))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_asset(video: Video, asset_type: str) -> Optional[VideoAsset]:
    for asset in video.assets:
        if asset.asset_type == asset_type:
            return asset
    return None
