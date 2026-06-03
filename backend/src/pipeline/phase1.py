"""
Phase 1 orchestrator: Raw video → Clean aligned transcript.

Steps:
  1. Audio normalization        ffmpeg → WAV 16kHz mono
  2. Whisper transcription      text + word-level timestamps
  3. Filler word cleaning       regex on search_text
  4. Speaker diarization        pyannote 3.1 — REQUIRES PYANNOTE_AUTH_TOKEN
  5. Speaker turn merging       gap < 1.5s same speaker → merge
  6. Speaker records creation   one DB row per unique speaker label
  7. LLM speaker naming         GPT suggests display_name + role (status="suggested")
  8. Status state machine       track progress at each step
  9. Idempotent                 any step restarts safely without re-running earlier steps

Diarization setup (one-time, free):
  1. Accept: https://hf.co/pyannote/speaker-diarization-3.1
  2. Accept: https://hf.co/pyannote/segmentation-3.0
  3. Token:  https://hf.co/settings/tokens  (read access)
  4. .env:   PYANNOTE_AUTH_TOKEN=hf_xxxx
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
from src.pipeline.filler_cleaning import clean_text
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
        llm_duplicates = _step5_llm_speaker_naming(video, db)
        _step6_deduplicate_speakers(video, db, llm_duplicates=llm_duplicates)

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
        num_spk = settings.pyannote_num_speakers
        min_spk = settings.pyannote_min_speakers
        max_spk = settings.pyannote_max_speakers

        if not num_spk and not (min_spk or max_spk) and settings.pyannote_use_title_speaker_hint:
            from src.pipeline.llm_speaker_naming import detect_show_format
            title = video.title or video.original_filename or ""
            min_spk, max_spk, fmt = detect_show_format(title)
            log.info("[%s] Title hint detected: '%s' → expected %d–%d speakers", video.id, fmt, min_spk, max_spk)
        elif num_spk or min_spk or max_spk:
            log.info("[%s] Speaker count override: num=%s min=%s max=%s", video.id, num_spk, min_spk, max_spk)
        else:
            log.info("[%s] Speaker count: auto-detect", video.id)

        wav_asset = _find_asset(video, "audio_wav")
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "audio.wav")
            get_client().fget_object(wav_asset.minio_bucket, wav_asset.object_key, wav_path)
            raw_turns = diarize(
                wav_path,
                num_speakers=num_spk,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )

        turns = align_transcript_with_diarization(segments, raw_turns)
        log.info("[%s] diarized %d raw turns, %d unique speakers",
                 video.id, len(raw_turns), len({t["speaker"] for t in raw_turns}))
    else:
        log.warning(
            "[%s] PYANNOTE_AUTH_TOKEN not set — all segments assigned to SPEAKER_00. "
            "To get real speaker detection:\n"
            "  1. Accept: https://hf.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept: https://hf.co/pyannote/segmentation-3.0\n"
            "  3. Token:  https://hf.co/settings/tokens\n"
            "  4. .env:   PYANNOTE_AUTH_TOKEN=hf_xxxx\n"
            "  5. Re-run: python scripts/run_pipeline.py --id %s --phases 1",
            video.id, video.id,
        )
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


# ── Step 5: LLM speaker naming ───────────────────────────────────────────────

def _step5_llm_speaker_naming(video: Video, db: Session) -> list[dict]:
    """
    Use GPT to suggest display_name + role for each speaker.
    Non-blocking — failure returns [] and does NOT fail Phase 1.
    Sets mapping_status = "suggested".
    Returns potential_duplicates list for Step 6.
    """
    from src.pipeline.llm_speaker_naming import (
        apply_suggestions_to_db,
        extract_first_turns,
        suggest_speaker_names,
    )

    transcript = db.query(Transcript).filter_by(video_id=video.id).first()
    if not transcript or not transcript.aligned_transcript:
        return []

    from src.models.speaker import Speaker as SpeakerModel
    unnamed = db.query(SpeakerModel).filter_by(video_id=video.id, mapping_status="auto").count()
    if unnamed == 0:
        log.info("[%s] All speakers already named, skipping LLM naming", video.id)
        return []

    try:
        speaker_turns = extract_first_turns(transcript.aligned_transcript)
        suggestions = suggest_speaker_names(
            speaker_turns=speaker_turns,
            video_title=video.title or video.original_filename,
        )
        title = video.title or video.original_filename or ""
        if suggestions:
            potential_duplicates = apply_suggestions_to_db(
                suggestions, video.id, db, video_title=title
            )
            log.info("[%s] LLM named %d speakers, %d duplicate candidates",
                     video.id, len(suggestions), len(potential_duplicates))
            return potential_duplicates
        # No LLM suggestions at all — still try fallback from title
        apply_suggestions_to_db([], video.id, db, video_title=title)
        log.info("[%s] LLM returned no suggestions — applied title fallback", video.id)
    except Exception as exc:
        log.warning("[%s] LLM speaker naming skipped: %s", video.id, exc)
    return []


# ── Step 6: Duplicate speaker detection ──────────────────────────────────────

def _step6_deduplicate_speakers(
    video: Video, db: Session, llm_duplicates: Optional[list] = None
) -> None:
    """
    Detect and merge duplicate speaker labels.

    Method A (always): if LLM gave 2 labels the same name → merge.
    Method B (if audio available): voice embedding cosine similarity > 0.85 → merge.

    Non-blocking — failure does NOT fail Phase 1.
    """
    from src.models.speaker import Speaker as SpeakerModel
    from src.pipeline.speaker_deduplication import deduplicate_speakers

    transcript = db.query(Transcript).filter_by(video_id=video.id).first()
    if not transcript or not transcript.aligned_transcript:
        return

    speakers = db.query(SpeakerModel).filter_by(video_id=video.id).all()
    if len(speakers) < 2:
        return  # Only 1 speaker → nothing to deduplicate

    # Try to get WAV path for Method B (voice embedding)
    wav_path: Optional[str] = None
    wav_asset = _find_asset(video, "audio_wav")
    _tmp_dir = None
    try:
        if wav_asset:
            import tempfile
            import os
            _tmp_dir = tempfile.mkdtemp()
            wav_path = os.path.join(_tmp_dir, "audio.wav")
            get_client().fget_object(wav_asset.minio_bucket, wav_asset.object_key, wav_path)
    except Exception:
        wav_path = None  # Method B disabled gracefully

    try:
        updated_transcript, merge_map = deduplicate_speakers(
            aligned_transcript=transcript.aligned_transcript,
            speakers=speakers,
            wav_path=wav_path,
            llm_duplicates=llm_duplicates or [],
            video_id=video.id,
            db=db,
        )

        if merge_map:
            transcript.aligned_transcript = updated_transcript
            db.commit()
            log.info("[%s] Deduplication merged %d speaker label(s): %s", video.id, len(merge_map), merge_map)
        else:
            log.info("[%s] No duplicate speakers found", video.id)

    except Exception as exc:
        log.warning("[%s] Speaker deduplication skipped (non-fatal): %s", video.id, exc)
    finally:
        if _tmp_dir:
            import shutil
            shutil.rmtree(_tmp_dir, ignore_errors=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_asset(video: Video, asset_type: str) -> Optional[VideoAsset]:
    for asset in video.assets:
        if asset.asset_type == asset_type:
            return asset
    return None
