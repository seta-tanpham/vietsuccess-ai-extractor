"""
YouTube caption fetch (youtube-transcript-api) — PRIMARY transcript source.

Used before Whisper STT to avoid paying STT API tokens when YouTube already has
a caption track (manual or auto-generated). No OAuth / API key required.

Auto-captions arrive as many short, time-overlapping cues; we group them into
~`caption_segment_target_s`-second segments so the downstream GPT diarization
and chunking see Whisper-like segment granularity.

Returns segments shaped like Whisper output: [{"start": s, "end": s, "text": str}]
(seconds), so Phase 1 step 3+ can consume them unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.config import settings

log = logging.getLogger(__name__)


def _languages() -> list[str]:
    return [code.strip() for code in settings.caption_languages.split(",") if code.strip()]


def fetch_caption_segments(
    youtube_video_id: str,
    languages: Optional[list[str]] = None,
    target_segment_s: Optional[float] = None,
) -> Optional[dict]:
    """
    Fetch + group a YouTube caption track.

    Returns {"segments": [...], "language": str, "is_generated": bool} or None
    if no usable caption exists (caller should fall back to Whisper STT).
    """
    langs = languages or _languages()
    target = target_segment_s if target_segment_s is not None else settings.caption_segment_target_s

    cues, language, is_generated = _fetch_raw(youtube_video_id, langs)
    if not cues:
        return None

    segments = _group_cues(cues, target)
    if not segments:
        return None

    return {"segments": segments, "language": language, "is_generated": is_generated}


def _fetch_raw(video_id: str, languages: list[str]):
    """Return (cues, language_code, is_generated) or (None, None, None).

    cues: list of {"text", "start", "duration"} in seconds.
    Prefers a manually-created track, then an auto-generated one.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        log.warning("youtube-transcript-api not installed — cannot use caption source")
        return None, None, None

    try:
        api = YouTubeTranscriptApi()
        tlist = api.list(video_id)

        transcript = None
        try:
            transcript = tlist.find_manually_created_transcript(languages)
        except Exception:
            try:
                transcript = tlist.find_generated_transcript(languages)
            except Exception:
                transcript = None
        if transcript is None:
            log.info("[caption] no track in %s for %s", languages, video_id)
            return None, None, None

        fetched = transcript.fetch()
        cues = [
            {"text": s.text, "start": float(s.start), "duration": float(s.duration)}
            for s in fetched.snippets
        ]
        return cues, transcript.language_code, transcript.is_generated

    except Exception as exc:
        # TranscriptsDisabled, NoTranscriptFound, network errors, etc. — non-fatal.
        log.info("[caption] no usable caption for %s: %s", video_id, exc)
        return None, None, None


def _group_cues(cues: list[dict], target_s: float) -> list[dict]:
    """Merge short, overlapping caption cues into ~target_s-second segments."""
    segments: list[dict] = []
    buf: list[str] = []
    buf_start: Optional[float] = None
    buf_end: Optional[float] = None

    for cue in cues:
        text = (cue.get("text") or "").replace("\n", " ").strip()
        # Skip pure sound tags like [Music], [Applause]
        if not text or (text.startswith("[") and text.endswith("]")):
            continue

        start = cue["start"]
        end = cue["start"] + cue["duration"]
        if buf_start is None:
            buf_start = start
        buf_end = max(buf_end if buf_end is not None else end, end)
        buf.append(text)

        if buf_end - buf_start >= target_s:
            segments.append({"start": buf_start, "end": buf_end, "text": " ".join(buf).strip()})
            buf, buf_start, buf_end = [], None, None

    if buf and buf_start is not None:
        segments.append({"start": buf_start, "end": buf_end, "text": " ".join(buf).strip()})

    return segments
