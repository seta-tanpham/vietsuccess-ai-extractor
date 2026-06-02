"""Phase 1 — Step 4: speaker diarization via pyannote."""
from typing import Any

from src.config import settings


def diarize(wav_path: str) -> list[dict[str, Any]]:
    """
    Run pyannote diarization. Requires PYANNOTE_AUTH_TOKEN.
    Returns raw turns: [{speaker, start_ms, end_ms, text=""}]
    """
    if not settings.pyannote_auth_token:
        raise ValueError("PYANNOTE_AUTH_TOKEN is not set in .env")

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=settings.pyannote_auth_token,
    )
    diarization = pipeline(wav_path)

    turns = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "speaker": speaker,
            "start_ms": int(segment.start * 1000),
            "end_ms": int(segment.end * 1000),
            "text": "",
        })
    return turns


def align_transcript_with_diarization(
    whisper_segments: list[dict],
    diarization_turns: list[dict],
) -> list[dict[str, Any]]:
    """
    Map each Whisper segment to the speaker with maximum time overlap.
    Returns [{speaker, start_ms, end_ms, text}]
    """
    aligned = []
    for seg in whisper_segments:
        seg_start = int(seg["start"] * 1000)
        seg_end = int(seg["end"] * 1000)

        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0

        for turn in diarization_turns:
            overlap = min(seg_end, turn["end_ms"]) - max(seg_start, turn["start_ms"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        aligned.append({
            "speaker": best_speaker,
            "start_ms": seg_start,
            "end_ms": seg_end,
            "text": seg.get("text", "").strip(),
        })
    return aligned
