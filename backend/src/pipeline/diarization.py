"""
Phase 1 — Step 4: Speaker diarization via pyannote.audio 3.x.

Requirements:
  1. pip install pyannote.audio torch
  2. Accept model license on HuggingFace (free):
       https://hf.co/pyannote/speaker-diarization-3.1   ← accept
       https://hf.co/pyannote/segmentation-3.0          ← accept
  3. Create read token at https://hf.co/settings/tokens
  4. Set PYANNOTE_AUTH_TOKEN=hf_xxxx in .env

Apple M-series: pipeline auto-uses MPS (Metal GPU) for ~3-5x speedup vs CPU.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import settings

log = logging.getLogger(__name__)


def diarize(
    wav_path: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run pyannote 3.1 diarization on a WAV file.

    Args:
        wav_path:     Path to 16kHz mono WAV file.
        num_speakers: If known, pass exact count for better accuracy.
                      None = auto-detect.

    Returns:
        Raw turns sorted by start time: [{speaker, start_ms, end_ms}]
    """
    if not settings.pyannote_auth_token:
        raise ValueError(
            "PYANNOTE_AUTH_TOKEN not set.\n"
            "Setup:\n"
            "  1. Accept: https://hf.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept: https://hf.co/pyannote/segmentation-3.0\n"
            "  3. Token:  https://hf.co/settings/tokens\n"
            "  4. .env:   PYANNOTE_AUTH_TOKEN=hf_xxxx"
        )

    import torch
    from huggingface_hub import login
    from pyannote.audio import Pipeline

    # PyTorch 2.6 changed weights_only default to True — pyannote checkpoints need False
    # Safe: pyannote models are from HuggingFace, not user-supplied files
    _orig_load = torch.load
    torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})

    # Login globally — works with all pyannote/huggingface_hub version combos
    login(token=settings.pyannote_auth_token, add_to_git_credential=False)

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    finally:
        torch.load = _orig_load  # restore original after loading

    # Apple M-series: use Metal GPU (MPS) — ~3-5x faster than CPU
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        log.info("pyannote: using MPS (Apple Metal GPU)")
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        log.info("pyannote: using CUDA")
    else:
        log.info("pyannote: using CPU")

    # Run diarization — constrain speaker count to reduce over-segmentation
    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
        log.info("pyannote: forcing num_speakers=%d", num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        if min_speakers or max_speakers:
            log.info("pyannote: min_speakers=%s, max_speakers=%s", min_speakers, max_speakers)

    diarization = pipeline(wav_path, **kwargs)

    turns = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "speaker": speaker,          # e.g. "SPEAKER_00"
            "start_ms": int(segment.start * 1000),
            "end_ms": int(segment.end * 1000),
        })

    turns.sort(key=lambda t: t["start_ms"])
    log.info("pyannote detected %d raw turns, %d unique speakers",
             len(turns), len({t["speaker"] for t in turns}))
    return turns


def align_transcript_with_diarization(
    whisper_segments: list[dict],
    diarization_turns: list[dict],
) -> list[dict[str, Any]]:
    """
    Assign a speaker label to every Whisper segment using max time-overlap.

    Strategy:
    - For each Whisper segment, compute overlap with every diarization turn.
    - Assign the speaker with the maximum total overlap.
    - If overlap = 0 (gap / silence), assign the nearest speaker by proximity.

    Returns [{speaker, start_ms, end_ms, text}]
    """
    aligned = []

    for seg in whisper_segments:
        seg_start = int(seg["start"] * 1000)
        seg_end = int(seg["end"] * 1000)
        text = seg.get("text", "").strip()

        speaker = _find_speaker(seg_start, seg_end, diarization_turns)

        aligned.append({
            "speaker": speaker,
            "start_ms": seg_start,
            "end_ms": seg_end,
            "text": text,
        })

    return aligned


def _find_speaker(seg_start: int, seg_end: int, turns: list[dict]) -> str:
    """Return speaker with max overlap; fall back to nearest if no overlap."""
    best_speaker = "SPEAKER_UNKNOWN"
    best_overlap = 0

    for turn in turns:
        overlap = min(seg_end, turn["end_ms"]) - max(seg_start, turn["start_ms"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    if best_speaker == "SPEAKER_UNKNOWN" and turns:
        # No overlap — assign closest turn by distance
        best_speaker = min(
            turns,
            key=lambda t: min(
                abs(seg_start - t["end_ms"]),
                abs(seg_end - t["start_ms"]),
            ),
        )["speaker"]

    return best_speaker


# ── GPT-based diarization (text only — no audio, no pyannote) ──────────────────

_DIARIZATION_SYSTEM_PROMPT = """You are analyzing a Vietnamese transcript to perform speaker diarization (labeling which speaker spoke each segment).

Your task:
Analyze the flow of conversation, questions and answers, forms of address (e.g. 'anh', 'chị', 'em', 'tôi', 'dạ', 'hỏi', 'trả lời'), and conversational context.
Assign a speaker label to each segment (e.g. SPEAKER_00, SPEAKER_01, SPEAKER_02).

Constraints:
1. Ensure the speaker labels are consistent across the entire transcript.
2. Minimize the number of unique speakers (typically 2-3 speakers for an interview/podcast).
3. Do not change the original segment indexes or skip any segments. Every input segment index must be mapped to a speaker label.
4. Output ONLY valid JSON in the requested format.
"""

_DIARIZATION_USER_TEMPLATE = """VIDEO TITLE: {title}

Transcript segments to label:
{segment_blocks}

Respond ONLY with a JSON object in this format:
{{
  "speaker_map": {{
    "0": "SPEAKER_00",
    "1": "SPEAKER_00",
    "2": "SPEAKER_01"
  }}
}}"""


def diarize_with_gpt(
    whisper_segments: list[dict],
    video_title: str = "",
) -> list[dict[str, Any]]:
    """
    Diarize transcript segments using gpt-4o-transcribe-diarize,
    falling back to settings.openai_model if needed. Works purely on text —
    no audio file or pyannote token required.
    """
    import json
    from openai import OpenAI, APIError

    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY not set — all segments assigned to SPEAKER_00.")
        return [
            {
                "speaker": "SPEAKER_00",
                "start_ms": int(seg.get("start", 0.0) * 1000),
                "end_ms": int(seg.get("end", 0.0) * 1000),
                "text": seg.get("text", "").strip(),
            }
            for seg in whisper_segments
        ]

    # Format segments
    segment_lines = []
    for i, seg in enumerate(whisper_segments):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        segment_lines.append(f"{i} [{start:.2f}s - {end:.2f}s]: {text}")
    segment_blocks = "\n".join(segment_lines)

    prompt = _DIARIZATION_USER_TEMPLATE.format(
        title=video_title or "VietSuccess video",
        segment_blocks=segment_blocks,
    )

    client = OpenAI(api_key=settings.openai_api_key)

    model_name = settings.diarization_gpt_model
    log.info("Requesting GPT-based diarization using model: %s", model_name)

    content = None

    def _call(model: str, use_json: bool):
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": _DIARIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs).choices[0].message.content

    try:
        content = _call(model_name, use_json=True)
    except APIError as e:
        # Some models reject response_format=json_object — retry without it, then fall back.
        log.warning("Failed to call %s with json mode: %s. Falling back to %s", model_name, e, settings.openai_model)
        try:
            content = _call(settings.openai_model, use_json=True)
        except Exception as fallback_err:
            log.error("Fallback model %s also failed: %s", settings.openai_model, fallback_err)
    except Exception as e:
        log.warning("Unexpected error with %s: %s. Falling back to %s", model_name, e, settings.openai_model)
        try:
            content = _call(settings.openai_model, use_json=True)
        except Exception as fallback_err:
            log.error("Fallback model %s also failed: %s", settings.openai_model, fallback_err)

    speaker_map = {}
    if content:
        try:
            speaker_map = json.loads(content).get("speaker_map", {})
        except Exception as json_err:
            log.error("Failed to parse GPT diarization JSON response: %s", json_err)

    aligned = []
    for i, seg in enumerate(whisper_segments):
        speaker = speaker_map.get(str(i)) or speaker_map.get(i) or "SPEAKER_00"
        if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
            speaker = f"SPEAKER_{speaker}"
        aligned.append({
            "speaker": speaker,
            "start_ms": int(seg.get("start", 0.0) * 1000),
            "end_ms": int(seg.get("end", 0.0) * 1000),
            "text": seg.get("text", "").strip(),
        })

    return aligned
