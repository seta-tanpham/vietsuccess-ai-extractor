"""
LLM-based speaker name suggestion — Phase 1, Step 5.

Improvements vs v1:
- Extracts names and show format directly from the video filename/title
- Passes expected speaker count to help LLM consolidate over-segmented labels
- Asks LLM to flag potential duplicates (same person, different label)
- Uses Vietnamese-aware format detection (Kháng Thương, HaveASip, Beyond Income, etc.)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_TURNS_PER_SPEAKER = 8          # enough for rare speakers (observer, child)
_MIN_TURNS_TO_INCLUDE = 1      # include even speakers with just 1 turn


# ── Show format registry ──────────────────────────────────────────────────────
# Each entry: (pattern, expected_min_speakers, expected_max_speakers, description)
_SHOW_FORMATS = [
    # More specific patterns first
    (r"haveasip\s*kids", 4, 6,
     "Kids format: 2 hosts (Thuỳ Minh + Linh Louis) + 1-2 child guests + 1-2 parents"),
    (r"#haveasipkids",   4, 6,
     "Kids format: 2 hosts (Thuỳ Minh + Linh Louis) + 1-2 child guests + 1-2 parents"),
    (r"kháng thương",   2, 3,
     "1-on-1 therapy interview: host Hải Uyên + guest + optional observer (họa sĩ)"),
    (r"beyond income",  2, 3,
     "Financial interview: host + 1 expert guest"),
    (r"have a sip",     2, 3,
     "1-on-1 conversation: host Thuỳ Minh + guest"),
    (r"haveasip",       2, 3,
     "1-on-1 conversation: host Thuỳ Minh + guest"),
]


def detect_show_format(title: str) -> tuple[int, int, str]:
    """Return (min_speakers, max_speakers, format_description) from title."""
    t = title.lower()
    for pattern, mn, mx, desc in _SHOW_FORMATS:
        if re.search(pattern, t):
            return mn, mx, desc
    return 2, 5, "Vietnamese interview or podcast"


def extract_names_from_title(title: str) -> list[str]:
    """
    Extract likely person names from a Vietnamese video title.
    Common formats:
      "Topic - Tên Người | Show EP#"
      "Tên Người | Show"
      "Topic | Tên Người ｜ Show SS# EP#"
    """
    # Split on | ｜ - — and common separators
    parts = re.split(r"[|\｜\-—]+", title)
    names = []
    for part in parts:
        part = part.strip()
        # Remove hashtags, episode markers, timestamps
        part = re.sub(r"#\w+", "", part)
        part = re.sub(r"\b(EP|SS|S)\d+\b", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\[\w+\]", "", part)
        part = part.strip()

        # A name is typically 2-4 capitalized Vietnamese words, shorter than a sentence
        if 2 <= len(part.split()) <= 5 and len(part) < 40:
            # Skip show names we know
            skip_keywords = {"kháng thương", "haveasip", "beyond income", "vietcetera",
                            "british council", "vietsuccess", "podcast", "ep"}
            if not any(kw in part.lower() for kw in skip_keywords):
                names.append(part)
    return names


# ── Prompt builder ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are analyzing a Vietnamese interview/podcast transcript for speaker identification.

Your tasks:
1. Identify each speaker's role and suggest their real name if detectable
2. Flag if two speaker labels are likely the SAME person (over-segmentation by diarization)
3. Use the video title and known names as strong clues

Speaker roles in Vietnamese shows:
- host (người dẫn): asks questions, guides conversation, uses formal Vietnamese
- guest (khách mời): shares personal experiences, answers questions
- child (trẻ em): simpler vocabulary, shorter sentences
- parent (phụ huynh): speaks about their child
- observer (người quan sát): rarely speaks, reacts

Output ONLY valid JSON."""

_KNOWN_HOSTS = {
    r"haveasip\s*kids": ["Thuỳ Minh (main host)", "Linh Louis (teen host, 14 tuổi)"],
    r"#haveasipkids":   ["Thuỳ Minh (main host)", "Linh Louis (teen host, 14 tuổi)"],
    r"kháng thương":    ["Hải Uyên (host, chuyên gia tâm lý)"],
    r"haveasip":        ["Thuỳ Minh (host)"],
    r"have a sip":      ["Thuỳ Minh (host)"],
    r"beyond income":   [""],
}


def _get_known_hosts(title: str) -> list[str]:
    t = title.lower()
    for pattern, hosts in _KNOWN_HOSTS.items():
        if re.search(pattern, t):
            return [h for h in hosts if h]
    return []


_USER_TEMPLATE = """VIDEO TITLE: {title}
SHOW FORMAT: {format_desc}
EXPECTED NUMBER OF SPEAKERS: {min_s}–{max_s} people
KNOWN NAMES FROM TITLE: {known_names}
KNOWN HOSTS FOR THIS SHOW: {known_hosts}

DETECTED SPEAKER LABELS: {n_detected} labels ({labels})
IMPORTANT: If {n_detected} > {max_s}, diarization over-segmented — find and flag the duplicate pairs.

First {n_turns} transcript turns per speaker:
{speaker_blocks}

Respond ONLY with this JSON structure:
{{
  "total_real_speakers": <int>,
  "speakers": [
    {{
      "label": "SPEAKER_00",
      "suggested_role": "host",
      "suggested_name": "Hải Uyên",
      "confidence": "high",
      "reasoning": "asks all questions, introduces topic, formal Vietnamese"
    }}
  ],
  "potential_duplicates": [
    {{
      "label_a": "SPEAKER_00",
      "label_b": "SPEAKER_02",
      "reason": "both labeled as host, similar formal speech pattern",
      "confidence": "medium"
    }}
  ]
}}"""


def suggest_speaker_names(
    speaker_turns: dict[str, list[str]],
    video_title: str = "",
) -> list[dict[str, Any]]:
    """
    Returns list of speaker suggestion dicts AND updates with potential_duplicates.
    Result includes a special key 'potential_duplicates' on the first item (if any).
    """
    from openai import OpenAI
    from src.config import settings

    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY not set — skipping LLM speaker naming")
        return []

    if not speaker_turns:
        return []

    min_s, max_s, format_desc = detect_show_format(video_title)
    known_names = extract_names_from_title(video_title)
    known_hosts = _get_known_hosts(video_title)
    labels = sorted(speaker_turns.keys())
    speaker_blocks = _build_speaker_blocks(speaker_turns)

    prompt = _USER_TEMPLATE.format(
        title=video_title or "VietSuccess interview",
        format_desc=format_desc,
        min_s=min_s,
        max_s=max_s,
        known_names=", ".join(known_names) if known_names else "not identifiable from title",
        known_hosts=", ".join(known_hosts) if known_hosts else "not specified",
        n_detected=len(labels),
        labels=", ".join(labels),
        n_turns=_TURNS_PER_SPEAKER,
        speaker_blocks=speaker_blocks,
    )

    client = OpenAI(api_key=settings.openai_api_key)
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        suggestions = data.get("speakers", [])
        duplicates = data.get("potential_duplicates", [])

        log.info(
            "LLM: %d speakers named, %d potential duplicates flagged. Total real speakers: %s",
            len(suggestions), len(duplicates), data.get("total_real_speakers", "?")
        )
        for dup in duplicates:
            log.info("  Duplicate candidate: %s ↔ %s (%s, confidence=%s)",
                     dup.get("label_a"), dup.get("label_b"),
                     dup.get("reason", ""), dup.get("confidence", ""))

        # Attach duplicates info to first suggestion for caller to use
        if suggestions and duplicates:
            suggestions[0]["_potential_duplicates"] = duplicates

        return suggestions

    except Exception as exc:
        log.warning("LLM speaker naming failed (non-fatal): %s", exc)
        return []


def apply_suggestions_to_db(
    suggestions: list[dict],
    video_id,
    db,
    video_title: str = "",
) -> list[dict]:
    """
    Update Speaker records. Returns potential_duplicates list for deduplication.
    Sets mapping_status = "suggested".

    Fallback: if any speaker is still unnamed after LLM suggestions,
    assign remaining known names from the video title.
    """
    from src.models.speaker import Speaker

    potential_duplicates = []
    if suggestions:
        potential_duplicates = suggestions[0].pop("_potential_duplicates", [])

    suggested_labels = set()
    for suggestion in suggestions:
        label = suggestion.get("label")
        if not label:
            continue

        speaker = db.query(Speaker).filter_by(
            video_id=video_id, diarization_label=label
        ).first()
        if not speaker:
            continue

        if speaker.mapping_status == "auto":
            speaker.display_name = suggestion.get("suggested_name", label)
            speaker.role = suggestion.get("suggested_role", "guest")
            speaker.mapping_status = "suggested"
            suggested_labels.add(label)
            log.info("Named %s → '%s' (%s) [%s]",
                     label, speaker.display_name, speaker.role,
                     suggestion.get("confidence", "?"))

    # Fallback: assign known names from title to any still-unnamed speakers
    if video_title:
        _fallback_name_from_title(video_id, video_title, db, suggested_labels)

    db.commit()
    return potential_duplicates


def _fallback_name_from_title(video_id, video_title: str, db, already_named: set) -> None:
    """
    If there are unnamed speakers AND unassigned known names from the title,
    assign them. Rule: non-host speakers get names extracted from the title.
    """
    from src.models.speaker import Speaker

    # Names already used as display_name in DB
    all_speakers = db.query(Speaker).filter_by(video_id=video_id).all()
    used_names = {
        (s.display_name or "").lower()
        for s in all_speakers
        if s.mapping_status in ("suggested", "confirmed")
    }

    # Unnamed speakers (status still "auto")
    unnamed = [s for s in all_speakers if s.mapping_status == "auto"]
    if not unnamed:
        return

    # Names from title not yet assigned
    known_names = extract_names_from_title(video_title)
    known_hosts = _get_known_hosts(video_title)
    host_names_lower = {h.split("(")[0].strip().lower() for h in known_hosts}

    # Filter out host names and already-used names from candidates
    available = [
        n for n in known_names
        if n.lower() not in used_names and n.lower() not in host_names_lower
    ]

    for speaker, name in zip(unnamed, available):
        speaker.display_name = name
        speaker.role = "guest"
        speaker.mapping_status = "suggested"
        log.info("Fallback named %s → '%s' (from title)", speaker.diarization_label, name)


def extract_first_turns(
    aligned_transcript: list[dict],
    n: int = _TURNS_PER_SPEAKER,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for turn in aligned_transcript:
        label = turn.get("speaker", "SPEAKER_UNKNOWN")
        text = (turn.get("text") or "").strip()
        if not text or len(text) < 5:
            continue
        if label not in result:
            result[label] = []
        if len(result[label]) < n:
            result[label].append(text)
    return result


def _build_speaker_blocks(speaker_turns: dict[str, list[str]]) -> str:
    blocks = []
    for label in sorted(speaker_turns):
        turns = speaker_turns[label][:_TURNS_PER_SPEAKER]
        lines = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(turns) if t.strip())
        blocks.append(f"{label}:\n{lines}")
    return "\n\n".join(blocks)
