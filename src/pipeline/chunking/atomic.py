"""
Phase 2 — Step 1: Aligned transcript → atomic chunks.

Rules (doc 3.4):
  - 1 atomic chunk = 1 complete sentence OR 1 short speaker turn
  - NO hard time splits (not every N seconds)
  - Long sentence (>20s): split at subordinate clause boundary (comma + connectors) or pause >= 0.3s
  - Short sentence (<2s): merge with adjacent same-speaker sentence
  - Speaker boundary: ALWAYS create new chunk on speaker change, even mid-sentence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.pipeline.filler_cleaning import clean_text

# Vietnamese + English sentence connectors used as split points for long sentences
_SPLIT_CONNECTORS = {"và", "nhưng", "vì", "mà", "tuy nhiên", "however", "but", "and", "because"}

# Minimum pause (seconds) to treat as sentence boundary
_PAUSE_THRESHOLD_S = 0.3

# Punctuation that ends a sentence
_SENTENCE_ENDERS = {".", "!", "?", "…", "。", "？", "！"}


@dataclass
class AtomicChunkData:
    speaker: str
    start_ms: int
    end_ms: int
    original_transcript: str
    search_text: str
    words: list[dict] = field(default_factory=list)


def build_atomic_chunks(
    aligned_turns: list[dict[str, Any]],
    whisper_raw_json: dict[str, Any],
    atomic_max_ms: int,
    atomic_min_ms: int,
) -> list[AtomicChunkData]:
    """
    Main entry point. Returns atomic chunks in chronological order.

    Args:
        aligned_turns:    [{speaker, start_ms, end_ms, text}] from Phase 1
        whisper_raw_json: raw Whisper output with word-level timestamps
        atomic_max_ms:    split sentences longer than this (default 20000ms)
        atomic_min_ms:    merge sentences shorter than this (default 2000ms)
    """
    all_words = _extract_all_words(whisper_raw_json)

    raw_chunks: list[AtomicChunkData] = []
    for turn in aligned_turns:
        turn_words = _words_in_range(all_words, turn["start_ms"] / 1000, turn["end_ms"] / 1000)
        chunks = _chunk_turn(turn, turn_words, atomic_max_ms)
        raw_chunks.extend(chunks)

    merged = _merge_short_chunks(raw_chunks, atomic_min_ms)
    return merged


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_all_words(raw_json: dict) -> list[dict]:
    words = []
    for seg in raw_json.get("segments", []):
        for w in seg.get("words", []):
            words.append(w)
    return sorted(words, key=lambda w: w.get("start", 0))


def _words_in_range(words: list[dict], start_s: float, end_s: float) -> list[dict]:
    return [w for w in words if start_s <= w.get("start", 0) < end_s]


def _chunk_turn(
    turn: dict,
    words: list[dict],
    max_ms: int,
) -> list[AtomicChunkData]:
    """Split one speaker turn into sentence-level atomic chunks."""
    speaker = turn["speaker"]

    if not words:
        # No word-level data — treat entire turn as one chunk
        return [AtomicChunkData(
            speaker=speaker,
            start_ms=turn["start_ms"],
            end_ms=turn["end_ms"],
            original_transcript=turn.get("text", "").strip(),
            search_text=clean_text(turn.get("text", "")),
            words=[],
        )]

    # Group words into sentence groups
    sentence_groups = _split_words_into_sentences(words)

    chunks = []
    for group in sentence_groups:
        if not group:
            continue
        start_ms = int(group[0]["start"] * 1000)
        end_ms = int(group[-1]["end"] * 1000)
        text = " ".join(w.get("word", "") for w in group).strip()
        duration_ms = end_ms - start_ms

        if duration_ms > max_ms:
            # Long sentence: split further at connectors + pauses
            sub_chunks = _force_split_long(group, speaker, max_ms)
            chunks.extend(sub_chunks)
        else:
            chunks.append(AtomicChunkData(
                speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                original_transcript=text,
                search_text=clean_text(text),
                words=group,
            ))

    return chunks


def _split_words_into_sentences(words: list[dict]) -> list[list[dict]]:
    """
    Group words into sentence-like groups using:
    1. Sentence-ending punctuation
    2. Pause >= 0.3s between consecutive words
    """
    if not words:
        return []

    groups: list[list[dict]] = []
    current: list[dict] = []

    for i, word in enumerate(words):
        current.append(word)
        word_text = word.get("word", "").strip()

        is_boundary = False

        # Sentence-ending punctuation
        if any(word_text.endswith(p) for p in _SENTENCE_ENDERS):
            is_boundary = True

        # Pause before next word >= threshold
        elif i + 1 < len(words):
            gap = words[i + 1].get("start", 0) - word.get("end", 0)
            if gap >= _PAUSE_THRESHOLD_S:
                is_boundary = True

        if is_boundary and current:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _force_split_long(words: list[dict], speaker: str, max_ms: int) -> list[AtomicChunkData]:
    """
    Split a group of words that exceeds max_ms.
    Strategy: find best split point (connector word after pause) and recurse.
    """
    if not words:
        return []

    total_ms = int((words[-1]["end"] - words[0]["start"]) * 1000)
    if total_ms <= max_ms:
        text = " ".join(w.get("word", "") for w in words).strip()
        return [AtomicChunkData(
            speaker=speaker,
            start_ms=int(words[0]["start"] * 1000),
            end_ms=int(words[-1]["end"] * 1000),
            original_transcript=text,
            search_text=clean_text(text),
            words=words,
        )]

    # Find best split point: connector word after the longest pause in the first half
    mid = len(words) // 2
    best_split = mid
    best_pause = -1.0

    for i in range(1, len(words)):
        gap = words[i].get("start", 0) - words[i - 1].get("end", 0)
        word_lower = words[i].get("word", "").strip().lower()
        is_connector = word_lower in _SPLIT_CONNECTORS

        # Score: prefer pauses + connector words in the middle third
        weight = gap * (2.0 if is_connector else 1.0)
        if weight > best_pause:
            best_pause = weight
            best_split = i

    left = _force_split_long(words[:best_split], speaker, max_ms)
    right = _force_split_long(words[best_split:], speaker, max_ms)
    return left + right


def _merge_short_chunks(chunks: list[AtomicChunkData], min_ms: int) -> list[AtomicChunkData]:
    """
    Merge chunks shorter than min_ms with adjacent same-speaker chunk.
    Rule: NEVER merge across speaker boundaries.
    """
    if not chunks:
        return []

    result: list[AtomicChunkData] = [chunks[0]]

    for chunk in chunks[1:]:
        prev = result[-1]
        duration_ms = chunk.end_ms - chunk.start_ms
        same_speaker = chunk.speaker == prev.speaker

        if duration_ms < min_ms and same_speaker:
            # Merge into previous
            prev.end_ms = chunk.end_ms
            prev.original_transcript = (prev.original_transcript + " " + chunk.original_transcript).strip()
            prev.search_text = clean_text(prev.original_transcript)
            prev.words.extend(chunk.words)
        else:
            result.append(chunk)

    return result
