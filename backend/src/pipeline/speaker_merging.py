"""Phase 1 — Step 5: merge consecutive speaker turns with small gaps."""
from typing import Any

from src.config import settings


def merge_speaker_turns(raw_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge consecutive turns from the same speaker if gap < SPEAKER_MERGE_GAP_MS.

    Rule (doc 3.3):
      - Same speaker AND gap < 1500ms → merge into 1 turn.
      - Different speaker (even gap=0) → keep separate.
      - Same speaker but gap >= 1500ms → keep separate (possible topic shift).

    Input/output schema: [{speaker, start_ms, end_ms, text}]
    """
    if not raw_turns:
        return []

    gap_ms = settings.speaker_merge_gap_ms
    merged = [raw_turns[0].copy()]

    for turn in raw_turns[1:]:
        prev = merged[-1]
        same_speaker = turn["speaker"] == prev["speaker"]
        gap = turn["start_ms"] - prev["end_ms"]

        if same_speaker and gap < gap_ms:
            prev["end_ms"] = turn["end_ms"]
            prev["text"] = (prev["text"] + " " + turn["text"]).strip()
        else:
            merged.append(turn.copy())

    return merged
