"""
Phase 2 — Step 3: Semantic chunks → topic segments (chapters).

Heuristic (doc phase 2, task 4):
  - Long inter-chunk pause (> PAUSE_THRESHOLD_MS)
  - Dominant speaker switches: if speaker distribution flips significantly
  - AI-based topic detection deferred to Phase 3+
"""
from __future__ import annotations

from typing import Any

# Silence gap between semantic chunks that signals a topic boundary
_PAUSE_THRESHOLD_MS = 2000

# Minimum number of semantic chunks per topic
_MIN_SEMANTICS_PER_TOPIC = 2


def group_into_topics(
    semantic_chunks: list[Any],   # list of SemanticChunkData
    atomics: list[Any],           # list of AtomicChunkData (for speaker analysis)
    atomic_indices_per_semantic: list[list[int]],  # maps semantic → atomic indices
) -> list[list[int]]:
    """
    Group semantic chunk indices into topic groups.

    Returns:
        list of groups, each group is a list of semantic chunk indices.
        E.g. [[0,1,2], [3,4,5,6], [7,8]] = 3 topics
    """
    if not semantic_chunks:
        return []

    boundaries = _detect_topic_boundaries(
        semantic_chunks, atomics, atomic_indices_per_semantic
    )

    groups: list[list[int]] = []
    current_group: list[int] = []

    for i, sem in enumerate(semantic_chunks):
        current_group.append(i)
        if i in boundaries and len(current_group) >= _MIN_SEMANTICS_PER_TOPIC:
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

    # Merge single-item groups into adjacent group
    return _merge_tiny_groups(groups)


def assign_topic_segment_paths(topic_index: int) -> str:
    return f"topic_{topic_index}"


# ── Private helpers ────────────────────────────────────────────────────────────

def _detect_topic_boundaries(
    semantic_chunks: list[Any],
    atomics: list[Any],
    atomic_indices_per_semantic: list[list[int]],
) -> set[int]:
    """Return set of semantic chunk indices where a topic boundary occurs AFTER them."""
    boundaries: set[int] = set()

    for i in range(len(semantic_chunks) - 1):
        curr = semantic_chunks[i]
        nxt = semantic_chunks[i + 1]

        # Boundary signal 1: long pause between chunks
        gap_ms = nxt.start_ms - curr.end_ms
        if gap_ms >= _PAUSE_THRESHOLD_MS:
            boundaries.add(i)
            continue

        # Boundary signal 2: dominant speaker flips between chunks
        curr_speakers = _dominant_speakers(atomics, atomic_indices_per_semantic[i])
        next_speakers = _dominant_speakers(atomics, atomic_indices_per_semantic[i + 1])
        if curr_speakers and next_speakers and not curr_speakers.intersection(next_speakers):
            boundaries.add(i)

    return boundaries


def _dominant_speakers(atomics: list[Any], indices: list[int]) -> set[str]:
    """Return speakers who cover > 40% of time in this semantic chunk."""
    if not indices:
        return set()

    speaker_time: dict[str, int] = {}
    total = 0
    for idx in indices:
        a = atomics[idx]
        dur = a.end_ms - a.start_ms
        spk = a.speaker or "UNKNOWN"
        speaker_time[spk] = speaker_time.get(spk, 0) + dur
        total += dur

    if total == 0:
        return set()

    return {spk for spk, t in speaker_time.items() if t / total > 0.40}


def _merge_tiny_groups(groups: list[list[int]]) -> list[list[int]]:
    """Merge groups that are too small (< _MIN_SEMANTICS_PER_TOPIC) into neighbours."""
    if not groups:
        return groups

    result: list[list[int]] = []
    for group in groups:
        if len(group) < _MIN_SEMANTICS_PER_TOPIC and result:
            result[-1].extend(group)
        else:
            result.append(group)

    return result
