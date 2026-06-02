"""
Phase 2 — Step 2: Atomic chunks → semantic chunks with mandatory overlap.

Rules (doc 3.5):
  - Target: 45–90 seconds per semantic chunk
  - MANDATORY overlap: 5–10s (or 1–2 atomic chunks) with adjacent chunks
  - Priority boundaries: speaker change, pause > 1s, topic shift
  - NO separate atomic chunk created for overlap region
  - parent_chunk_id set on each atomic chunk pointing to its semantic parent
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticChunkData:
    start_ms: int
    end_ms: int
    atomic_indices: list[int]          # indices into the atomic list
    overlap_head_indices: list[int]    # atomic indices borrowed from prev chunk (overlap)
    overlap_tail_indices: list[int]    # atomic indices lent to next chunk (overlap)
    chapter_index: int = 0


def build_semantic_chunks(
    atomics: list[Any],                # list of AtomicChunkData
    target_min_ms: int,
    target_max_ms: int,
    overlap_ms: int,
) -> list[SemanticChunkData]:
    """
    Group atomic chunks into semantic chunks of target_min_ms–target_max_ms with overlap.

    Returns SemanticChunkData list. The caller is responsible for DB persistence
    and assigning parent_chunk_id to atomics.
    """
    if not atomics:
        return []

    semantic_chunks: list[SemanticChunkData] = []
    n = len(atomics)
    i = 0

    while i < n:
        window_start = i
        window_end = i  # exclusive pointer

        # Grow window until we hit target_max_ms or a good boundary
        duration = 0
        while window_end < n:
            duration = atomics[window_end].end_ms - atomics[window_start].start_ms

            if duration >= target_max_ms:
                break

            if duration >= target_min_ms:
                # Check if next boundary is a good stop: speaker change or pause > 1s
                if window_end + 1 < n:
                    next_a = atomics[window_end + 1]
                    curr_a = atomics[window_end]
                    pause_ms = next_a.start_ms - curr_a.end_ms
                    speaker_change = next_a.speaker != curr_a.speaker

                    if pause_ms >= 1000 or speaker_change:
                        window_end += 1
                        break

            window_end += 1

        # Ensure at least one atomic in window
        if window_end == window_start:
            window_end = window_start + 1

        core_indices = list(range(window_start, window_end))

        # Overlap head: borrow 1–2 atomics from previous chunk
        head_overlap = _pick_overlap_head(atomics, window_start, overlap_ms)

        # Overlap tail: lend 1–2 atomics to next chunk (recorded but not re-chunked)
        tail_overlap = _pick_overlap_tail(atomics, window_end, n, overlap_ms)

        sem = SemanticChunkData(
            start_ms=atomics[head_overlap[0]].start_ms if head_overlap else atomics[window_start].start_ms,
            end_ms=atomics[tail_overlap[-1]].end_ms if tail_overlap else atomics[window_end - 1].end_ms,
            atomic_indices=core_indices,
            overlap_head_indices=head_overlap,
            overlap_tail_indices=tail_overlap,
        )
        semantic_chunks.append(sem)
        i = window_end

    return semantic_chunks


def assign_segment_paths(
    topic_index: int,
    semantic_chunks: list[SemanticChunkData],
) -> list[str]:
    """
    Return ltree segment paths for each semantic chunk.
    Format: topic_{N}.semantic_{M}
    """
    return [f"topic_{topic_index}.semantic_{m + 1}" for m in range(len(semantic_chunks))]


# ── Private helpers ────────────────────────────────────────────────────────────

def _pick_overlap_head(atomics, window_start: int, overlap_ms: int) -> list[int]:
    """Pick 1–2 atomic indices before window_start whose total time <= overlap_ms."""
    if window_start == 0:
        return []

    result = []
    accumulated = 0
    idx = window_start - 1

    while idx >= 0 and len(result) < 2:
        a = atomics[idx]
        dur = a.end_ms - a.start_ms
        if accumulated + dur > overlap_ms:
            break
        result.insert(0, idx)
        accumulated += dur
        idx -= 1

    return result


def _pick_overlap_tail(atomics, window_end: int, n: int, overlap_ms: int) -> list[int]:
    """Pick 1–2 atomic indices starting at window_end whose total time <= overlap_ms."""
    if window_end >= n:
        return []

    result = []
    accumulated = 0
    idx = window_end

    while idx < n and len(result) < 2:
        a = atomics[idx]
        dur = a.end_ms - a.start_ms
        if accumulated + dur > overlap_ms:
            break
        result.append(idx)
        accumulated += dur
        idx += 1

    return result
