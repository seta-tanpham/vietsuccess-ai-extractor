"""
Phase 2 — Step 4: Quality check gate before embedding.

6 conditions (doc 3.6). Chunk fails → is_low_quality=True, NOT embedded.

# | Check                    | Fail threshold
1 | Empty text               | search_text.strip() == ""
2 | Low confidence           | avg(atomic confidence) < 0.65
3 | Too short                | duration < 3000ms
4 | Too long                 | duration > 120000ms
5 | Too many null speakers   | > 30% atomic chunks have speaker=null/UNKNOWN
6 | Timestamp overlap        | start_ms of chunk N >= start_ms of chunk N+1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QualityResult:
    is_low_quality: bool
    fail_reasons: list[str]


def check_semantic_chunk(
    semantic: Any,                 # SemanticChunkData or Chunk ORM object
    atomic_chunks: list[Any],      # AtomicChunkData or Chunk ORM objects inside this semantic
    min_confidence: float = 0.65,
    min_duration_ms: int = 3000,
    max_duration_ms: int = 120_000,
    max_null_speaker_ratio: float = 0.30,
) -> QualityResult:
    """Run all 6 quality checks on a semantic chunk."""
    reasons: list[str] = []

    search_text = _get_attr(semantic, "search_text", "")
    start_ms = _get_attr(semantic, "start_ms", 0)
    end_ms = _get_attr(semantic, "end_ms", 0)
    duration_ms = end_ms - start_ms

    # Check 1: empty text
    if not (search_text or "").strip():
        reasons.append("empty_text")

    # Check 2: low confidence (avg of atomic chunks inside)
    if atomic_chunks:
        confidences = []
        for a in atomic_chunks:
            conf = _get_attr(a, "confidence", None)
            if conf is not None:
                confidences.append(conf)
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            if avg_conf < min_confidence:
                reasons.append("low_confidence")

    # Check 3: too short
    if duration_ms < min_duration_ms:
        reasons.append("too_short")

    # Check 4: too long
    if duration_ms > max_duration_ms:
        reasons.append("too_long")

    # Check 5: too many null / unknown speakers
    if atomic_chunks:
        null_count = sum(
            1 for a in atomic_chunks
            if not _get_attr(a, "speaker_id", None) and
               _get_attr(a, "speaker", "UNKNOWN") in (None, "UNKNOWN", "SPEAKER_UNKNOWN")
        )
        if null_count / len(atomic_chunks) > max_null_speaker_ratio:
            reasons.append("too_many_null_speakers")

    is_low = bool(reasons)
    return QualityResult(is_low_quality=is_low, fail_reasons=reasons)


def check_timestamp_integrity(chunks: list[Any]) -> list[int]:
    """
    Check 6: detect chunks where start_ms >= next chunk's start_ms (timestamp overlap).
    Returns list of (offending chunk index) in the input list.
    """
    bad_indices: list[int] = []
    sorted_chunks = sorted(enumerate(chunks), key=lambda x: _get_attr(x[1], "start_ms", 0))

    for i in range(len(sorted_chunks) - 1):
        orig_idx, curr = sorted_chunks[i]
        _, nxt = sorted_chunks[i + 1]
        if _get_attr(curr, "start_ms", 0) >= _get_attr(nxt, "start_ms", 0):
            bad_indices.append(orig_idx)

    return bad_indices


def generate_qc_report(chunks: list[Any]) -> dict:
    """
    Generate a per-video QC report dict.
    chunks: list of Chunk ORM objects that have is_low_quality + quality_fail_reasons.
    """
    total = len(chunks)
    if total == 0:
        return {"total": 0, "low_quality_count": 0, "low_quality_pct": 0, "fail_breakdown": {}}

    low_quality = [c for c in chunks if _get_attr(c, "is_low_quality", False)]
    low_pct = round(len(low_quality) / total * 100, 1)

    breakdown: dict[str, int] = {}
    for c in low_quality:
        reasons = _get_attr(c, "quality_fail_reasons", []) or []
        for r in reasons:
            breakdown[r] = breakdown.get(r, 0) + 1

    return {
        "total": total,
        "low_quality_count": len(low_quality),
        "low_quality_pct": low_pct,
        "pass_count": total - len(low_quality),
        "fail_breakdown": breakdown,
        "above_threshold": low_pct > 8.0,  # doc says > 8% → investigate
    }


def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
