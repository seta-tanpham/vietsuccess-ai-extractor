"""
Speaker deduplication — Phase 1, Step 6.

Diarization thường split 1 người thành nhiều label:
  SPEAKER_00 = Host (lúc đầu)
  SPEAKER_02 = Host (sau giờ nghỉ)

Hai phương pháp detect:

Method A — LLM name match (fast, no audio needed):
  Nếu GPT đặt cùng tên cho 2 label → khả năng cao là cùng người.
  Merge label ít turns vào label nhiều turns.

Method B — Voice embedding similarity (accurate, cần audio):
  Dùng pyannote SpeechBrain embedding model, tính cosine similarity.
  Nếu similarity > threshold (default 0.85) → merge.
  Chạy sau Method A để bắt các case LLM bỏ sót.

Sau merge:
  - aligned_transcript được update (label cũ → label canonical)
  - Speaker record thừa bị xóa khỏi DB
  - chunk.speaker_id cũng được update nếu Phase 2 đã chạy trước đó
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

# Cosine similarity threshold to treat 2 speakers as the same person
_EMBED_SIMILARITY_THRESHOLD = 0.82  # slightly lower — handles voice variation across time

# Minimum audio seconds per speaker to compute a reliable embedding
_MIN_AUDIO_S_FOR_EMBED = 1.5  # capture brief speakers like observers


def deduplicate_speakers(
    aligned_transcript: list[dict[str, Any]],
    speakers: list[Any],                       # Speaker ORM objects
    wav_path: str | None = None,               # If provided, run Method C too
    llm_duplicates: list[dict] | None = None,  # From LLM's potential_duplicates
    video_id=None,
    db=None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Detect and merge duplicate speaker labels.

    Method A — LLM name match:      same display_name → merge
    Method B — LLM flagged:         LLM explicitly said "these might be same person"
    Method C — Role uniqueness:     a show can only have 1 host → merge duplicate hosts
    Method D — Voice embedding:     cosine similarity > 0.85 (needs WAV + pyannote)

    Returns:
        (updated_aligned_transcript, merge_map)
        merge_map: {old_label: canonical_label}
    """
    merge_map: dict[str, str] = {}

    # Method A: same display_name after LLM naming
    name_merges = _find_duplicates_by_name(speakers)
    merge_map.update(name_merges)

    # Method B: use LLM's explicit duplicate suggestions
    if llm_duplicates:
        llm_merges = _find_duplicates_from_llm(llm_duplicates, speakers)
        for old, canonical in llm_merges.items():
            if old not in merge_map:
                merge_map[old] = canonical

    # Method C: role uniqueness (only 1 host allowed — merge duplicate hosts)
    role_merges = _find_duplicates_by_role(speakers, merge_map)
    for old, canonical in role_merges.items():
        if old not in merge_map:
            merge_map[old] = canonical

    # Method D: voice embedding similarity (only if wav provided and pyannote available)
    if wav_path:
        try:
            embed_merges = _find_duplicates_by_embedding(
                aligned_transcript, speakers, wav_path
            )
            for old, canonical in embed_merges.items():
                if old not in merge_map:
                    merge_map[old] = canonical
        except Exception as exc:
            log.warning("Voice embedding dedup skipped (non-fatal): %s", exc)

    if not merge_map:
        log.info("No duplicate speakers detected")
        return aligned_transcript, {}

    log.info("Duplicate speaker merges: %s", merge_map)

    # Apply merge to transcript turns
    updated = _apply_merge(aligned_transcript, merge_map)

    # Update DB if session provided
    if db and video_id:
        _apply_merge_to_db(merge_map, video_id, db)

    return updated, merge_map


# ── Method B: LLM-flagged duplicates ─────────────────────────────────────────

def _find_duplicates_from_llm(
    llm_duplicates: list[dict],
    speakers: list[Any],
) -> dict[str, str]:
    """Use LLM's potential_duplicates list to build merge map."""
    label_set = {s.diarization_label for s in speakers}
    turn_counts = _count_turns_per_speaker(speakers)
    merge_map: dict[str, str] = {}

    for dup in llm_duplicates:
        la = dup.get("label_a", "")
        lb = dup.get("label_b", "")
        confidence = dup.get("confidence", "low")

        if la not in label_set or lb not in label_set:
            continue
        if confidence not in ("high", "medium"):
            continue  # Only act on high/medium confidence

        # Keep the one with more turns as canonical
        if turn_counts.get(la, 0) >= turn_counts.get(lb, 0):
            merge_map[lb] = la
        else:
            merge_map[la] = lb

        log.info("Method B (LLM): %s → %s [%s confidence, reason: %s]",
                 merge_map.get(lb, la), la if lb in merge_map else lb,
                 confidence, dup.get("reason", ""))

    return merge_map


# ── Method C: Role uniqueness ─────────────────────────────────────────────────

# Roles where only ONE person is expected per video
# Roles where duplicates are suspicious but NOT automatically merged
# (some shows have 2 hosts — merge only if voice embedding confirms)
_CHECK_FOR_DUPLICATE_ROLES = {"host", "moderator", "anchor"}

# Only merge same-role speakers when voice embedding agrees too
_SINGLETON_ROLES: set = set()  # disabled — too aggressive for co-host shows


def _find_duplicates_by_role(
    speakers: list[Any],
    existing_merge_map: dict[str, str],
) -> dict[str, str]:
    """
    If 2+ speakers have the same singleton role (e.g. 'host'),
    keep the one with the most turns, merge the rest.
    """
    from collections import defaultdict

    role_groups: dict[str, list[Any]] = defaultdict(list)
    for spk in speakers:
        role = (spk.role or "").strip().lower()
        # Skip already-marked-for-merge speakers
        if spk.diarization_label not in existing_merge_map and role in _SINGLETON_ROLES:
            role_groups[role].append(spk)

    merge_map: dict[str, str] = {}
    turn_counts = _count_turns_per_speaker(speakers)

    for role, group in role_groups.items():
        if len(group) < 2:
            continue
        # canonical = highest turn count
        canonical = max(group, key=lambda s: turn_counts.get(s.diarization_label, 0))
        for spk in group:
            if spk.diarization_label != canonical.diarization_label:
                merge_map[spk.diarization_label] = canonical.diarization_label
                log.info("Method C (role uniqueness): %s → %s (duplicate '%s' role)",
                         spk.diarization_label, canonical.diarization_label, role)

    return merge_map


def _count_turns_per_speaker(speakers: list[Any]) -> dict[str, int]:
    """Placeholder — real count comes from aligned_transcript. Used for ordering."""
    return {s.diarization_label: 0 for s in speakers}


# ── Method A: LLM name deduplication ─────────────────────────────────────────

def _find_duplicates_by_name(speakers: list[Any]) -> dict[str, str]:
    """
    Group speakers with the same suggested/confirmed display_name.
    Keep the label with the most turns (by DB record age = earlier created_at).
    Return {old_label: canonical_label}.
    """
    name_to_labels: dict[str, list[Any]] = defaultdict(list)

    for spk in speakers:
        name = (spk.display_name or "").strip().lower()
        status = spk.mapping_status or "auto"
        if name and status in ("suggested", "confirmed"):
            name_to_labels[name].append(spk)

    merge_map: dict[str, str] = {}
    for name, group in name_to_labels.items():
        if len(group) < 2:
            continue

        # canonical = the one created first (earliest diarization label usually = correct)
        canonical = sorted(group, key=lambda s: s.created_at)[0]
        for spk in group:
            if spk.diarization_label != canonical.diarization_label:
                merge_map[spk.diarization_label] = canonical.diarization_label
                log.info(
                    "Method A: %s → %s (same name: '%s')",
                    spk.diarization_label, canonical.diarization_label, name
                )

    return merge_map


# ── Method B: Voice embedding similarity ──────────────────────────────────────

def _find_duplicates_by_embedding(
    aligned_transcript: list[dict],
    speakers: list[Any],
    wav_path: str,
) -> dict[str, str]:
    """
    Extract d-vector embeddings for each speaker, compute pairwise cosine similarity.
    Pairs above threshold → merge the smaller one into the larger.
    """
    import numpy as np
    from pyannote.audio import Inference, Model

    # Load pyannote's speaker embedding model
    # wespeaker-voxceleb-resnet34-LM is lightweight + accurate
    try:
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        inference = Inference(model, window="whole")
    except Exception as exc:
        log.warning("Could not load embedding model: %s", exc)
        return {}

    # Collect time ranges per speaker
    speaker_segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for turn in aligned_transcript:
        label = turn.get("speaker", "")
        if label:
            speaker_segments[label].append(
                (turn["start_ms"] / 1000, turn["end_ms"] / 1000)
            )

    # Compute mean embedding per speaker (average over first N segments)
    speaker_embeddings: dict[str, Any] = {}
    for label, segs in speaker_segments.items():
        total_s = sum(e - s for s, e in segs)
        if total_s < _MIN_AUDIO_S_FOR_EMBED:
            continue
        try:
            from pyannote.core import Segment
            embeddings = []
            for start, end in segs[:5]:  # use first 5 segments
                if end - start < 0.5:
                    continue
                emb = inference.crop(wav_path, Segment(start, end))
                embeddings.append(emb)
            if embeddings:
                speaker_embeddings[label] = np.mean(embeddings, axis=0)
        except Exception:
            pass

    if len(speaker_embeddings) < 2:
        return {}

    # Pairwise cosine similarity
    labels = list(speaker_embeddings.keys())
    merge_map: dict[str, str] = {}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            ea = speaker_embeddings[la]
            eb = speaker_embeddings[lb]

            # Cosine similarity
            sim = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))

            log.info("Voice similarity %s ↔ %s = %.3f", la, lb, sim)

            if sim >= _EMBED_SIMILARITY_THRESHOLD:
                # Keep the label that appears more in transcript
                count_a = len(speaker_segments[la])
                count_b = len(speaker_segments[lb])
                if count_a >= count_b:
                    merge_map[lb] = la
                    log.info("Method B: %s → %s (similarity=%.3f)", lb, la, sim)
                else:
                    merge_map[la] = lb
                    log.info("Method B: %s → %s (similarity=%.3f)", la, lb, sim)

    return merge_map


# ── Apply merge ───────────────────────────────────────────────────────────────

def _apply_merge(
    aligned_transcript: list[dict],
    merge_map: dict[str, str],
) -> list[dict]:
    """Replace old speaker labels with canonical labels in transcript."""
    if not merge_map:
        return aligned_transcript

    updated = []
    for turn in aligned_transcript:
        new_turn = turn.copy()
        label = turn.get("speaker", "")
        # Resolve transitively: A→B, B→C means A→C
        canonical = _resolve_canonical(label, merge_map)
        new_turn["speaker"] = canonical
        updated.append(new_turn)

    return updated


def _resolve_canonical(label: str, merge_map: dict[str, str]) -> str:
    """Follow the merge chain until stable (handles A→B→C transitively)."""
    seen = set()
    while label in merge_map and label not in seen:
        seen.add(label)
        label = merge_map[label]
    return label


def _apply_merge_to_db(merge_map: dict[str, str], video_id, db) -> None:
    """
    Delete duplicate speaker records from DB.
    Update any chunk.speaker_id pointing to the old label.
    """
    from src.models.speaker import Speaker
    from src.models.chunk import Chunk

    for old_label, canonical_label in merge_map.items():
        old_spk = db.query(Speaker).filter_by(
            video_id=video_id, diarization_label=old_label
        ).first()
        canonical_spk = db.query(Speaker).filter_by(
            video_id=video_id, diarization_label=canonical_label
        ).first()

        if not old_spk or not canonical_spk:
            continue

        # Re-point any chunks that reference the duplicate speaker
        db.query(Chunk).filter_by(
            video_id=video_id, speaker_id=old_spk.id
        ).update({"speaker_id": canonical_spk.id})

        db.delete(old_spk)
        log.info("DB: removed duplicate speaker %s (merged into %s)", old_label, canonical_label)

    db.commit()
