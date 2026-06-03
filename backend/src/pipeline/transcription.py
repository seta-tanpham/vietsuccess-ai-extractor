"""
Phase 1 — Step 2: WAV → raw transcript JSON.

Backend priority (auto-selected):
  1. mlx-whisper  — Apple Silicon only, uses Metal GPU. ~10-20x realtime on M-series.
  2. faster-whisper — CPU fallback, optimized with beam_size=1 + language hint.

Set WHISPER_BACKEND=faster-whisper in .env to force CPU mode.
"""
from typing import Any

from src.config import settings


def transcribe(wav_path: str) -> dict[str, Any]:
    backend = settings.whisper_backend
    if backend == "auto":
        backend = _detect_best_backend()

    if backend == "openai":
        return _transcribe_openai(wav_path)
    elif backend == "mlx":
        return _transcribe_mlx(wav_path)
    return _transcribe_faster_whisper(wav_path)


def _detect_best_backend() -> str:
    """Use mlx on Apple Silicon, faster-whisper elsewhere."""
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "faster-whisper"


# ── mlx-whisper (Apple Silicon — uses Metal GPU) ──────────────────────────────

def _transcribe_mlx(wav_path: str) -> dict[str, Any]:
    import mlx_whisper

    # mlx model name format: mlx-community/whisper-{size}-mlx
    # large-v3-turbo is the sweet spot: same quality as large-v3, 3x smaller, 4x faster
    model_map = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
    }
    model_name = model_map.get(settings.whisper_model, f"mlx-community/whisper-{settings.whisper_model}-mlx")

    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=model_name,
        language=settings.whisper_language or None,
        word_timestamps=True,
        # Greedy decoding — fastest, minimal quality loss for clean audio
        temperature=0.0,
        # Silence filter — skips non-speech regions before feeding to model
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        verbose=False,
    )

    return _normalize_openai_result(result, backend="mlx", model=model_name)


# ── faster-whisper (CPU fallback, optimized) ─────────────────────────────────

_fw_model = None


def _load_fw_model():
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel
        _fw_model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            num_workers=2,
        )
    return _fw_model


def _transcribe_faster_whisper(wav_path: str) -> dict[str, Any]:
    model = _load_fw_model()

    segments_iter, info = model.transcribe(
        wav_path,
        word_timestamps=True,
        language=settings.whisper_language or None,
        # beam_size=1 is greedy decoding — ~2.5x faster than beam_size=5
        beam_size=1,
        # Skip language detection overhead if language is set
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        # VAD filter skips silence chunks before sending to model
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
    )

    segments = []
    words = []
    for seg in segments_iter:
        seg_words = [
            {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
            for w in (seg.words or [])
        ]
        words.extend(seg_words)
        segments.append({
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "words": seg_words,
        })

    confidence = sum(w["probability"] for w in words) / len(words) if words else 0.0

    return {
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "confidence": confidence,
        "model": settings.whisper_model,
        "backend": "faster-whisper",
        "compute_type": settings.whisper_compute_type,
    }


# ── OpenAI Whisper API ─────────────────────────────────────────────────────────

def _split_wav_ffmpeg(wav_path: str, temp_dir: str, segment_time_sec: int = 600) -> list[str]:
    import subprocess
    import glob
    import os
    chunk_pattern = os.path.join(temp_dir, "chunk_%03d.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(segment_time_sec),
        "-c", "copy",
        chunk_pattern
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(glob.glob(os.path.join(temp_dir, "chunk_*.wav")))


def _get_wav_duration_seconds(file_path: str) -> float:
    import os
    # 16kHz, 16-bit (2 bytes) mono PCM WAV has 44 bytes header and 32000 bytes/sec
    size = os.path.getsize(file_path)
    if size <= 44:
        return 0.0
    return (size - 44) / 32000.0


def _transcribe_openai(wav_path: str) -> dict[str, Any]:
    from openai import OpenAI
    import logging
    import os
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    log = logging.getLogger(__name__)
    log.info("Transcribing via OpenAI Whisper API: %s", wav_path)

    client = OpenAI(api_key=settings.openai_api_key)
    file_size = os.path.getsize(wav_path)
    max_bytes = 24 * 1024 * 1024  # 24 MB

    if file_size <= max_bytes:
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )

        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = dict(response)
    else:
        log.info("File size (%d bytes) exceeds 24MB. Splitting and transcribing in parallel...", file_size)
        
        data = {
            "text": "",
            "segments": [],
            "words": []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = _split_wav_ffmpeg(wav_path, temp_dir, segment_time_sec=600)
            num_chunks = len(chunks)
            log.info("Split WAV into %d chunk(s)", num_chunks)
            
            # Calculate offsets upfront
            offsets = []
            current_offset = 0.0
            for chunk_path in chunks:
                offsets.append(current_offset)
                current_offset += _get_wav_duration_seconds(chunk_path)
            
            # Define worker function for parallel transcription
            def transcribe_chunk(idx: int, chunk_path: str) -> tuple[int, dict]:
                log.info("Starting transcription for chunk %d/%d: %s", idx + 1, num_chunks, chunk_path)
                with open(chunk_path, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        file=f,
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"]
                    )
                log.info("Completed transcription for chunk %d/%d", idx + 1, num_chunks)
                if hasattr(resp, "model_dump"):
                    return idx, resp.model_dump()
                return idx, dict(resp)
            
            # Execute in parallel threads
            with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                futures = [executor.submit(transcribe_chunk, idx, path) for idx, path in enumerate(chunks)]
                results = [f.result() for f in futures]
            
            # Sort by index to maintain correct chronological order
            results.sort(key=lambda x: x[0])
            chunk_data_list = [r[1] for r in results]
            
            # Merge results
            segment_id_counter = 0
            for i, chunk_data in enumerate(chunk_data_list):
                offset_seconds = offsets[i]
                
                # Append text
                if data["text"]:
                    data["text"] += " " + chunk_data.get("text", "").strip()
                else:
                    data["text"] = chunk_data.get("text", "").strip()
                
                # Process words
                chunk_words = chunk_data.get("words") or []
                for w in chunk_words:
                    w["start"] = w["start"] + offset_seconds
                    w["end"] = w["end"] + offset_seconds
                    data["words"].append(w)
                
                # Process segments
                chunk_segments = chunk_data.get("segments") or []
                for seg in chunk_segments:
                    seg["id"] = segment_id_counter
                    segment_id_counter += 1
                    seg["start"] = seg["start"] + offset_seconds
                    seg["end"] = seg["end"] + offset_seconds
                    data["segments"].append(seg)

    # Group top-level words into segments
    words = data.get("words", [])
    segments = data.get("segments")
    if words and segments:
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg["words"] = [
                w for w in words
                if seg_start - 0.05 <= w.get("start", 0.0) <= seg_end + 0.05
            ]

    return _normalize_openai_result(data, backend="openai", model="whisper-1")


# ── Shared normalizer for openai-style result (mlx returns openai format) ─────

def _normalize_openai_result(result: dict, backend: str, model: str) -> dict[str, Any]:
    segments = []
    all_words = []

    for seg in result.get("segments", []):
        seg_words = [
            {
                "start": w.get("start", seg["start"]),
                "end": w.get("end", seg["end"]),
                "word": w.get("word", ""),
                "probability": w.get("probability", 1.0),
            }
            for w in seg.get("words", [])
        ]
        all_words.extend(seg_words)
        segments.append({
            "id": seg.get("id", 0),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "avg_logprob": seg.get("avg_logprob", 0.0),
            "no_speech_prob": seg.get("no_speech_prob", 0.0),
            "words": seg_words,
        })

    confidence = (
        sum(w["probability"] for w in all_words) / len(all_words) if all_words else 0.0
    )

    return {
        "segments": segments,
        "language": result.get("language", "vi"),
        "language_probability": result.get("language_probability", 1.0),
        "confidence": confidence,
        "model": model,
        "backend": backend,
    }
