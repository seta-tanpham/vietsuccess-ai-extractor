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

    if backend == "mlx":
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
