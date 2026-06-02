"""Phase 1 — Step 1: normalize MP4 → WAV 16kHz mono with loudnorm."""
import subprocess
from pathlib import Path

from src.config import settings


def normalize_audio(input_path: str, output_path: str) -> str:
    """Convert video to normalized WAV. Returns output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", str(settings.audio_sample_rate),
        "-ac", str(settings.audio_channels),
        "-af", "loudnorm",
        "-vn",          # strip video track
        "-y",           # overwrite if exists
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path
