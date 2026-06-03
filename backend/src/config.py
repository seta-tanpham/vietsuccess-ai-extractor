from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://nguyenthituyetmay:iloveyou044@localhost:5433/vietsuccess"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin123"
    minio_bucket_videos: str = "vietsuccess-videos"
    minio_bucket_clips: str = "vietsuccess-clips"
    minio_secure: bool = False

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Langfuse
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # Pipeline - Whisper
    # Backend: auto | mlx | faster-whisper
    # "auto" picks mlx on Apple Silicon, faster-whisper elsewhere
    whisper_backend: str = "auto"
    # Model: large-v3-turbo (recommended M5) | large-v3 | medium | small | base
    whisper_model: str = "large-v3-turbo"
    # Language hint — set "vi" to skip detection and save ~30% time
    whisper_language: str = "vi"
    # faster-whisper CPU settings (used only when backend=faster-whisper)
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    pyannote_auth_token: str = ""
    pyannote_num_speakers: Optional[int] = None
    pyannote_min_speakers: Optional[int] = None
    pyannote_max_speakers: Optional[int] = None
    pyannote_use_title_speaker_hint: bool = False

    diarization_backend: str = "gpt"  # gpt | pyannote
    diarization_gpt_model: str = "gpt-4o-transcribe-diarize"

    # Embedding backend
    embedding_backend: str = "openai"  # openai | bge-m3

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # Chunking thresholds
    atomic_max_duration_ms: int = 20000
    atomic_min_duration_ms: int = 2000
    semantic_target_min_ms: int = 45000
    semantic_target_max_ms: int = 90000
    semantic_overlap_ms: int = 7500
    speaker_merge_gap_ms: int = 1500

    # Quality gate
    quality_min_confidence: float = 0.65
    quality_min_duration_ms: int = 3000
    quality_max_duration_ms: int = 120000
    quality_max_null_speaker_ratio: float = 0.30
    quality_max_low_quality_ratio: float = 0.08

    # App
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
