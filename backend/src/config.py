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

    # Diarization backend: "gpt" (text-based, no audio/pyannote) | "pyannote"
    diarization_backend: str = "gpt"
    diarization_gpt_model: str = "gpt-4o-transcribe-diarize"

    # Embedding backend
    embedding_backend: str = "openai"  # openai | bge-m3

    # YouTube crawl (yt-dlp)
    # Merged mp4 lets us run STT and reuse the file for clip export later.
    youtube_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    youtube_download_dir: str = "/tmp/vietsuccess_youtube"
    youtube_fetch_thumbnail: bool = True

    # Anti-bot throttling.
    # yt-dlp sleeps a random amount in [sleep_interval, max_sleep_interval] before each
    # download, and sleep_interval_requests seconds between metadata requests.
    youtube_sleep_interval_s: float = 1.0
    youtube_max_sleep_interval_s: float = 5.0
    youtube_sleep_requests_s: float = 0.75
    # Rest a random amount in [crawl_rest_min_s, crawl_rest_max_s] between videos in a playlist.
    crawl_rest_min_s: float = 20.0
    crawl_rest_max_s: float = 45.0

    # YouTube caption (PRIMARY transcript source — saves Whisper API tokens).
    # When True, YouTube videos try caption first; Whisper STT is the fallback.
    caption_first: bool = True
    caption_languages: str = "vi,en"          # comma-separated preference order
    caption_segment_target_s: float = 10.0    # group short cues into ~Ns segments

    # Phase 4 — summarization
    summary_enabled: bool = True
    summary_model: str = "gpt-4o-mini"
    # Cap transcript chars sent to the LLM (keeps cost/context bounded for long videos)
    summary_max_input_chars: int = 48000
    # Per-topic-segment summaries (each retrievable chunk gets its topic summary)
    topic_summary_enabled: bool = True
    # Embed topic + video summaries into chunk_embeddings so they are searchable in RAG
    summary_embed_enabled: bool = True

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
