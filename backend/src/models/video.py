from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.channel import Channel
    from src.models.chunk import Chunk
    from src.models.clip import Clip
    from src.models.speaker import Speaker
    from src.models.summary import VideoSummary
    from src.models.transcript import Transcript


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    duplicate_of_video_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="SET NULL"))
    video_sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    title: Mapped[Optional[str]] = mapped_column(Text)
    # Nullable: only upload videos have an original filename; YouTube videos use youtube_video_id.
    original_filename: Mapped[Optional[str]] = mapped_column(Text)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    # uploaded|normalizing|transcribing|diarizing|chunking|quality_check|embedding|summarizing|ready|error
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="uploaded")
    category: Mapped[Optional[str]] = mapped_column(String(100))
    recorded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # ── Source / provenance ───────────────────────────────────────────────────
    # 'upload' (default, existing rows) | 'youtube'
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, default="upload")
    youtube_video_id: Mapped[Optional[str]] = mapped_column(String(20), index=True)
    youtube_url: Mapped[Optional[str]] = mapped_column(Text)
    canonical_url: Mapped[Optional[str]] = mapped_column(Text, index=True)
    # pending|processing|completed|failed (crawl phase only; processing tracked by `status`)
    crawl_status: Mapped[Optional[str]] = mapped_column(String(20))
    raw_metadata_key: Mapped[Optional[str]] = mapped_column(Text)

    # ── YouTube metadata (snippet / contentDetails / statistics) ──────────────
    description: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    channel_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("channels.id", ondelete="SET NULL"))
    tags: Mapped[Optional[list]] = mapped_column(JSONB)
    youtube_category_id: Mapped[Optional[str]] = mapped_column(String(20))
    default_language: Mapped[Optional[str]] = mapped_column(String(10))
    default_audio_language: Mapped[Optional[str]] = mapped_column(String(10))
    definition: Mapped[Optional[str]] = mapped_column(String(5))
    licensed_content: Mapped[Optional[bool]] = mapped_column(Boolean)
    caption_available: Mapped[Optional[bool]] = mapped_column(Boolean)
    view_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    like_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    comment_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    statistics_crawled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    assets: Mapped[list["VideoAsset"]] = relationship("VideoAsset", back_populates="video", cascade="all, delete-orphan")
    transcripts: Mapped[list["Transcript"]] = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
    speakers: Mapped[list["Speaker"]] = relationship("Speaker", back_populates="video", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="video", cascade="all, delete-orphan")
    clips: Mapped[list["Clip"]] = relationship("Clip", back_populates="video", cascade="all, delete-orphan")
    channel: Mapped[Optional["Channel"]] = relationship("Channel", back_populates="videos")
    summary: Mapped[Optional["VideoSummary"]] = relationship(
        "VideoSummary", back_populates="video", uselist=False, cascade="all, delete-orphan"
    )


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    # raw_video|normalized_video|audio_wav|thumbnail|exported_clip
    asset_type: Mapped[str] = mapped_column(String(30), nullable=False)
    minio_bucket: Mapped[str] = mapped_column(String(100), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="assets")
