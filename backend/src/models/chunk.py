from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.clip import Clip
    from src.models.embedding import ChunkEmbedding
    from src.models.speaker import Speaker
    from src.models.video import Video


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    # Self-ref: atomic.parent → semantic; semantic.parent → topic_segment
    parent_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"))
    speaker_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("speakers.id", ondelete="SET NULL"))
    # atomic | semantic | topic_segment
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # ltree path: topic_2.semantic_5.atomic_12
    segment_path: Mapped[Optional[str]] = mapped_column(Text)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    original_transcript: Mapped[Optional[str]] = mapped_column(Text)
    search_text: Mapped[Optional[str]] = mapped_column(Text)
    # Approx token count of search_text (size control for embedding/RAG)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    # IDs of source transcript segments this chunk was built from (trace back to transcript)
    segment_ids: Mapped[Optional[list]] = mapped_column(JSONB)
    is_low_quality: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_fail_reasons: Mapped[Optional[list]] = mapped_column(JSONB)
    # Filled in later phases
    summary: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[Optional[list]] = mapped_column(JSONB)
    topic_label: Mapped[Optional[str]] = mapped_column(String(200))
    is_best_moment: Mapped[Optional[bool]] = mapped_column(Boolean)
    best_moment_reason: Mapped[Optional[str]] = mapped_column(Text)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    chapter_index: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="chunks")
    speaker: Mapped[Optional["Speaker"]] = relationship("Speaker", back_populates="chunks")
    embeddings: Mapped[list["ChunkEmbedding"]] = relationship("ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan")
    clips: Mapped[list["Clip"]] = relationship("Clip", back_populates="chunk")
