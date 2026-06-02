from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.video import Video


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    # whisper-large-v3 | phowhisper | manual
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    # Raw Whisper output: {segments:[{start,end,text,words:[{word,start,end,probability}]}]}
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    # After speaker turn merging: [{speaker,start_ms,end_ms,text}]
    aligned_transcript: Mapped[Optional[list]] = mapped_column(JSONB)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    # Average word-level probability — <0.65 needs review
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="transcripts")
