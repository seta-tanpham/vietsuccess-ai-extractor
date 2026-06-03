from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.video import Video


class VideoSummary(Base):
    """Phase 4 output — LLM-generated summary / topics for a whole video.
    One row per video (video_id is unique)."""

    __tablename__ = "video_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    short_summary: Mapped[Optional[str]] = mapped_column(Text)
    long_summary: Mapped[Optional[str]] = mapped_column(Text)
    key_takeaways: Mapped[Optional[list]] = mapped_column(JSONB)
    main_topics: Mapped[Optional[list]] = mapped_column(JSONB)
    sub_topics: Mapped[Optional[list]] = mapped_column(JSONB)
    mentioned_entities: Mapped[Optional[list]] = mapped_column(JSONB)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="summary")
