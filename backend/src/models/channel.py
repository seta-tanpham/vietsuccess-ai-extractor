from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.video import Video


class Channel(Base):
    """YouTube channel a crawled video belongs to. Only youtube_channel_id + title
    are required for MVP; richer fields are filled opportunistically from yt-dlp metadata."""

    __tablename__ = "channels"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    youtube_channel_id: Mapped[str] = mapped_column(String(30), nullable=False, unique=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(300))
    description: Mapped[Optional[str]] = mapped_column(Text)
    custom_url: Mapped[Optional[str]] = mapped_column(String(100))
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text)
    country: Mapped[Optional[str]] = mapped_column(String(5))
    subscriber_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    view_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    video_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    raw_metadata_key: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    videos: Mapped[list["Video"]] = relationship("Video", back_populates="channel")
