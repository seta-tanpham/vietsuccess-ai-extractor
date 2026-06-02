from typing import TYPE_CHECKING, Optional
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.chunk import Chunk
    from src.models.video import Video


class Speaker(Base):
    __tablename__ = "speakers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    diarization_label: Mapped[str] = mapped_column(String(30), nullable=False)  # e.g. SPEAKER_00
    display_name: Mapped[Optional[str]] = mapped_column(String(200))
    # host | guest | narrator
    role: Mapped[Optional[str]] = mapped_column(String(30))
    # auto | suggested | confirmed — only confirmed used for production query
    mapping_status: Mapped[str] = mapped_column(String(20), nullable=False, default="auto")
    mapped_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    mapped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="speakers")
    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="speaker")
