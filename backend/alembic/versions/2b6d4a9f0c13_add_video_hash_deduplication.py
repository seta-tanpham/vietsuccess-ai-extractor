"""add video hash deduplication

Revision ID: 2b6d4a9f0c13
Revises: c6ac9be0ed85
Create Date: 2026-06-02 16:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "2b6d4a9f0c13"
down_revision: Union[str, None] = "c6ac9be0ed85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("videos", sa.Column("duplicate_of_video_id", sa.UUID(), nullable=True))
    op.add_column("videos", sa.Column("video_sha256", sa.String(length=64), nullable=True))
    op.create_index("ix_videos_video_sha256", "videos", ["video_sha256"])
    op.create_foreign_key(
        "fk_videos_duplicate_of_video_id_videos",
        "videos",
        "videos",
        ["duplicate_of_video_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_videos_duplicate_of_video_id_videos", "videos", type_="foreignkey")
    op.drop_index("ix_videos_video_sha256", table_name="videos")
    op.drop_column("videos", "video_sha256")
    op.drop_column("videos", "duplicate_of_video_id")
