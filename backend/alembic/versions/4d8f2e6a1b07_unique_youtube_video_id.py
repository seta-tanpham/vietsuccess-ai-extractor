"""unique youtube_video_id (block duplicate crawls at the DB level)

Revision ID: 4d8f2e6a1b07
Revises: 3c7e1a2b9d04
Create Date: 2026-06-03 20:00:00.000000

Partial unique index so the same YouTube video can never be crawled twice.
NULL youtube_video_id (upload videos) is excluded, and Postgres treats NULLs as
distinct anyway, so upload rows are unaffected.

If this fails with a uniqueness error, the DB already has duplicate
youtube_video_id rows — clean them up before upgrading.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "4d8f2e6a1b07"
down_revision: Union[str, None] = "3c7e1a2b9d04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the old non-unique index (created in the youtube_crawl migration) and
    # replace it with a partial UNIQUE index.
    op.drop_index("ix_videos_youtube_video_id", table_name="videos")
    op.create_index(
        "uq_videos_youtube_video_id",
        "videos",
        ["youtube_video_id"],
        unique=True,
        postgresql_where=sa.text("youtube_video_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_videos_youtube_video_id", table_name="videos")
    op.create_index("ix_videos_youtube_video_id", "videos", ["youtube_video_id"])
