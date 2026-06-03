"""youtube crawl source + video summaries

Revision ID: 3c7e1a2b9d04
Revises: 2b6d4a9f0c13
Create Date: 2026-06-03 10:00:00.000000

Adds:
  - channels table
  - video_summaries table (Phase 4 output)
  - YouTube metadata columns on videos (all nullable; upload flow unaffected)
  - chunks.token_count / chunks.segment_ids
  - transcripts.source
  - videos.original_filename made nullable + source-aware CHECK constraint
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "3c7e1a2b9d04"
down_revision: Union[str, None] = "2b6d4a9f0c13"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── channels ──────────────────────────────────────────────────────────────
    op.create_table(
        "channels",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("youtube_channel_id", sa.String(length=30), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("custom_url", sa.String(length=100), nullable=True),
        sa.Column("thumbnail_url", sa.Text(), nullable=True),
        sa.Column("country", sa.String(length=5), nullable=True),
        sa.Column("subscriber_count", sa.BigInteger(), nullable=True),
        sa.Column("view_count", sa.BigInteger(), nullable=True),
        sa.Column("video_count", sa.BigInteger(), nullable=True),
        sa.Column("raw_metadata_key", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("youtube_channel_id"),
    )
    op.create_index("ix_channels_youtube_channel_id", "channels", ["youtube_channel_id"])

    # ── videos: source + YouTube metadata columns ────────────────────────────
    op.add_column("videos", sa.Column("source_type", sa.String(length=20), nullable=False, server_default="upload"))
    op.add_column("videos", sa.Column("youtube_video_id", sa.String(length=20), nullable=True))
    op.add_column("videos", sa.Column("youtube_url", sa.Text(), nullable=True))
    op.add_column("videos", sa.Column("canonical_url", sa.Text(), nullable=True))
    op.add_column("videos", sa.Column("crawl_status", sa.String(length=20), nullable=True))
    op.add_column("videos", sa.Column("raw_metadata_key", sa.Text(), nullable=True))
    op.add_column("videos", sa.Column("description", sa.Text(), nullable=True))
    op.add_column("videos", sa.Column("published_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("videos", sa.Column("channel_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("videos", sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("videos", sa.Column("youtube_category_id", sa.String(length=20), nullable=True))
    op.add_column("videos", sa.Column("default_language", sa.String(length=10), nullable=True))
    op.add_column("videos", sa.Column("default_audio_language", sa.String(length=10), nullable=True))
    op.add_column("videos", sa.Column("definition", sa.String(length=5), nullable=True))
    op.add_column("videos", sa.Column("licensed_content", sa.Boolean(), nullable=True))
    op.add_column("videos", sa.Column("caption_available", sa.Boolean(), nullable=True))
    op.add_column("videos", sa.Column("view_count", sa.BigInteger(), nullable=True))
    op.add_column("videos", sa.Column("like_count", sa.BigInteger(), nullable=True))
    op.add_column("videos", sa.Column("comment_count", sa.BigInteger(), nullable=True))
    op.add_column("videos", sa.Column("statistics_crawled_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index("ix_videos_youtube_video_id", "videos", ["youtube_video_id"])
    op.create_index("ix_videos_canonical_url", "videos", ["canonical_url"])
    op.create_foreign_key(
        "fk_videos_channel_id_channels", "videos", "channels", ["channel_id"], ["id"], ondelete="SET NULL"
    )

    # original_filename: was NOT NULL (upload-only); now nullable for YouTube rows.
    op.alter_column("videos", "original_filename", existing_type=sa.Text(), nullable=True)
    op.create_check_constraint(
        "ck_videos_source_identity",
        "videos",
        "(source_type = 'upload' AND original_filename IS NOT NULL) "
        "OR (source_type = 'youtube' AND youtube_video_id IS NOT NULL)",
    )

    # ── transcripts.source ───────────────────────────────────────────────────
    op.add_column("transcripts", sa.Column("source", sa.String(length=20), nullable=True))

    # ── chunks: token_count + segment_ids ────────────────────────────────────
    op.add_column("chunks", sa.Column("token_count", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("segment_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # ── video_summaries (Phase 4) ────────────────────────────────────────────
    op.create_table(
        "video_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("short_summary", sa.Text(), nullable=True),
        sa.Column("long_summary", sa.Text(), nullable=True),
        sa.Column("key_takeaways", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("main_topics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("sub_topics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("mentioned_entities", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id"),
    )


def downgrade() -> None:
    op.drop_table("video_summaries")

    op.drop_column("chunks", "segment_ids")
    op.drop_column("chunks", "token_count")

    op.drop_column("transcripts", "source")

    op.drop_constraint("ck_videos_source_identity", "videos", type_="check")
    op.alter_column("videos", "original_filename", existing_type=sa.Text(), nullable=False)
    op.drop_constraint("fk_videos_channel_id_channels", "videos", type_="foreignkey")
    op.drop_index("ix_videos_canonical_url", table_name="videos")
    op.drop_index("ix_videos_youtube_video_id", table_name="videos")
    for col in (
        "statistics_crawled_at", "comment_count", "like_count", "view_count",
        "caption_available", "licensed_content", "definition", "default_audio_language",
        "default_language", "youtube_category_id", "tags", "channel_id", "published_at",
        "description", "raw_metadata_key", "crawl_status", "canonical_url", "youtube_url",
        "youtube_video_id", "source_type",
    ):
        op.drop_column("videos", col)

    op.drop_index("ix_channels_youtube_channel_id", table_name="channels")
    op.drop_table("channels")
