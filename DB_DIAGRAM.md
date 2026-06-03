# Database Diagram

Mermaid ER diagram for the current SQLAlchemy/Alembic schema.

```mermaid
erDiagram
    CHANNELS {
        uuid id PK
        string youtube_channel_id UK
        string title
        text description
        string custom_url
        text thumbnail_url
        string country
        bigint subscriber_count
        bigint view_count
        bigint video_count
        text raw_metadata_key
        timestamptz created_at
    }

    VIDEOS {
        uuid id PK
        uuid duplicate_of_video_id FK
        string video_sha256
        text title
        text original_filename
        int duration_ms
        string language
        string status
        string category
        timestamptz recorded_at
        timestamptz created_at
        timestamptz processed_at
        string source_type
        string youtube_video_id
        text youtube_url
        text canonical_url
        string crawl_status
        text raw_metadata_key
        text description
        timestamptz published_at
        uuid channel_id FK
        jsonb tags
        string youtube_category_id
        string default_language
        string default_audio_language
        string definition
        boolean licensed_content
        boolean caption_available
        bigint view_count
        bigint like_count
        bigint comment_count
        timestamptz statistics_crawled_at
    }

    VIDEO_ASSETS {
        uuid id PK
        uuid video_id FK
        string asset_type
        string minio_bucket
        text object_key
        string mime_type
        bigint size_bytes
        int duration_ms
        timestamptz created_at
    }

    TRANSCRIPTS {
        uuid id PK
        uuid video_id FK
        string provider
        string source
        jsonb raw_json
        jsonb aligned_transcript
        string language
        float confidence
        string model_version
        timestamptz created_at
    }

    SPEAKERS {
        uuid id PK
        uuid video_id FK
        string diarization_label
        string display_name
        string role
        string mapping_status
        uuid mapped_by
        timestamptz mapped_at
        timestamptz created_at
    }

    CHUNKS {
        uuid id PK
        uuid video_id FK
        uuid parent_chunk_id FK
        uuid speaker_id FK
        string chunk_type
        text segment_path
        int start_ms
        int end_ms
        text original_transcript
        text search_text
        int token_count
        jsonb segment_ids
        boolean is_low_quality
        jsonb quality_fail_reasons
        text summary
        jsonb keywords
        string topic_label
        boolean is_best_moment
        text best_moment_reason
        float sentiment_score
        int chapter_index
        timestamptz created_at
    }

    CHUNK_EMBEDDINGS {
        uuid id PK
        uuid chunk_id FK
        vector embedding
        string model_name
        timestamptz created_at
    }

    CLIPS {
        uuid id PK
        uuid video_id FK
        uuid chunk_id FK
        int start_ms
        int end_ms
        string minio_bucket
        text object_key
        text signed_url
        timestamptz signed_url_expires_at
        string status
        uuid requested_by
        timestamptz created_at
    }

    VIDEO_SUMMARIES {
        uuid id PK
        uuid video_id FK
        text short_summary
        text long_summary
        jsonb key_takeaways
        jsonb main_topics
        jsonb sub_topics
        jsonb mentioned_entities
        string model_version
        timestamptz created_at
    }

    CHANNELS ||--o{ VIDEOS : owns
    VIDEOS ||--o{ VIDEO_ASSETS : has
    VIDEOS ||--o{ TRANSCRIPTS : has
    VIDEOS ||--o{ SPEAKERS : has
    VIDEOS ||--o{ CHUNKS : has
    VIDEOS ||--o{ CLIPS : has
    VIDEOS ||--o| VIDEO_SUMMARIES : summarized_by
    VIDEOS ||--o{ VIDEOS : duplicates
    SPEAKERS ||--o{ CHUNKS : speaks_in
    CHUNKS ||--o{ CHUNKS : parent_of
    CHUNKS ||--o{ CHUNK_EMBEDDINGS : embedded_as
    CHUNKS ||--o{ CLIPS : exported_as
```

## Main Flow

```text
channels
  -> videos
      -> video_assets
      -> transcripts
      -> speakers
      -> chunks
          -> chunk_embeddings
          -> clips
      -> video_summaries
```

## Chunk Hierarchy

```text
topic_segment chunk
  -> semantic chunk
      -> atomic chunk
```

Search currently retrieves embedded semantic chunks, while atomic chunks are mainly used for chunk construction, speaker assignment, and quality checks.
