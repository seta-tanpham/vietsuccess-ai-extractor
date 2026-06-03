# VietSuccess Backend

FastAPI backend and data pipeline for turning long-form videos into searchable transcript moments. Ingest videos either by **file upload** or by **crawling YouTube** (single video or whole playlist).

## Pipeline

```text
Source: file upload  OR  YouTube crawl (yt-dlp)
  ↓
Phase 1: transcript + speaker turns
  - Transcript:
      • YouTube videos → caption first (youtube-transcript-api, no Whisper tokens)
      • no caption / upload → ffmpeg audio + mlx-whisper / faster-whisper STT
  - Diarization: GPT on text (default, no audio/pyannote) | pyannote (optional)
  - speaker turn merge + LLM speaker naming
  ↓
Phase 2: chunking + quality check
  - atomic sentence chunks → semantic chunks → topic segments
  - quality gates
  ↓
Phase 3: embeddings + pgvector search
  - OpenAI text-embedding-3-small by default, HNSW index
  ↓
Phase 4: summaries → RAG
  - per topic-segment summary + whole-video summary
  - summaries embedded into pgvector (searchable)
  - search enriches each chunk with its topic_summary + video_summary
```

Anti-token / anti-bot design: YouTube videos use the existing caption when available (no Whisper cost), diarization runs on text via GPT (no audio download / no pyannote), and playlist crawling rests randomly between videos to avoid bot detection.

## Requirements

- Docker Desktop
- Python 3.9+ or newer
- ffmpeg
- OpenAI API key for embeddings
- Optional Hugging Face token for pyannote speaker diarization

Install ffmpeg on macOS:

```bash
brew install ffmpeg
```

## Setup

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `.env`:

```env
OPENAI_API_KEY=...
```

Optional speaker detection:

```env
PYANNOTE_AUTH_TOKEN=hf_...
PYANNOTE_USE_TITLE_SPEAKER_HINT=false
```

Start local services:

```bash
docker compose up -d
```

Run migrations:

```bash
alembic upgrade head
```

Run API:

```bash
uvicorn src.api.main:app --reload
```

## Services

| Service | Host |
|---|---|
| FastAPI | `http://localhost:8000` |
| Swagger docs | `http://localhost:8000/docs` |
| PostgreSQL | `localhost:5433` |
| MinIO API | `localhost:9000` |
| MinIO Console | `http://localhost:9001` |

MinIO default credentials:

```text
minioadmin / minioadmin123
```

## Pipeline CLI

List videos:

```bash
python scripts/run_pipeline.py --list
```

Run full pipeline:

```bash
python scripts/run_pipeline.py "video/my-video.webm"
```

Run specific phases on an existing video:

```bash
python scripts/run_pipeline.py --id <video_id> --phases 1
python scripts/run_pipeline.py --id <video_id> --phases 2
python scripts/run_pipeline.py --id <video_id> --phases 3
python scripts/run_pipeline.py --id <video_id> --phases 2 3
```

Per-phase smoke tests:

```bash
python scripts/test_phase1.py "video/my-video.webm"
python scripts/test_phase2.py <video_id>
python scripts/test_phase3.py <video_id>
python scripts/test_phase3.py <video_id> --re
```

## YouTube Crawl

Crawl a single video via the API (processing runs in the background):

```bash
curl -X POST http://localhost:8000/videos/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

Or run a crawl in the foreground with live per-step logs:

```bash
# Single video (skips if already 'ready'; --reprocess to force, --fresh to ignore dedup)
python scripts/crawl_youtube.py 'https://www.youtube.com/watch?v=VIDEO_ID'

# Whole playlist (sequential, resumable — skips videos already 'ready')
python scripts/crawl_playlist.py 'https://www.youtube.com/playlist?list=PLAYLIST_ID'
python scripts/crawl_playlist.py '<playlist_url>' --limit 3            # first 3 only
python scripts/crawl_playlist.py '<playlist_url>' --rest-min 40 --rest-max 90  # rest longer
```

Notes:
- Wrap the URL in single quotes so the shell does not mangle `?`, `&`, `=`.
- Duplicate crawls are blocked by `youtube_video_id` (DB unique index) — the same video is never downloaded/processed twice.
- A YouTube video with subtitles disabled falls back to Whisper STT automatically.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/videos/upload` | Upload video and start processing |
| `POST` | `/videos/crawl` | Crawl a YouTube video and start processing |
| `GET` | `/videos` | List videos |
| `GET` | `/videos/{id}` | Video status and speakers |
| `GET` | `/videos/{id}/transcript` | Transcript and aligned turns |
| `PATCH` | `/videos/{id}/speakers/{speaker_id}` | Confirm speaker name |
| `POST` | `/chunks/{video_id}/run` | Start Phase 2 |
| `GET` | `/chunks/{video_id}` | List chunks |
| `GET` | `/chunks/{video_id}/qc` | Quality report |
| `POST` | `/chunks/{video_id}/rechunk` | Re-run Phase 2 |
| `GET` | `/search` | Vector search; `scope=chunks` (default) `summaries` `all` |
| `GET` | `/search/videos/{video_id}/coverage` | Embedding coverage |

Search examples:

```bash
curl "http://localhost:8000/search?q=cách+kiếm+tiền&limit=5"
curl "http://localhost:8000/search?q=học+ngoại+ngữ&video_id=<video_id>"
# search the topic/video summaries instead of transcript chunks
curl "http://localhost:8000/search?q=thông+điệp+chính&scope=summaries"
```

Each result carries the transcript chunk plus two summary layers: `topic_summary` (its topic segment) and `video_summary` (the whole video), along with chunk IDs, timestamps, speaker metadata, and similarity score.

## Data Model

```text
channels
videos
  ├─ video_assets
  ├─ transcripts
  ├─ speakers
  ├─ chunks
  │   └─ chunk_embeddings
  ├─ video_summaries
  └─ clips
```

Important tables:

- `videos`: status, source (`upload`/`youtube`), YouTube metadata, dedup keys
- `channels`: YouTube channel of a crawled video
- `video_assets`: raw video and normalized audio objects in MinIO
- `transcripts`: raw segments, aligned speaker turns, `source` (`caption`/`stt`)
- `speakers`: diarization labels plus optional confirmed names
- `chunks`: atomic, semantic, topic-segment, and video-summary chunks
- `chunk_embeddings`: pgvector embeddings (transcript chunks + summaries)
- `video_summaries`: Phase 4 whole-video summary, takeaways, topics, entities
- `clips`: reserved for generated video clips

## Status Flow

```text
uploaded
  → transcribing        (caption or, if needed, normalizing → Whisper STT)
  → diarizing
  → ready_for_chunking
  → chunking
  → quality_check
  → ready_for_embedding
  → embedding
  → ready_for_summary
  → summarizing
  → ready
```

Any phase may set status to `error`. (For caption-based YouTube videos the
`normalizing` step is skipped — audio is only fetched when Whisper is needed.)

## Duplicate Handling

**Uploads** are hashed with SHA-256 before processing. If a processed video with the same hash exists:

- a new `videos` row is still created
- `duplicate_of_video_id` points to the canonical video
- transcript and speaker data are copied
- upload/transcription/chunking can be skipped

**YouTube crawls** are deduplicated by `youtube_video_id` (a partial unique index in the DB). The same video — even across multiple playlists or repeated runs — is never downloaded or processed twice; the crawl endpoint/scripts return the existing record instead.

## Speaker Detection

Diarization runs on the **transcript text via GPT by default** (`DIARIZATION_BACKEND=gpt`) — no audio download and no Hugging Face token required. Speakers are then named by an LLM from the transcript + title.

To use audio-based pyannote diarization instead:

```env
DIARIZATION_BACKEND=pyannote
PYANNOTE_AUTH_TOKEN=hf_...
# optional speaker-count hints
PYANNOTE_NUM_SPEAKERS=2
PYANNOTE_MIN_SPEAKERS=2
PYANNOTE_MAX_SPEAKERS=4
```

With `pyannote` selected but no token, Phase 1 falls back to a single `SPEAKER_00`.

## Configuration

Key `.env` values:

```env
DATABASE_URL=postgresql://nguyenthituyetmay:iloveyou044@localhost:5433/vietsuccess
MINIO_ENDPOINT=localhost:9000

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Whisper is the fallback STT (auto = mlx on Apple Silicon, else faster-whisper)
WHISPER_BACKEND=auto
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=vi

EMBEDDING_BACKEND=openai

# Diarization: gpt (text, default) | pyannote (audio)
DIARIZATION_BACKEND=gpt

# YouTube caption as primary transcript (Whisper is the fallback)
CAPTION_FIRST=true
CAPTION_LANGUAGES=vi,en

# Phase 4 summaries → RAG
SUMMARY_ENABLED=true
TOPIC_SUMMARY_ENABLED=true
SUMMARY_EMBED_ENABLED=true

# Anti-bot throttling for crawling
YOUTUBE_SLEEP_INTERVAL_S=1.0
YOUTUBE_MAX_SLEEP_INTERVAL_S=5.0
CRAWL_REST_MIN_S=20.0
CRAWL_REST_MAX_S=45.0
```

## Troubleshooting

Check DB rows from command line:

```bash
docker exec -it vietsuccess_postgres psql -U nguyenthituyetmay -d vietsuccess -c "select id, status, original_filename from videos;"
```

MinIO console:

```text
http://localhost:9101
```

If Phase 3 says everything is already embedded, that is expected for reruns:

```text
All 20 chunks already embedded
```

If diarization logs fallback to `SPEAKER_00`, check `PYANNOTE_AUTH_TOKEN` and model access on Hugging Face.

## Safety

- Never commit `.env`.
- Do not delete migration files in normal development.
- Use `alembic upgrade head` to apply schema changes.
