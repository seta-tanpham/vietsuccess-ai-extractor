# VietSuccess Backend

FastAPI backend and data pipeline for turning long-form videos into searchable transcript moments.

## Pipeline

```text
Video file
  ↓
Phase 1: transcription + speaker turns
  - ffmpeg audio normalization
  - mlx-whisper / faster-whisper transcription
  - optional pyannote diarization
  - speaker turn merge
  ↓
Phase 2: chunking + quality check
  - atomic sentence chunks
  - semantic chunks
  - topic segments
  - quality gates
  ↓
Phase 3: embeddings + pgvector search
  - OpenAI text-embedding-3-small by default
  - HNSW index on pgvector
  - vector search API with timestamps
```

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
| MinIO API | `localhost:9100` |
| MinIO Console | `http://localhost:9101` |

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

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/videos/upload` | Upload video and start processing |
| `GET` | `/videos` | List videos |
| `GET` | `/videos/{id}` | Video status and speakers |
| `GET` | `/videos/{id}/transcript` | Transcript and aligned turns |
| `PATCH` | `/videos/{id}/speakers/{speaker_id}` | Confirm speaker name |
| `POST` | `/chunks/{video_id}/run` | Start Phase 2 |
| `GET` | `/chunks/{video_id}` | List chunks |
| `GET` | `/chunks/{video_id}/qc` | Quality report |
| `POST` | `/chunks/{video_id}/rechunk` | Re-run Phase 2 |
| `GET` | `/search` | Vector search over semantic chunks |
| `GET` | `/search/videos/{video_id}/coverage` | Embedding coverage |

Search examples:

```bash
curl "http://localhost:8000/search?q=cách+kiếm+tiền&limit=5"
curl "http://localhost:8000/search?q=học+ngoại+ngữ&video_id=<video_id>"
curl "http://localhost:8000/search?q=leadership&speaker_id=<speaker_id>"
```

Search returns chunk IDs, video IDs, timestamps, speaker metadata, transcript text, and similarity score.

## Data Model

```text
videos
  ├─ video_assets
  ├─ transcripts
  ├─ speakers
  ├─ chunks
  │   └─ chunk_embeddings
  └─ clips
```

Important tables:

- `videos`: processing status, duplicate hash, source metadata
- `video_assets`: raw video and normalized audio objects in MinIO
- `transcripts`: raw Whisper segments and aligned speaker turns
- `speakers`: diarization labels plus optional confirmed names
- `chunks`: atomic, semantic, and topic segment chunks
- `chunk_embeddings`: pgvector embeddings
- `clips`: reserved for generated video clips

## Status Flow

```text
uploaded
  → normalizing
  → transcribing
  → diarizing
  → ready_for_chunking
  → chunking
  → quality_check
  → ready_for_embedding
  → embedding
  → ready
```

Any phase may set status to `error`.

## Duplicate Handling

Videos are hashed with SHA-256 before processing.

If a processed video with the same hash already exists:

- a new `videos` row is still created
- `duplicate_of_video_id` points to the canonical video
- transcript and speaker data are copied
- upload/transcription/chunking can be skipped

## Speaker Detection

`PYANNOTE_AUTH_TOKEN` enables real speaker diarization.

Without the token, Phase 1 falls back to:

```text
SPEAKER_00
```

Speaker count is flexible by default. Optional overrides:

```env
PYANNOTE_NUM_SPEAKERS=2
PYANNOTE_MIN_SPEAKERS=2
PYANNOTE_MAX_SPEAKERS=4
PYANNOTE_USE_TITLE_SPEAKER_HINT=false
```

## Configuration

Key `.env` values:

```env
DATABASE_URL=postgresql://nguyenthituyetmay:iloveyou044@localhost:5433/vietsuccess
MINIO_ENDPOINT=localhost:9100

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

WHISPER_BACKEND=auto
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=vi
WHISPER_COMPUTE_TYPE=int8
WHISPER_DEVICE=cpu

EMBEDDING_BACKEND=openai
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
