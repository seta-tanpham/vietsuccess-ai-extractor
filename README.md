# VietSuccess AI Extractor

AI search workspace for VietSuccess/Vietcetera-style long-form video content.

The app ingests interview videos, transcribes speech, maps speaker turns, chunks transcripts into searchable moments, embeds semantic chunks with pgvector, and exposes both a FastAPI backend and a lightweight ChatGPT-style frontend.

## What It Does

```text
Video upload
  -> audio normalization
  -> Whisper transcription with word timestamps
  -> speaker turn alignment and naming
  -> atomic / semantic / topic chunking
  -> quality checks
  -> semantic chunk embeddings
  -> vector search + AI chat answers with timestamps
```

## Structure

```text
.
├── backend/     FastAPI API, SQLAlchemy models, Alembic migrations, pipeline scripts
└── frontend/    Static Alpine.js + Tailwind UI
```

## Pipeline

### Phase 1: Transcription + Speaker Turns

- Normalizes audio with `ffmpeg`
- Transcribes with configurable Whisper backend:
  - `openai`: OpenAI Whisper API
  - `mlx`: fast local Apple Silicon backend via Metal
  - `faster-whisper`: local CPU fallback
- Aligns transcript into speaker turns
- Suggests speaker names and roles where possible

### Phase 2: Chunking + Quality

Chunking is rule-based custom code, not a separate LLM model.

- `atomic`: sentence-level or short speaker-turn chunks
- `semantic`: 45-90 second searchable chunks with overlap
- `topic_segment`: chapter-like containers grouping semantic chunks
- Quality gates mark bad semantic chunks before embedding

### Phase 3: Embeddings + Search

- Embeds only good `semantic` chunks
- Default embedding model: `text-embedding-3-small`
- Stores vectors in PostgreSQL with `pgvector`
- Builds HNSW vector index for retrieval

## Requirements

- Docker Desktop
- Python 3.10+
- `ffmpeg`
- OpenAI API key

Install `ffmpeg` on macOS:

```bash
brew install ffmpeg
```

## Quick Start

Start local infrastructure:

```bash
cd backend
docker compose up -d
```

Create backend environment:

```bash
cd backend
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
```

Set `OPENAI_API_KEY` in `backend/.env`.

Run the API:

```bash
cd backend
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

Open the frontend:

```bash
open frontend/index.html
```

If the browser blocks local file behavior, serve the frontend:

```bash
python -m http.server 5500 -d frontend
```

Then open:

```text
http://localhost:5500
```

## Key Configuration

Main config lives in `backend/.env`.

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# openai | mlx | faster-whisper | auto
WHISPER_BACKEND=openai
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=vi

# Used by faster-whisper CPU fallback
WHISPER_COMPUTE_TYPE=int8
WHISPER_DEVICE=cpu

EMBEDDING_BACKEND=openai
```

Recommended Whisper modes:

```text
openai          Fast cloud transcription
mlx             Fast local transcription on Apple Silicon
faster-whisper  Local CPU fallback, slower
auto            Pick mlx on Apple Silicon if available, otherwise faster-whisper
```

Chunking thresholds:

```env
ATOMIC_MAX_DURATION_MS=20000
ATOMIC_MIN_DURATION_MS=2000
SEMANTIC_TARGET_MIN_MS=45000
SEMANTIC_TARGET_MAX_MS=90000
SEMANTIC_OVERLAP_MS=7500
```

## Local Services

| Service | URL |
|---|---|
| API | `http://localhost:8000` |
| API docs | `http://localhost:8000/docs` |
| PostgreSQL | `localhost:5433` |
| MinIO API | `localhost:9000` |
| MinIO Console | `http://localhost:9001` |

Default MinIO login:

```text
user: minioadmin
pass: minioadmin123
```

## Common Commands

List videos:

```bash
cd backend
python scripts/run_pipeline.py --list
```

Run the full pipeline for a local video:

```bash
cd backend
python scripts/run_pipeline.py "video/my-video.webm"
```

Run selected phases on an existing video:

```bash
python scripts/run_pipeline.py --id <video_id> --phases 1
python scripts/run_pipeline.py --id <video_id> --phases 2
python scripts/run_pipeline.py --id <video_id> --phases 3
python scripts/run_pipeline.py --id <video_id> --phases 2 3
```

Search API:

```bash
curl "http://localhost:8000/search?q=cách+kiếm+tiền&limit=5"
```

Chat agent API:

```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"Khách mời nói gì về áp lực?","video_id":null,"session_id":"demo"}'
```

## API Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/videos/upload` | Upload video and start processing |
| `GET` | `/videos` | List videos |
| `GET` | `/videos/{id}` | Video status, URL, and speakers |
| `GET` | `/videos/{id}/transcript` | Transcript and aligned turns |
| `PATCH` | `/videos/{id}/speakers/{speaker_id}` | Confirm speaker name |
| `POST` | `/chunks/{video_id}/run` | Run Phase 2 + Phase 3 |
| `GET` | `/chunks/{video_id}` | List atomic, semantic, or topic chunks |
| `GET` | `/chunks/{video_id}/qc` | Quality report |
| `POST` | `/chunks/{video_id}/rechunk` | Re-run chunking and embedding |
| `GET` | `/search` | Vector search over semantic chunks |
| `POST` | `/agent/chat` | AI chat answer using retrieval |

## Notes

- Do not commit `.env`; it contains secrets.
- Search and chat retrieval currently use good `semantic` chunks only.
- `atomic` chunks are used to build semantic chunks and run quality checks.
- `topic_segment` chunks are stored as chapter containers, but are not the main search unit yet.
- Phase 3 is idempotent: existing embeddings are skipped.
- Duplicate videos are detected by SHA-256 and skipped when a processed copy already exists.
- There is no real user/account auth layer yet; chat `session_id` is only a frontend conversation ID.
