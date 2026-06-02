# VietSuccess — AI Content CMS · Data Pipeline

AI-powered content management system for VietSuccess knowledge community.  
Processes long-form interview videos into searchable, embeddable transcript chunks.

---

## Pipeline Overview

```
MP4 / WebM
    │
    ▼ Phase 1 — Raw → Clean Transcript
    ├── Audio normalization   ffmpeg → WAV 16kHz mono loudnorm
    ├── Transcription         mlx-whisper (Apple Silicon) / faster-whisper (CPU)
    ├── Speaker diarization   pyannote 3.x  (optional — needs HuggingFace token)
    └── Speaker turn merging  gap < 1.5s same speaker → merge
    │
    ▼ Phase 2 — Chunking + Quality Check
    ├── Atomic chunking       1 sentence = 1 chunk  (rule-based, not fixed-time)
    ├── Semantic chunking     45–90s windows with 5–10s overlap
    ├── Topic segmentation    chapter grouping via heuristics
    └── Quality gate          6 conditions → flag is_low_quality, block embedding
    │
    ▼ Phase 3 — Embedding + pgvector
    ├── Embed search_text     OpenAI text-embedding-3-small  /  local bge-m3
    ├── HNSW index            pgvector cosine similarity
    └── Search API            GET /search?q=...  → ranked chunks with timestamps
```

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Database | PostgreSQL 16 + pgvector + ltree | Vector search + hierarchical chunk paths |
| Object storage | MinIO | S3-compatible, self-hosted |
| Transcription | mlx-whisper `large-v3-turbo` | Apple M-series GPU — ~5 min / 60 min video |
| Diarization | pyannote 3.x | Best open-source speaker detection |
| Embeddings | OpenAI `text-embedding-3-small` | Default; swap to `bge-m3` for zero cost |
| API | FastAPI + SQLAlchemy 2.0 | Async-ready, typed |
| Migrations | Alembic | Schema versioning |

---

## Prerequisites

- Docker Desktop
- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- OpenAI API key (for embeddings)

---

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY in .env
```

### 2. Start infrastructure

```bash
docker compose up -d
docker compose ps   # wait until postgres = healthy
```

| Service | URL |
|---|---|
| PostgreSQL | `localhost:5433` |
| MinIO API | `localhost:9100` |
| MinIO Console | `http://localhost:9101`  (minioadmin / minioadmin123) |

### 3. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Apple Silicon: install mlx-whisper for GPU-accelerated transcription
pip install mlx-whisper
```

### 4. Run database migrations

```bash
# If re-generating after model changes:
rm -f alembic/versions/*.py
alembic revision --autogenerate -m "init schema"
alembic upgrade head
```

---

## Running the Pipeline

### Full pipeline — one command

```bash
python scripts/run_pipeline.py video/myvideo.webm
```

Output:
```
Phase 1  : 185s   (transcription)
Phase 2  : 3s     (chunking)
Phase 3  : 5s     (embedding)
Total    : 193s

Chunks:
  Atomic   : 312
  Semantic : 22
  QC pass  : 21 / 22
  Cost     : $0.0004
```

### Run specific phases on existing video

```bash
# List all processed videos
python scripts/run_pipeline.py --list

# Re-run only Phase 3 (e.g. after changing embedding model)
python scripts/run_pipeline.py --id <video_id> --phases 3

# Re-run Phase 2 + 3 (e.g. after tweaking chunking params)
python scripts/run_pipeline.py --id <video_id> --phases 2 3
```

### Per-phase test scripts

```bash
python scripts/test_phase1.py video/myvideo.webm   # test transcription
python scripts/test_phase2.py <video_id>            # test chunking + QC
python scripts/test_phase3.py <video_id>            # test embedding
python scripts/test_phase3.py <video_id> --re       # re-embed with new model
```

---

## API

```bash
uvicorn src.api.main:app --reload
# Docs: http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/videos/upload` | Upload video → trigger Phase 1 |
| `GET` | `/videos/{id}` | Video status + speakers |
| `GET` | `/videos/{id}/transcript` | Aligned transcript with timestamps |
| `PATCH` | `/videos/{id}/speakers/{sid}` | Confirm speaker name |
| `POST` | `/chunks/{id}/run` | Trigger Phase 2 |
| `GET` | `/chunks/{id}` | List chunks (filterable by type) |
| `GET` | `/chunks/{id}/qc` | QC dashboard — fail breakdown |
| `POST` | `/chunks/{id}/rechunk` | Re-run Phase 2 |
| `GET` | `/search` | Vector search |
| `GET` | `/search/videos/{id}/coverage` | Embedding coverage check |

### Search examples

```bash
# Search across all videos
curl "http://localhost:8000/search?q=tăng+trưởng+bền+vững&limit=5"

# Search in a specific video
curl "http://localhost:8000/search?q=mở+rộng+kinh+doanh&video_id=<id>"

# Filter by speaker
curl "http://localhost:8000/search?q=leadership&speaker_id=<id>"
```

Search response:
```json
{
  "query": "tăng trưởng bền vững",
  "total": 5,
  "results": [
    {
      "chunk_id": "...",
      "video_title": "...",
      "start_ms": 1823000,
      "end_ms": 1891000,
      "speaker_name": "Nguyễn Văn A",
      "transcript": "Tôi nghĩ tăng trưởng bền vững...",
      "score": 0.9124
    }
  ]
}
```

---

## Configuration

Key settings in `.env`:

```bash
# Whisper — auto picks mlx on Apple Silicon
WHISPER_BACKEND=auto
WHISPER_MODEL=large-v3-turbo   # best speed/quality on M-series
WHISPER_LANGUAGE=vi            # skip language detection, saves ~30%

# Embeddings
EMBEDDING_BACKEND=openai       # openai | bge-m3
OPENAI_API_KEY=sk-...

# Chunking
SEMANTIC_TARGET_MIN_MS=45000   # 45s min per semantic chunk
SEMANTIC_TARGET_MAX_MS=90000   # 90s max
SEMANTIC_OVERLAP_MS=7500       # 7.5s overlap between chunks

# Quality gate
QUALITY_MIN_CONFIDENCE=0.65    # Whisper word confidence threshold
QUALITY_MAX_LOW_QUALITY_RATIO=0.08  # alert if > 8% chunks fail
```

### Switching to bge-m3 (free, local)

```bash
pip install FlagEmbedding
# In .env:
EMBEDDING_BACKEND=bge-m3
# Then re-embed existing videos:
python scripts/test_phase3.py <video_id> --re
```

---

## Project Structure

```
.
├── docker-compose.yml
├── .env / .env.example
├── requirements.txt
├── alembic/                    # DB migrations
│
├── src/
│   ├── config.py               # All settings via pydantic-settings
│   ├── database.py             # SQLAlchemy engine + session
│   │
│   ├── models/                 # ORM models
│   │   ├── video.py            # videos, video_assets
│   │   ├── transcript.py       # transcripts (raw_json + aligned)
│   │   ├── speaker.py          # speakers (diarization labels → names)
│   │   ├── chunk.py            # chunks (atomic / semantic / topic_segment)
│   │   ├── embedding.py        # chunk_embeddings (pgvector)
│   │   └── clip.py             # clips (exported video segments)
│   │
│   ├── pipeline/
│   │   ├── audio_normalization.py   # ffmpeg: MP4 → WAV 16kHz
│   │   ├── transcription.py         # mlx-whisper / faster-whisper
│   │   ├── filler_cleaning.py       # regex filler word removal
│   │   ├── diarization.py           # pyannote speaker detection
│   │   ├── speaker_merging.py       # merge turns (gap < 1.5s)
│   │   ├── embedding.py             # OpenAI / bge-m3 batch embed
│   │   ├── phase1.py                # Orchestrator: Phase 1
│   │   ├── phase2.py                # Orchestrator: Phase 2
│   │   ├── phase3.py                # Orchestrator: Phase 3
│   │   ├── runner.py                # Full pipeline (1→2→3)
│   │   ├── video_deduplication.py   # SHA-256 duplicate detection
│   │   └── chunking/
│   │       ├── atomic.py            # Sentence-level chunking rules
│   │       ├── semantic.py          # 45–90s windows + overlap
│   │       ├── topic.py             # Chapter grouping heuristics
│   │       └── quality.py           # 6-condition quality gate
│   │
│   ├── api/
│   │   ├── main.py                  # FastAPI app
│   │   └── routes/
│   │       ├── videos.py            # Upload, status, speaker confirm
│   │       ├── chunks.py            # Chunking, QC dashboard, rechunk
│   │       └── search.py            # Vector search + coverage check
│   │
│   └── storage/
│       └── minio_client.py          # MinIO upload / presigned URLs
│
└── scripts/
    ├── run_pipeline.py         # Main CLI — full pipeline or per-phase
    ├── test_phase1.py          # Test transcription
    ├── test_phase2.py          # Test chunking + QC
    └── test_phase3.py          # Test embedding
```

---

## Database Schema

```
videos ──┬──< video_assets
         ├──< transcripts
         ├──< speakers
         ├──< chunks ──< chunk_embeddings
         │      └── (self-ref: atomic → semantic → topic_segment)
         └──< clips
```

### Video status state machine

```
uploaded → normalizing → transcribing → diarizing
        → ready_for_chunking → chunking → quality_check
        → ready_for_embedding → embedding → ready
        → error  (any step)
```

---

## Quality Gate

Phase 2 runs 6 checks on every semantic chunk before allowing it to be embedded:

| # | Check | Fail threshold | Action |
|---|---|---|---|
| 1 | Empty text | `search_text.strip() == ""` | Flag, skip embed |
| 2 | Low confidence | avg word probability < 0.65 | Flag, allow manual fix |
| 3 | Too short | duration < 3s | Try merge, else flag |
| 4 | Too long | duration > 2 min | Force re-split |
| 5 | Too many null speakers | > 30% atomic chunks without speaker | Require editor confirm |
| 6 | Timestamp overlap | `start_ms[N] >= start_ms[N+1]` | Data corruption — log, skip video |

Target: **< 8% low-quality rate**. Above 8% → investigate before embedding.

---

## Roadmap

- [x] Phase 1 — Raw video → clean aligned transcript
- [x] Phase 2 — Chunking + quality gate
- [x] Phase 3 — Embedding + pgvector search
- [ ] Phase 4 — AI enrichment (summary, key takeaways, best moments)
- [ ] Phase 5 — NL query agent (RAG chatbot)
- [ ] Phase 6 — Platform publishing (YouTube, TikTok, Spotify draft)
