# VietSuccess AI Search

AI search workspace for VietSuccess/Vietcetera-style long-form video content.

The backend ingests interview videos, transcribes them, chunks transcript by topic, embeds searchable chunks, and exposes a FastAPI search API. The frontend is a lightweight ChatGPT-style UI for asking questions across videos.

## Structure

```text
.
├── backend/     FastAPI, SQLAlchemy, pipeline scripts, Alembic migrations
└── frontend/    Static Alpine/Tailwind chat UI
```

## Quick Start

Start infrastructure:

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

Set `OPENAI_API_KEY` in `backend/.env`, then run the API:

```bash
cd backend
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

Open the UI:

```bash
open frontend/index.html
```

If browser CORS or file loading gets fussy, serve the frontend:

```bash
python -m http.server 5500 -d frontend
```

Then open:

```text
http://localhost:5500
```

## Local Services

| Service | URL |
|---|---|
| API | `http://localhost:8000` |
| API docs | `http://localhost:8000/docs` |
| PostgreSQL | `localhost:5433` |
| MinIO API | `localhost:9100` |
| MinIO Console | `http://localhost:9101` |

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
python scripts/run_pipeline.py "video/Học bao nhiêu ngoại ngữ là đủ？ ｜ #HaveASipKids x British Council [NvvmeG23eQ4].webm"
```

Run one phase on an existing video:

```bash
python scripts/run_pipeline.py --id <video_id> --phases 3
```

Search API:

```bash
curl "http://localhost:8000/search?q=cách+kiếm+tiền&limit=5"
```

## Notes

- Do not commit `.env`; it contains secrets.
- Phase 1 can run without `PYANNOTE_AUTH_TOKEN`, but all transcript turns will fall back to `SPEAKER_00`.
- Phase 3 is idempotent: existing embeddings are skipped.
- Duplicate videos are detected by SHA-256 and skipped when a processed copy already exists.
