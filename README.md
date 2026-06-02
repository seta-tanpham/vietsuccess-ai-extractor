# VietSuccess — AI Content CMS

```
├── backend/    Data pipeline + FastAPI (Phase 1/2/3)
└── frontend/   Chatbot UI (index.html — no build needed)
```

## Quick start

```bash
# 1. Start infra
cd backend && docker compose up -d

# 2. Start API
cd backend && source .venv/bin/activate && uvicorn src.api.main:app --reload

# 3. Open UI
open frontend/index.html
```
