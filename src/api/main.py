from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chunks, videos
from src.api.routes import search as search_routes

app = FastAPI(title="VietSuccess CMS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router, prefix="/videos", tags=["videos"])
app.include_router(chunks.router, prefix="/chunks", tags=["chunks"])
app.include_router(search_routes.router, prefix="/search", tags=["search"])


@app.get("/health")
def health():
    return {"status": "ok"}
