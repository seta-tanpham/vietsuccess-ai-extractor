import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database import get_db
from src.agent.chat_agent import run_chat_agent

router = APIRouter()


class AgentChatRequest(BaseModel):
    query: str
    video_id: Optional[uuid.UUID] = None


@router.post("/chat")
def chat_agent_endpoint(req: AgentChatRequest, db: Session = Depends(get_db)):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    try:
        result = run_chat_agent(db=db, query=req.query, video_id=req.video_id)
        return {
            "content": result["answer"],
            "results": result["search_results"]
        }
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Agent chat endpoint failed: %s", exc)
        raise HTTPException(500, f"Agent error: {exc}")
