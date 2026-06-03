from typing import TypedDict, List, Optional
import uuid
import json
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.config import settings
from src.pipeline.embedding import embed_query
from src.models.speaker import Speaker
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    query: str
    video_id: Optional[uuid.UUID]
    db: Session
    speaker_id: Optional[uuid.UUID]
    speaker_name: Optional[str]
    search_topic: Optional[str]
    detected_speakers: List[dict]
    search_results: List[dict]
    answer: str


def format_ms(ms: int) -> str:
    s = ms // 1000
    m = s // 60
    h = m // 60
    if h > 0:
        return f"{h}:{m % 60:02d}:{s % 60:02d}"
    return f"{m}:{s % 60:02d}"


# ── LangGraph Nodes ────────────────────────────────────────────────────────────

def fetch_speakers(state: AgentState):
    db = state["db"]
    video_id = state.get("video_id")
    
    if not video_id:
        return {"detected_speakers": []}
        
    speakers = db.query(Speaker).filter_by(video_id=video_id).all()
    speakers_list = [
        {
            "id": str(s.id),
            "display_name": s.display_name or s.diarization_label,
            "diarization_label": s.diarization_label,
            "role": s.role or "guest"
        }
        for s in speakers
    ]
    return {"detected_speakers": speakers_list}


def parse_query(state: AgentState):
    query = state.get("query", "")
    speakers = state.get("detected_speakers", [])
    
    if not speakers:
        return {"speaker_id": None, "speaker_name": None, "search_topic": query}
        
    speakers_list_str = "\n".join([
        f"- ID: {s['id']}, Name: {s['display_name']} ({s['diarization_label']}), Role: {s['role']}"
        for s in speakers
    ])
    
    prompt = f"""You are analyzing a user query to see if they are asking about what a specific speaker in the video said.

List of speakers in the video:
{speakers_list_str}

User Query: "{query}"

Analyze the query:
1. Does the query refer to a specific speaker from the list (e.g. "Khánh", "Thuỳ Minh", "anh A", "SPEAKER_01", "khách mời", etc.)? Match flexibly.
2. Extract the topic or question they are asking (e.g. "cách vượt qua áp lực", "lời khuyên học tiếng Anh").

Respond ONLY with a JSON object in this format (no other text or codeblocks):
{{
  "speaker_matched": true or false,
  "speaker_id": "matched speaker UUID (string) or null",
  "speaker_name": "display name of the matched speaker or null",
  "search_topic": "extracted topic query"
}}
"""
    
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Clean markdown code blocks if any
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        sp_id = None
        if data.get("speaker_matched") and data.get("speaker_id"):
            sp_id = uuid.UUID(data["speaker_id"])
            
        return {
            "speaker_id": sp_id,
            "speaker_name": data.get("speaker_name") if sp_id else None,
            "search_topic": data.get("search_topic") or query
        }
    except Exception:
        # Fallback to original query
        return {"speaker_id": None, "speaker_name": None, "search_topic": query}


def retrieve_chunks(state: AgentState):
    db = state["db"]
    query = state.get("search_topic") or state.get("query")
    video_id = state.get("video_id")
    speaker_id = state.get("speaker_id")
    
    try:
        query_vector = embed_query(query.strip())
    except Exception as exc:
        raise RuntimeError(f"Embedding query failed: {exc}")
        
    filters = ["c.chunk_type = 'semantic'", "c.is_low_quality = false", "ce.embedding IS NOT NULL"]
    params = {"query_vec": str(query_vector), "limit": 5}
    
    if video_id:
        filters.append("c.video_id = :video_id")
        params["video_id"] = str(video_id)
        
    if speaker_id:
        filters.append("c.speaker_id = :speaker_id")
        params["speaker_id"] = str(speaker_id)
        
    where_clause = " AND ".join(filters)
    
    sql = text(f"""
        SELECT
            c.id::text              AS chunk_id,
            c.video_id::text        AS video_id,
            c.start_ms,
            c.end_ms,
            c.speaker_id::text      AS speaker_id,
            c.original_transcript,
            c.search_text,
            c.segment_path,
            c.chapter_index,
            s.display_name          AS speaker_name,
            v.title                 AS video_title,
            v.original_filename,
            1 - (ce.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM chunks c
        JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        JOIN videos v ON v.id = c.video_id
        LEFT JOIN speakers s ON s.id = c.speaker_id
        WHERE {where_clause}
        ORDER BY ce.embedding <=> CAST(:query_vec AS vector)
        LIMIT :limit
    """)
    
    rows = db.execute(sql, params).mappings().all()
    
    results = [
        {
            "chunk_id": row["chunk_id"],
            "video_id": row["video_id"],
            "video_title": row["video_title"] or row["original_filename"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "duration_ms": row["end_ms"] - row["start_ms"],
            "speaker_id": row["speaker_id"],
            "speaker_name": row["speaker_name"],
            "transcript": row["original_transcript"],
            "segment_path": row["segment_path"],
            "chapter_index": row["chapter_index"],
            "score": round(float(row["score"]), 4),
        }
        for row in rows
    ]
    
    # Fallback: if speaker filter returned nothing, retry search without speaker filter
    if not results and speaker_id:
        filters_fallback = ["c.chunk_type = 'semantic'", "c.is_low_quality = false", "ce.embedding IS NOT NULL"]
        params_fallback = {"query_vec": str(query_vector), "limit": 5}
        if video_id:
            filters_fallback.append("c.video_id = :video_id")
            params_fallback["video_id"] = str(video_id)
            
        where_clause_fallback = " AND ".join(filters_fallback)
        sql_fallback = text(f"""
            SELECT
                c.id::text              AS chunk_id,
                c.video_id::text        AS video_id,
                c.start_ms,
                c.end_ms,
                c.speaker_id::text      AS speaker_id,
                c.original_transcript,
                c.search_text,
                c.segment_path,
                c.chapter_index,
                s.display_name          AS speaker_name,
                v.title                 AS video_title,
                v.original_filename,
                1 - (ce.embedding <=> CAST(:query_vec AS vector)) AS score
            FROM chunks c
            JOIN chunk_embeddings ce ON ce.chunk_id = c.id
            JOIN videos v ON v.id = c.video_id
            LEFT JOIN speakers s ON s.id = c.speaker_id
            WHERE {where_clause_fallback}
            ORDER BY ce.embedding <=> CAST(:query_vec AS vector)
            LIMIT :limit
        """)
        rows_fallback = db.execute(sql_fallback, params_fallback).mappings().all()
        results = [
            {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "video_title": row["video_title"] or row["original_filename"],
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "duration_ms": row["end_ms"] - row["start_ms"],
                "speaker_id": row["speaker_id"],
                "speaker_name": row["speaker_name"],
                "transcript": row["original_transcript"],
                "segment_path": row["segment_path"],
                "chapter_index": row["chapter_index"],
                "score": round(float(row["score"]), 4),
            }
            for row in rows_fallback
        ]
        
    return {"search_results": results}


def generate_answer(state: AgentState):
    query = state.get("query")
    results = state.get("search_results", [])
    speaker_name = state.get("speaker_name")
    
    if not results:
        return {"answer": "Không tìm thấy nội dung phù hợp trong kho video để trả lời câu hỏi của bạn."}
        
    context_list = []
    for idx, r in enumerate(results):
        start_time = format_ms(r["start_ms"])
        end_time = format_ms(r["end_ms"])
        speaker = r["speaker_name"] or "Không rõ"
        context_list.append(
            f"Segment {idx+1}:\n"
            f"Speaker: {speaker}\n"
            f"Time: {start_time} -> {end_time}\n"
            f"Transcript: {r['transcript']}\n"
        )
    context_str = "\n".join(context_list)
    
    prompt = f"""You are Antigravity, the VietSuccess Content AI Agent.
You are helping an editor/user find specific moments in the video.

User Query: "{query}"

Here are the search results from the video transcript:
{context_str}

Respond in Vietnamese. Your response must:
1. Directly answer the user's question.
2. For each relevant point or section mentioned, specify the exact timestamp range (e.g. [02:15 - 04:30] or [01:05:12 - 01:07:45]) and state who is speaking.
3. Provide a concise summary of what was said.
4. Keep the answer structured, well-formatted, and easy to read.

If the user asked specifically about speaker "{speaker_name or ''}" but the results only contain statements from other speakers, mention that you couldn't find statements by "{speaker_name}" on this topic, but point out what other speakers said instead.

Keep the response concise, engaging, and professional.
"""
    
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0.3
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content.strip()}


# ── Graph Construction ─────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("fetch_speakers", fetch_speakers)
workflow.add_node("parse_query", parse_query)
workflow.add_node("retrieve_chunks", retrieve_chunks)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("fetch_speakers")
workflow.add_edge("fetch_speakers", "parse_query")
workflow.add_edge("parse_query", "retrieve_chunks")
workflow.add_edge("retrieve_chunks", "generate_answer")
workflow.add_edge("generate_answer", END)

agent = workflow.compile()


def run_chat_agent(db: Session, query: str, video_id: Optional[uuid.UUID] = None) -> dict:
    """Run the LangGraph agent synchronously and return results."""
    initial_state = {
        "query": query,
        "video_id": video_id,
        "db": db,
        "speaker_id": None,
        "speaker_name": None,
        "search_topic": None,
        "detected_speakers": [],
        "search_results": [],
        "answer": ""
    }
    
    result = agent.invoke(initial_state)
    return {
        "answer": result["answer"],
        "search_results": result["search_results"]
    }
