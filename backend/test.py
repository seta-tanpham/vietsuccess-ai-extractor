import sys
import os
import uuid
import logging

# Add backend src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../Documents/vietsuccess_POC/backend")))

from src.database import SessionLocal
from src.config import settings
from src.models.video import Video
from src.models.speaker import Speaker
from src.models.chunk import Chunk
from src.models.embedding import ChunkEmbedding
from src.pipeline.embedding import embed_query
from src.agent.chat_agent import run_chat_agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_agent_script")

def run_test():
    db = SessionLocal()
    video_id = uuid.uuid4()
    speaker_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    
    log.info("Creating mock DB records...")
    try:
        # Create video
        video = Video(
            id=video_id,
            title="Cuộc trò chuyện với Quốc Khánh về khởi nghiệp",
            original_filename="quockhanh_khoinghiep.mp4",
            status="ready"
        )
        db.add(video)
        
        # Create speaker
        speaker = Speaker(
            id=speaker_id,
            video_id=video_id,
            diarization_label="SPEAKER_00",
            display_name="Quốc Khánh",
            role="host",
            mapping_status="confirmed"
        )
        db.add(speaker)
        
        # Create chunk
        transcript_text = "Hôm nay tôi muốn chia sẻ về áp lực khởi nghiệp. Khi mới bắt đầu, tôi gặp rất nhiều khó khăn và cảm thấy áp lực lớn, nhưng việc vượt qua áp lực chính là chìa khóa để thành công."
        chunk = Chunk(
            id=chunk_id,
            video_id=video_id,
            speaker_id=speaker_id,
            chunk_type="semantic",
            start_ms=90000,   # 1:30
            end_ms=150000,   # 2:30
            original_transcript=transcript_text,
            search_text=transcript_text.lower(),
            is_low_quality=False
        )
        db.add(chunk)
        db.flush()
        
        # Create embedding
        log.info("Embedding transcript text...")
        emb_vector = embed_query(transcript_text)
        embedding = ChunkEmbedding(
            chunk_id=chunk_id,
            embedding=emb_vector,
            model_name=settings.openai_embedding_model
        )
        db.add(embedding)
        db.commit()
        
        log.info("Mock records successfully committed. Running Agent query...")
        
        query = "Đoạn nào anh Khánh nói về áp lực khởi nghiệp vậy?"
        result = run_chat_agent(db=db, query=query, video_id=video_id)
        
        log.info("=" * 60)
        log.info("AGENT RESPONSE:")
        log.info(result["answer"])
        log.info("=" * 60)
        
        log.info("CITATIONS / RESULTS:")
        for r in result["search_results"]:
            log.info(f"- [{format_time(r['start_ms'])} -> {format_time(r['end_ms'])}] {r['speaker_name']}: {r['transcript']}")
        log.info("=" * 60)
        
        # Assertions
        assert len(result["search_results"]) > 0, "Should have retrieved the mock chunk"
        assert "Quốc Khánh" in result["answer"], "Answer should mention the speaker name"
        assert "1:30" in result["answer"] or "2:30" in result["answer"], "Answer should mention the timestamp"
        
        log.info("TEST PASSED SUCCESSFULLY! ✓")
        
    except Exception as exc:
        log.exception("Test failed: %s", exc)
        raise exc
    finally:
        log.info("Cleaning up mock DB records...")
        try:
            db.query(ChunkEmbedding).filter_by(chunk_id=chunk_id).delete()
            db.query(Chunk).filter_by(id=chunk_id).delete()
            db.query(Speaker).filter_by(id=speaker_id).delete()
            db.query(Video).filter_by(id=video_id).delete()
            db.commit()
        except Exception as cleanup_exc:
            log.warning("Cleanup failed: %s", cleanup_exc)
            db.rollback()
        db.close()

def format_time(ms: int) -> str:
    s = ms // 1000
    m = s // 60
    return f"{m}:{s % 60:02d}"

if __name__ == "__main__":
    run_test()

