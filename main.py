from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import asyncio
import os
import uvicorn

from logger import logger
from db import create_table, save_chat, get_chat_history

# -----------------------------------------------------
# Ensure FAISS index exists (auto-ingest for container)
# -----------------------------------------------------
if not os.path.exists("faiss_index/index.faiss"):
    logger.info("FAISS index missing → running ingestion pipeline")

    import ingest
    ingest.main()

from rag import retrieve_context
from llmservice import call_llm
from prompts import build_prompt
from safety import is_safe_input, sanitize_context
from query_rewriter import rewrite_query


# ---------------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------------
logger.info("Starting Enterprise RAG API")

create_table()

app = FastAPI()


# ---------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    query: str | None = None      
    session_id: str | None = None
    domain: str | None = None   # metadata filter support


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]
    ci_cd : dict


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/")
def health():
    logger.info("Health check called")
    return {"status": "Enterprise Bedrock RAG running"}


# ---------------------------------------------------------
# MAIN CHAT ENDPOINT
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    # create or reuse session
    session_id = req.session_id or str(uuid.uuid4())
    user_message = req.query

    if not user_message or not user_message.strip():
        raise HTTPException(
            status_code=400,
            detail="Please send 'query' in the request body."
        )

    user_message = user_message.strip()

    logger.info(f"New chat request | session={session_id}")

    # -----------------------------------------------------
    # Guardrails check
    # -----------------------------------------------------
    if not is_safe_input(user_message):
        return {
            "answer": "Request blocked by policy.",
            "session_id": session_id,
            "sources": [],
            "ci_cd": {"version": "v4 deployed"}

        }

    # -----------------------------------------------------
    # Load conversation memory
    # -----------------------------------------------------
    history = get_chat_history(session_id)

    # -----------------------------------------------------
    # Rewrite vague/follow-up questions for better retrieval.
    # query_rewriter.py owns the rewrite prompt and LLM mode.
    # -----------------------------------------------------
    rewritten_query = await asyncio.to_thread(
        rewrite_query,
        user_message,
        history=history
    )
    logger.info(f"Retrieval query: {rewritten_query}")

    # -----------------------------------------------------
    # Metadata filter (domain routing)
    # -----------------------------------------------------
    metadata_filter = None

    if req.domain:
        metadata_filter = {
            "doc_type": req.domain.strip()
        }
        logger.info(f"Metadata filter: {metadata_filter}")

    # -----------------------------------------------------
    # Retrieve context from FAISS (non-blocking)
    # -----------------------------------------------------
    context, sources, _debug_results = await asyncio.to_thread(
        retrieve_context,
        rewritten_query,
        metadata_filter
    )

    if not context:
        logger.warning("No retrieval context found")
        context = "No company knowledge found."
    else:
        context = sanitize_context(context)

    # -----------------------------------------------------
    # Build final prompt
    # -----------------------------------------------------
    prompt = build_prompt(history, context, user_message)

    # -----------------------------------------------------
    # Generate the final user answer.
    # Bedrock calls are blocking, so running them in a worker thread
    # to keep FastAPI responsive when served by uvicorn.
    # -----------------------------------------------------
    answer = await asyncio.to_thread(call_llm, prompt, "answer")

    # -----------------------------------------------------
    # Save conversation memory
    # -----------------------------------------------------
    save_chat(session_id, user_message, answer)

    logger.info(f"Response completed | session={session_id}")
    logger.info(f"context:{context}")

    print("CI/CD deployed successfully...")

    return {
        "answer": answer,
        "session_id": session_id,
        "sources": sources,
        "ci_cd": {"version": "v4 deployed"}
    }

    
