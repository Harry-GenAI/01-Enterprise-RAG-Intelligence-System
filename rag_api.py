from fastapi import FastAPI
from pydantic import BaseModel
from rag import retrieve_context

app = FastAPI()

@app.get("/")
def rag_knowledge_api():
    return {"status":"rag_knowledge_retriever_api_running"}


class QueryRequest(BaseModel):
    query : str

class ChatResponse(BaseModel):
    context : str
    sources : list[str]


@app.post("/rag", response_model=ChatResponse)
def rag_endpoint(request: QueryRequest):
    context, sources, _debug_results = retrieve_context(request.query)

    return {
        "context":context,
        "sources":sources
    }
