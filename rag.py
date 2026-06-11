import json
import boto3
from logger import logger
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv
import asyncio

# ----------------------------
# Load ENV (LangSmith etc)
# ----------------------------
load_dotenv()

# ----------------------------
# Bedrock client
# ----------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# ----------------------------
# Embedding (Titan)
# ----------------------------
def create_embd(query: str):
    body = {"inputText": query}

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["embedding"]

# ----------------------------
# Load FAISS
# ----------------------------
vector_db = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=None,
    allow_dangerous_deserialization=True
)

# ----------------------------
# Reranker
# ----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ----------------------------
# 🔥 Sentence-level compression
# ----------------------------
def compress_context(query, docs):
    sentences = []

    for doc in docs:
        lines = doc.page_content.split("\n")

        for line in lines:
            clean = line.strip()
            if clean:
                sentences.append((clean, doc.metadata))

    # score each sentence
    pairs = [(query, s[0]) for s in sentences]
    sentence_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(sentences, sentence_scores),
        key=lambda x: x[1],
        reverse=True
    )

    #lets take top 5 sentences with threshold score as more than 0.3 only
    threshold_score = 0.3
    top_sentences = [s for s in ranked if s[1]>threshold_score][:5]
    if not top_sentences and ranked:
        top_sentences = ranked[:1]
        logger.info("Sentence-compression fallback applied -> top sentence selected.")
    logger.info(f"after sentence-compression, {len(top_sentences)} sentences selected.")

    #Build context
    context = ""
    sources = []

    for (sentence, metadata), _sentence_score in top_sentences:
        context += f"[SOURCE : {metadata.get('source')}]\n"
        context += sentence + "\n\n"
        sources.append(metadata.get("source"))

    return context, list(set(sources))


# ----------------------------
# MAIN RETRIEVAL
# ----------------------------
def retrieve_context(query: str, metadata_filter: dict | None = None):

    # Step-1: Embed query
    query_vector = create_embd(query)

    # Step-2: Retrieve (HIGH k for recall)
    results = vector_db.similarity_search_with_score_by_vector(
        query_vector,
        k=8
    )

    filtered_docs = []

    # Step-3: Metadata filter =>  API req might come in this way {"doc_type":"policy_terms"}
    for doc, _faiss_score in results:

        if metadata_filter:
            match = all(
                doc.metadata.get(k) == v
                for k, v in metadata_filter.items()
            )
            if not match:
                continue

        filtered_docs.append(doc)

    # Step-4: Fallback
    if not filtered_docs:
        return "No relevant company knowledge found please refine metadata filter or query", [], results

    # ----------------------------
    # 🔥 Step-5: RERANKING
    # ----------------------------
    pairs = [(query, doc.page_content) for doc in filtered_docs]
    reranker_scores = reranker.predict(pairs)
    
    #sort by reranker score (higher = better) 
    reranked = sorted(
        zip(filtered_docs, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # take top 3 best chunks
    top_docs = [doc for doc, _reranker_score in reranked][:3]

    logger.info(f"Reranking applied -> top {len(top_docs)} chunks selected")

    # ----------------------------
    # 🔥 Step-6: COMPRESSION
    # ----------------------------
    context, sources = compress_context(query, top_docs)

    return context, sources, results