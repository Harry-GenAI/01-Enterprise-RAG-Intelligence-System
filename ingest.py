import os
import json
import boto3
from logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor
import time

#Bedrock client

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

#Create embd function
#text -> vector

def create_embedding(text: str):
    
    logger.info("Creating embedding")

    body = {"inputText":text}

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    
    return result["embedding"]


def create_embeddings(texts: list[str]):
    return [create_embedding(text) for text in texts]

#load pdf docs
def load_documents(folder="docs"):
    
    logger.info("loading PDF documents from folder docs folder")

    documents = []

    for file in os.listdir(folder):
        
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(folder,file)

        logger.info(f"Reading file:{file}")

        loader = PyPDFLoader(path)
        docs = loader.load()
        #--------------------------
        #metadata attachment(which will be use for filter and citations)
        #--------------------------

        for d in docs:
            d.metadata["source"] = file
            d.metadata["doc_type"]=file.replace(".pdf","")
            d.metadata["department"]="general"
        
        documents.extend(docs)

    logger.info(f"Total pages loaded:{len(documents)}")
    
    return documents

#Chunk docs

def chunk_documents(docs):

    logger.info("Chunking documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n","\n",".", " "]
    )

    chunks = splitter.split_documents(docs)

    logger.info(f"documents has been chunked with {len(chunks)} chunks" )

    return chunks


#Build FAISS

def build_faiss_index(chunks):
    logger.info("generating embeddings and building faiss")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    
    start_time = time.time()
    # Using ThreadPoolExecutor to make multiple API calls at the same time
    # 'max_workers=4' means 4 chunks are being embedded simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:
        # map() handles the looping for you and preserves order
        embeddings = list(executor.map(create_embedding, texts))
    
    end_time = time.time()-start_time
    
    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip(texts,embeddings)),
        embedding=None,
        metadatas=metadatas
    )

    return vector_db, end_time

#MAIN INGESTION PP:

def main():

    logger.info("starting ingestion pipeline")

    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_db, latency = build_faiss_index(chunks)

    logger.info("saving FAISS index to faiss_index/")
    vector_db.save_local("faiss_index")

    logger.info(f"Ingestion completed successfully with {len(chunks)} chunks and in {latency:.2f} secs")
    
    

if __name__ == "__main__":
    main()
    



            







