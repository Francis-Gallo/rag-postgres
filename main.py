from fastapi import FastAPI
from pydantic import BaseModel, Field
import psycopg2
import requests

app = FastAPI()

# ----------------------------
# Database connection
# ----------------------------
conn = psycopg2.connect(
    dbname="ragdb",
    user="raguser",
    password="ragpass",
    host="localhost",
    port=5432
)
conn.autocommit = True

# ----------------------------
# Config
# ----------------------------
EMBEDDING_URL = "http://127.0.0.1:1234/v1/embeddings"
CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL = "phi-3.1-mini-4k-instruct"

# ----------------------------
# Request Schemas
# ----------------------------
class IngestRequest(BaseModel):
    content: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1)

# ----------------------------
# Helpers
# ----------------------------
def get_embedding(text: str, prefix: str):
    # nomic-embed-text-v1.5 ignores the retrieval task unless the input
    # starts with search_document: or search_query:.
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBED_MODEL,
            "input": f"{prefix}{text}"
        }
    )
    return response.json()["data"][0]["embedding"]

def generate_answer(context: str, question: str):
    response = requests.post(
        CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer ONLY using the provided context."
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
                }
            ],
            "temperature": 0.2
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# ----------------------------
# Routes
# ----------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    embedding = get_embedding(req.content, "search_document: ")

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (req.content, embedding)
    )

    return {
        "status": "success",
        "embedding_length": len(embedding)
    }

@app.post("/search")
def search(req: SearchRequest):
    query_embedding = get_embedding(req.query, "search_query: ")

    cur = conn.cursor()
    # Cosine distance. These embeddings are normalized, and an ivfflat index
    # is used only when its opclass matches the operator (vector_cosine_ops).
    cur.execute(
        """
        SELECT content
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, req.top_k)
    )

    rows = cur.fetchall()
    documents = [row[0] for row in rows]

    context = "\n\n".join(documents)
    answer = generate_answer(context, req.query)

    return {
        "query": req.query,
        "answer": answer,
        "sources": documents
    }
