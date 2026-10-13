# rag-postgres: a minimal RAG API on PostgreSQL + pgvector

A small FastAPI service that stores text with its embeddings in PostgreSQL (pgvector), retrieves the closest
documents for a question, and asks a local LLM to answer **only from those documents**. Embeddings and
generation both use [LM Studio](https://lmstudio.ai/)'s OpenAI-compatible server, so nothing leaves your machine.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL%2016-pgvector-4169E1?logo=postgresql&logoColor=white)
![LM Studio](https://img.shields.io/badge/LLM-LM%20Studio%20(local)-6D28D9)

## How it works

```
POST /ingest  {content}          ──► embed search_document: … (nomic-embed-text-v1.5, 768-d) ──► INSERT INTO documents
POST /search  {query, top_k=3}   ──► embed search_query: … ──► ORDER BY embedding <=> query LIMIT top_k
                                                             ──► phi-3.1-mini-4k-instruct answers from the retrieved text
                                                             ──► {query, answer, sources}
```

nomic-embed-text-v1.5 requires those task prefixes, and it returns normalized vectors, so search uses cosine distance (`<=>`).

The whole service is one file, `main.py` (about 130 lines).

## Quickstart

**1. Start PostgreSQL with pgvector**

```bash
docker run -d --name pgvector-db -p 5432:5432 \
  -e POSTGRES_USER=raguser -e POSTGRES_PASSWORD=ragpass -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16
```

**2. Create the table.** `main.py` does not create it for you. For example, run `docker exec -it pgvector-db psql -U raguser -d ragdb`, then:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);
```

Optional ANN index, **after** the table has rows. `main.py` orders by cosine distance (`<=>`), so the opclass has to be `vector_cosine_ops` or PostgreSQL will not use the index. `lists` should stay well below the row count (a common start is `rows / 1000`). An ivfflat index built on an empty or tiny table is still chosen by the planner, and it can return fewer rows than `top_k`.

```sql
CREATE INDEX documents_embedding_idx ON documents
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**3. Install dependencies.** `test_embedding.py` imports `requests`, so install before running it.

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install fastapi uvicorn psycopg2-binary requests
```

**4. Start LM Studio** with the local server on `http://127.0.0.1:1234` and both models loaded:
`text-embedding-nomic-embed-text-v1.5` and `phi-3.1-mini-4k-instruct`. Check the embedding endpoint:

```bash
python test_embedding.py      # prints: Embedding length: 768
```

**5. Run the API**

```bash
uvicorn main:app --reload                           # Swagger UI: http://127.0.0.1:8000/docs
```

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /ingest` | `{"content": "Postgres with pgvector is excellent for RAG systems"}` | `{"status": "success", "embedding_length": 768}` |
| `POST /search` | `{"query": "What is pgvector used for?", "top_k": 3}` | `{"query", "answer", "sources": [...]}` |

```bash
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
     -d '{"content": "pgvector adds a vector type and similarity search to PostgreSQL."}'
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"query": "What does pgvector add?", "top_k": 3}'
```

## Configuration

The database credentials (`raguser` / `ragpass` / `ragdb` on `localhost:5432`), LM Studio URLs and model names are
constants at the top of `main.py`. These are local development defaults. Change them there, or move them to
environment variables before deploying anywhere.

## Limitations

- Each `/ingest` call stores the text as one row. There is no chunking, file upload or deduplication yet.
- One global database connection with autocommit. Fine for a demo, not for concurrent load.
- No error handling for LM Studio being down or returning an error. These surface as HTTP 500.

## Ideas for next steps

- Chunk long documents on ingest and keep their metadata (source, position)
- Replace the IVFFlat index with HNSW (`vector_cosine_ops`) for better recall on a larger table
- Read settings from environment variables, add `requirements.txt` and a `docker-compose.yml` that starts both services
- A small retrieval evaluation (hit rate @k on a handful of question/answer pairs)

## Author

Francis Gallo Sahayaraj. Built to understand RAG on a relational database from the ground up.
