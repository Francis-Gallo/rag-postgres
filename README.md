\# RAG Backend with FastAPI and PostgreSQL (pgvector)



This project is a minimal but production-style Retrieval-Augmented Generation (RAG)

backend built using FastAPI, PostgreSQL, and pgvector.



It demonstrates how to store vector embeddings in a relational database and perform

semantic search using cosine similarity.



---



\## What This Project Does



1\. Accepts text documents through an API

2\. Converts text into vector embeddings using a local embedding model (LM Studio)

3\. Stores embeddings in PostgreSQL using pgvector

4\. Performs semantic similarity search

5\. Returns the most relevant documents

6\. Generates an answer using retrieved context



---



\## Tech Stack



\- Python

\- FastAPI

\- PostgreSQL

\- pgvector

\- Docker

\- LM Studio (OpenAI-compatible local API)



---



\## High-Level Architecture



Client  

→ FastAPI Backend  

→ Embedding Model (LM Studio)  

→ PostgreSQL + pgvector  

→ Semantic Search Results  



---



\## PostgreSQL + pgvector Setup



Run PostgreSQL with pgvector using Docker:



```bash

docker run -d \\

&nbsp; --name pgvector-db \\

&nbsp; -p 5432:5432 \\

&nbsp; -e POSTGRES\_USER=raguser \\

&nbsp; -e POSTGRES\_PASSWORD=ragpass \\

&nbsp; -e POSTGRES\_DB=ragdb \\

&nbsp; pgvector/pgvector:pg16

```



Enable the pgvector extension:



```sql

CREATE EXTENSION IF NOT EXISTS vector;

```



---



\## Database Schema



```sql

CREATE TABLE documents (

&nbsp;   id SERIAL PRIMARY KEY,

&nbsp;   content TEXT,

&nbsp;   embedding vector(768)

);



CREATE INDEX documents\_embedding\_idx

ON documents

USING ivfflat (embedding vector\_cosine\_ops)

WITH (lists = 100);

```



---



\## Running the Backend



Create and activate a virtual environment:



```bash

python -m venv venv

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install fastapi uvicorn psycopg2 requests

```



Start the API server:



```bash

uvicorn main:app --reload

```



Open Swagger UI:



```

http://127.0.0.1:8000/docs

```



---



\## LM Studio Setup



1\. Load an embedding model (example: nomic-embed-text-v1.5)

2\. Enable OpenAI-compatible API

3\. Default embedding endpoint used:



```

http://127.0.0.1:1234/v1/embeddings

```



---



\## API Endpoints



\### Ingest Document



POST `/ingest`



```json

{

&nbsp; "content": "Postgres with pgvector is excellent for RAG systems"

}

```



---



\### Semantic Search



POST `/search`



```json

{

&nbsp; "query": "What is pgvector used for?",

&nbsp; "top\_k": 3

}

```



---



\## Embedding Test Script



Run:



```bash

python test\_embedding.py

```



Expected output:



```

Embedding length: 768

```



---



\## Why This Is Production-Style



\- Uses PostgreSQL instead of an in-memory vector store

\- Clean API boundaries with FastAPI

\- Dockerized infrastructure

\- OpenAI-compatible local model support

\- Easily extendable to chat or agent-based systems



---



\## Author



Francis Gallo  

Built to understand real-world RAG systems from the ground up



