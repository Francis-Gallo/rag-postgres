import requests
url = "http://127.0.0.1:1234/v1/embeddings"

payload = { 
    "model" : "text-embedding-nomic-embed-text-v1.5",
    "input" : "Postgres with pgvector is great for RAG systems"
}

response = requests.post(url, json=payload)
data = response.json()

print("Embedding length:", len(data["data"][0]["embedding"]))
