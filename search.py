import sys
import chromadb
from sentence_transformers import SentenceTransformer

question = " ".join(sys.argv[1:]) or "What is this document about?"

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

results = collection.query(
    query_embeddings=model.encode([question]).tolist(), n_results=3
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"--- {meta['source']} (page {meta['page']})")
    print(doc[:300].replace("\n", " "))
    print()