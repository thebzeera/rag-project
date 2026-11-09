import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_chunks

chunks = load_chunks()
print(f"Loaded {len(chunks)} chunks")

if not chunks:
    raise SystemExit("No chunks found. Check the PDFs in the data folder.")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    [c["text"] for c in chunks], show_progress_bar=True
).tolist()

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("docs")
except Exception:
    pass
collection = client.create_collection("docs")

BATCH = 500
for i in range(0, len(chunks), BATCH):
    batch = chunks[i : i + BATCH]
    collection.add(
        ids=[f"chunk-{i + j}" for j in range(len(batch))],
        documents=[c["text"] for c in batch],
        embeddings=embeddings[i : i + BATCH],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
    )

print(f"Stored {collection.count()} chunks in Chroma")
