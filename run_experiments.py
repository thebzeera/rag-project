import json
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_chunks

with open("eval_questions.json", encoding="utf-8") as f:
    questions = [q for q in json.load(f) if q["answerable"] and q.get("pages")]

# (chunk_size, overlap, embedding model)
CONFIGS = [
    (500, 100, "all-MiniLM-L6-v2"),
    (500, 100, "BAAI/bge-small-en-v1.5"),
    (500, 100, "sentence-transformers/all-mpnet-base-v2"),
]
KS = [3, 4, 6]
BATCH = 500

models = {}
client = chromadb.EphemeralClient()
rows = []

for i, (size, overlap, model_name) in enumerate(CONFIGS):
    if model_name not in models:
        models[model_name] = SentenceTransformer(model_name)
    model = models[model_name]

    chunks = load_chunks(chunk_size=size, chunk_overlap=overlap)
    emb = model.encode([c["text"] for c in chunks]).tolist()

    col = client.create_collection(f"exp{i}")
    for s in range(0, len(chunks), BATCH):
        b = chunks[s : s + BATCH]
        col.add(
            ids=[f"c{s + j}" for j in range(len(b))],
            documents=[c["text"] for c in b],
            embeddings=emb[s : s + BATCH],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in b],
        )

    file_hits = {k: 0 for k in KS}
    page_hits = {k: 0 for k in KS}
    q_emb = model.encode([q["question"] for q in questions]).tolist()
    for q, qe in zip(questions, q_emb):
        res = col.query(query_embeddings=[qe], n_results=max(KS))
        metas = res["metadatas"][0]
        for k in KS:
            top = metas[:k]
            if any(m["source"] == q["source_file"] for m in top):
                file_hits[k] += 1
            if any(m["source"] == q["source_file"] and m["page"] in q["pages"] for m in top):
                page_hits[k] += 1

    n = len(questions)
    for k in KS:
        row = (
            f"| {size} | {overlap} | {model_name} | {k} | "
            f"{page_hits[k] / n:.0%} ({page_hits[k]}/{n}) | "
            f"{file_hits[k] / n:.0%} ({file_hits[k]}/{n}) | {len(chunks)} |"
        )
        rows.append(row)
        print(row)

header = [
    "| Chunk size | Overlap | Embedding model | Top-k | File + page hit | File hit | Chunks |",
    "|---|---|---|---|---|---|---|",
]
with open("results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(header + rows) + "\n")
print("\nSaved results.md")