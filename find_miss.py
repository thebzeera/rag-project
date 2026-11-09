import json
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_chunks

CHUNK, OVERLAP, K = 500, 100, 6

questions = [
    q
    for q in json.load(open("eval_questions.json", encoding="utf-8"))
    if q["answerable"] and q.get("pages")
]

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = load_chunks(chunk_size=CHUNK, chunk_overlap=OVERLAP)
emb = model.encode([c["text"] for c in chunks]).tolist()

col = chromadb.EphemeralClient().create_collection("test")
for s in range(0, len(chunks), 500):
    b = chunks[s : s + 500]
    col.add(
        ids=[f"c{s + j}" for j in range(len(b))],
        documents=[c["text"] for c in b],
        embeddings=emb[s : s + 500],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in b],
    )

missed = 0
for q in questions:
    res = col.query(query_embeddings=model.encode([q["question"]]).tolist(), n_results=K)
    metas = res["metadatas"][0]
    ok = any(m["source"] == q["source_file"] and m["page"] in q["pages"] for m in metas)
    if not ok:
        missed += 1
        print("\nMISSED:", q["question"])
        print("  expected:", q["source_file"][:40], "pages", q["pages"])
        for m in metas:
            print("  got:", m["source"][:40], "page", m["page"])

print(f"\n{missed} missed out of {len(questions)}")