import json
import sys
import chromadb
from sentence_transformers import SentenceTransformer

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

total = file_hits = page_hits = 0
misses = []

for q in questions:
    if not q["answerable"] or not q.get("pages"):
        continue
    total += 1
    res = collection.query(
        query_embeddings=model.encode([q["question"]]).tolist(), n_results=K
    )
    metas = res["metadatas"][0]
    file_ok = any(m["source"] == q["source_file"] for m in metas)
    page_ok = any(
        m["source"] == q["source_file"] and m["page"] in q["pages"] for m in metas
    )
    file_hits += file_ok
    page_hits += page_ok
    if not page_ok:
        misses.append(q["question"])

print(f"Top-{K} retrieval, {total} answerable questions")
print(f"Right file:         {file_hits}/{total} = {file_hits / total:.0%}")
print(f"Right file + page:  {page_hits}/{total} = {page_hits / total:.0%}")
if misses:
    print("\nMissed questions:")
    for m in misses:
        print(" -", m)