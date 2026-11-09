import json
import chromadb
from sentence_transformers import SentenceTransformer

K = 4
model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
questions = json.load(open("eval_questions.json", encoding="utf-8"))

for q in questions:
    if not q["answerable"]:
        continue
    if not q.get("pages"):
        print("NO PAGES:", q["question"])
        continue
    res = collection.query(
        query_embeddings=model.encode([q["question"]]).tolist(), n_results=K
    )
    metas = res["metadatas"][0]
    ok = any(
        m["source"] == q["source_file"] and m["page"] in q["pages"] for m in metas
    )
    if not ok:
        print("\nMISS:", q["question"])
        print("  expected:", q["source_file"][:40], "pages", q["pages"])
        for m in metas:
            print("  got:", m["source"][:40], "page", m["page"])