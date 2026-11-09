import json
import re
import unicodedata
import chromadb
from sentence_transformers import SentenceTransformer


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).lower()


model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
qs = {q["question"]: q for q in json.load(open("eval_questions.json", encoding="utf-8"))}

for r in json.load(open("eval_answers.json", encoding="utf-8")):
    if r.get("correct") is False:
        q = qs[r["question"]]
        res = col.query(query_embeddings=model.encode([q["question"]]).tolist(), n_results=6)
        ev = norm(q["evidence"])
        found = [i + 1 for i, d in enumerate(res["documents"][0]) if ev in norm(d)]
        print("\nQ:", q["question"][:80])
        print("   evidence phrase in retrieved chunks:", f"YES (rank {found})" if found else "NO")