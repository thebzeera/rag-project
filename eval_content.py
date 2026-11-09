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
qs = [q for q in json.load(open("eval_questions.json", encoding="utf-8")) if q["answerable"]]

KS = [4, 6, 10]
hits = {k: 0 for k in KS}
missed = {k: [] for k in KS}
for q in qs:
    docs = col.query(query_embeddings=model.encode([q["question"]]).tolist(), n_results=max(KS))["documents"][0]
    ev = norm(q["evidence"])
    for k in KS:
        if ev in norm(" ".join(docs[:k])):
            hits[k] += 1
        else:
            missed[k].append(q["question"][:70])

for k in KS:
    print(f"Answer text in top-{k}: {hits[k]}/{len(qs)} = {hits[k] / len(qs):.0%}")
print("\nMissed at top-6:")
for m in missed[6]:
    print(" -", m)