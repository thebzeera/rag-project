import json
import re
import unicodedata
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).lower()


def tokens(s):
    s = re.sub(r"(?<=\d),(?=\d)", "", norm(s))  # 7,288,915 -> 7288915
    return re.findall(r"\w+", s)


model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
data = col.get(include=["documents", "metadatas"])
ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
pos = {i: n for n, i in enumerate(ids)}
bm25 = BM25Okapi([tokens(d) for d in docs])

qs = [q for q in json.load(open("eval_questions.json", encoding="utf-8")) if q["answerable"] and q.get("pages")]
POOL, RRF_K, KS = 50, 60, [4, 6, 10]


def vector_rank(question):
    res = col.query(query_embeddings=model.encode([question]).tolist(), n_results=POOL)
    return [pos[i] for i in res["ids"][0]]


def bm25_rank(question):
    scores = bm25.get_scores(tokens(question))
    return sorted(range(len(docs)), key=lambda n: -scores[n])[:POOL]


def fuse(*rankings):
    score = {}
    for r in rankings:
        for rank, n in enumerate(r):
            score[n] = score.get(n, 0) + 1 / (RRF_K + rank + 1)
    return sorted(score, key=lambda n: -score[n])


methods = {"vector only": lambda q: vector_rank(q),
           "keyword only": lambda q: bm25_rank(q),
           "hybrid": lambda q: fuse(vector_rank(q), bm25_rank(q))}

for name, fn in methods.items():
    content = {k: 0 for k in KS}
    page = {k: 0 for k in KS}
    missed6 = []
    for q in qs:
        order = fn(q["question"])
        ev = norm(q["evidence"])
        for k in KS:
            top = order[:k]
            if ev in norm(" ".join(docs[n] for n in top)):
                content[k] += 1
            elif k == 6:
                missed6.append(q["question"][:70])
            if any(metas[n]["source"] == q["source_file"] and metas[n]["page"] in q["pages"] for n in top):
                page[k] += 1
    n = len(qs)
    print(f"\n{name}")
    for k in KS:
        print(f"  top-{k}: answer text {content[k]}/{n} = {content[k]/n:.0%} | file+page {page[k]}/{n} = {page[k]/n:.0%}")
    if name == "hybrid":
        print("  missed answer text at top-6:")
        for m in missed6:
            print("   -", m)