import re
import sys
import unicodedata
import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
TOP_K = 10   # chunks sent to the model
POOL = 50    # candidates taken from each search before merging
RRF_K = 60   # reciprocal rank fusion constant

embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
client = genai.Client()

_data = collection.get(include=["documents", "metadatas"])
IDS, DOCS, METAS = _data["ids"], _data["documents"], _data["metadatas"]
POS = {i: n for n, i in enumerate(IDS)}


def _tokens(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)  # 7,288,915 -> 7288915
    return re.findall(r"\w+", s)


BM25 = BM25Okapi([_tokens(d) for d in DOCS])


def retrieve(question, k=TOP_K):
    res = collection.query(
        query_embeddings=embedder.encode([question]).tolist(), n_results=POOL
    )
    vec = [POS[i] for i in res["ids"][0]]
    scores = BM25.get_scores(_tokens(question))
    kw = sorted(range(len(DOCS)), key=lambda n: -scores[n])[:POOL]
    fused = {}
    for ranking in (vec, kw):
        for rank, n in enumerate(ranking):
            fused[n] = fused.get(n, 0) + 1 / (RRF_K + rank + 1)
    top = sorted(fused, key=lambda n: -fused[n])[:k]
    return [(DOCS[n], METAS[n]) for n in top]


def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{m['source']}, page {m['page']}]\n{doc}" for doc, m in hits
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite sources as [file, page]. If the answer is not in the context, "
        "say: I couldn't find that in the documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    print(answer(q))