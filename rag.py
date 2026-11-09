import sys
import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL = "gemini-3.1-flash-lite"  # swap for a current free model name if rejected
TOP_K = 6

embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
client = genai.Client()


def retrieve(question, k=TOP_K):
    results = collection.query(
        query_embeddings=embedder.encode([question]).tolist(), n_results=k
    )
    return list(zip(results["documents"][0], results["metadatas"][0]))


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
