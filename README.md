# Document Q\&A Assistant (RAG) with Evaluation

A retrieval-augmented generation (RAG) system that answers questions about PDF documents, cites the file and page for every answer, and says "I couldn't find that in the documents" when the answer is not there. The project includes a small evaluation framework used to tune the system and to measure where it fails.

**Stack:** Python, pypdf, LangChain text splitters, sentence-transformers, ChromaDB, rank-bm25, Gemini API (free tier)

## How it works

```
PDFs -> extract text per page -> split into chunks -> embed (MiniLM) -> store in ChromaDB
Question -> vector search + keyword search (BM25) -> merge rankings -> top-10 chunks -> prompt Gemini -> answer with \[file, page] citations
```

1. `ingest.py` extracts text from each PDF page and splits it into 500-character chunks with 100 characters of overlap, keeping the file name and page number with each chunk.
2. `build\_index.py` embeds the chunks with `all-MiniLM-L6-v2` (runs locally) and stores them in ChromaDB.
3. `rag.py` retrieves candidates with both vector search and BM25 keyword search, merges the two rankings with reciprocal rank fusion, sends the top 10 chunks to Gemini, and asks it to answer using only that context, cite sources, and refuse if the answer is not present.

## Documents used

Four public SEC documents: an SEC press release on ICAP Securities, an SEC administrative order on Thayer Capital Partners, a broker-dealer annual report (Athene Securities, 2014), and a Schedule 13D/A filing. Add your own PDFs to the `data/` folder to use it on other documents.

## Evaluation

I drafted 33 test questions with an AI assistant from the documents and checked a sample of them against the source text. 28 can be answered from the documents (each with a reference answer and the page where the fact appears) and 5 cannot. I used them in three ways.

### 1\. Retrieval tuning (no LLM calls)

A question counts as a hit if one of the retrieved chunks comes from the right file and the right page.

**Chunk size and overlap** (embedding model: all-MiniLM-L6-v2, vector search)

|Chunk size|Overlap|Top-3|Top-4|Top-6|
|-|-|-|-|-|
|1000|200 (original)|86%|89%|96%|
|1000|100|89%|89%|96%|
|1000|300|86%|86%|96%|
|500|100|89%|93%|96%|
|1500|300|86%|86%|100%|

**Embedding models** (chunk size 500, overlap 100)

|Model|Top-3|Top-4|Top-6|
|-|-|-|-|
|all-MiniLM-L6-v2|89%|93%|96%|
|BAAI/bge-small-en-v1.5|89%|93%|96%|
|all-mpnet-base-v2|82%|86%|93%|

Findings:

* Retrieving more chunks (top-6) helped more than any other change.
* Overlap made no clear difference; with 28 questions, one question is about 3.6 points, so differences of one question are noise.
* A larger embedding model did not help on this corpus, so I kept the smallest one.
* 1500-character chunks reached 100% at top-6 but send about three times as much text to the model as 500-character chunks, so I chose 500 / 100.

### 2\. A stricter retrieval metric, and hybrid search

The page-level metric above is generous: a chunk can come from the right page without containing the answer. So I added a stricter check that measures whether the answer text itself is in the retrieved chunks. It is a lower bound, because an exact phrase split across two chunks counts as a miss.

Answer text found in the retrieved chunks (28 answerable questions):

|Method|Top-4|Top-6|Top-10|
|-|-|-|-|
|Vector only|71%|75%|86%|
|Keyword only (BM25)|82%|82%|93%|
|Hybrid (reciprocal rank fusion)|75%|79%|93%|

Keyword search helped on exact terms and numbers. Hybrid gained only 1 to 2 questions over vector-only, which is close to noise with 28 questions. My test questions reuse wording from the documents, which favors keyword matching, so these gains may be smaller on real user questions. I chose hybrid with 10 chunks, which ties the best score and should hold up better when questions are worded differently from the documents.

### 3\. End-to-end results (with Gemini)

All 33 questions were run through the full pipeline with `gemini-3.1-flash-lite`. The first column used vector search with 500-character chunks and the top 6 results. The second used hybrid search (keyword plus vector) with the top 10 results.

|Measure|Vector search, top-6|Hybrid search, top-10|
|-|-|-|
|Answer accuracy (28 answerable questions)|86% (24/28)|96% (27/28)|
|Citation accuracy (a cited page matches an expected page)|82% (23/28)|96% (27/28)|
|Correct refusals (5 unanswerable questions)|100% (5/5)|100% (5/5)|

The second column changed two things at once (hybrid retrieval and 10 chunks instead of 6), so the gain cannot be attributed to either one alone. These are single runs on a small test set that I also used to diagnose the earlier failures, so the scores are likely optimistic compared with new questions.

Answer accuracy was judged by a second Gemini call that compares each answer to the reference answer. I also read a sample of answers myself.



\### 4. Held-out test



After tuning, I wrote 16 new questions (13 answerable, 3 not) that were worded differently from the documents and not used for any tuning, and ran them once with the final setup (hybrid search, top-10):



| Measure | Result |

|---|---|

| Answer accuracy | 100% (13/13) |

| Citation accuracy | 92% (12/13) |

| Correct refusals | 100% (3/3) |



This is a small set drawn from the same four documents, with questions and reference answers drafted with an AI assistant, so it shows the system generalizes to new questions on this corpus, not that it is 100% accurate in general.



### Failure analysis

In the first run (vector search, top-6), four answers were wrong. For all four, I checked what was retrieved:

* Three of them returned "I couldn't find that in the documents". In each case the text containing the answer was not in the chunks sent to the model, so the refusal was the correct behavior for the context it had. The problem was retrieval, not the model.
* One answer was confidently wrong: a Schedule 13D has nearly identical cover pages for nine related companies, and the model reported the share count of a different company. The correct page was not retrieved.

After switching to hybrid retrieval with 10 chunks, the three false refusals were answered correctly. The share-count question for one of the nine entities is still the one failure.

## Known limitations

* Tables spread across pages, and near-duplicate pages from different entities, still cause a retrieval miss.
* The documents were OCR'd or saved from web pages, so some tables have text far from their labels.
* The test set is small (28 answerable questions), was drafted from the same documents the system searches, and was also used to diagnose failures, so scores are likely higher than they would be on real users' questions.
* Answer accuracy is judged by an LLM, which is a rough measure.
* Citation accuracy only checks that one cited page matches an expected page.

## Next steps

* Test on documents of other types and multi-step questions that were not used for tuning.
* Chunking across page boundaries for tables that split between pages.
* A reranker and prompt experiments.
* A Streamlit interface and Docker deployment.

## Setup

Requires Python 3.11 or newer (tested on 3.14).

```bash
git clone https://github.com/thebzeera/rag-project.git
cd rag-project
python -m venv venv
venv\\Scripts\\Activate.ps1        # Windows PowerShell; on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with a Gemini API key (free tier from Google AI Studio):

```
GEMINI\_API\_KEY=your-key-here
```

Put PDFs in the `data/` folder, then build the index and ask a question:

```bash
python build\_index.py
python rag.py "What did the SEC charge the broker with?"
```

## Project files

|File|Purpose|
|-|-|
|`ingest.py`|Loads PDFs and splits them into chunks with file and page metadata|
|`build\_index.py`|Embeds chunks and stores them in ChromaDB|
|`search.py`|Prints the most similar chunks for a question (vector search only)|
|`rag.py`|Full pipeline: hybrid retrieval, prompt Gemini, answer with citations|
|`eval\_questions.json`|The 33 test questions with reference answers and pages|
|`add\_pages.py`|Finds the expected page numbers for each question|
|`eval\_retrieval.py`|Measures retrieval hit rate (right file and page)|
|`eval\_content.py`|Measures whether the answer text itself is in the retrieved chunks|
|`eval\_hybrid.py`|Compares vector, keyword and hybrid retrieval|
|`run\_experiments.py`|Compares chunk sizes, overlaps, top-k and embedding models|
|`eval\_answers.py`|Runs all questions through Gemini and scores answers and refusals|
|`rescore\_citations.py`|Re-scores citations, reading multi-page citations correctly|
|`results\*.md`|Saved experiment tables|



