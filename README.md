# Document Q\&A Assistant (RAG) with Evaluation

A retrieval-augmented generation (RAG) system that answers questions about PDF documents, cites the file and page for every answer, and says "I couldn't find that in the documents" when the answer is not there. The project includes a small evaluation framework used to tune the system and to measure where it fails.

**Stack:** Python, pypdf, LangChain text splitters, sentence-transformers, ChromaDB, Gemini API (free tier)

## How it works

```
PDFs -> extract text per page -> split into chunks -> embed (MiniLM) -> store in ChromaDB
Question -> embed -> retrieve top-6 chunks -> prompt Gemini with the chunks -> answer with \[file, page] citations
```

1. `ingest.py` extracts text from each PDF page and splits it into 500-character chunks with 100 characters of overlap, keeping the file name and page number with each chunk.
2. `build\_index.py` embeds the chunks with `all-MiniLM-L6-v2` (runs locally) and stores them in ChromaDB.
3. `rag.py` retrieves the 6 most similar chunks for a question and asks Gemini to answer using only that context, cite sources, and refuse if the answer is not present.

## Documents used

Four public SEC documents: an SEC press release on ICAP Securities, an SEC administrative order on Thayer Capital Partners, a broker-dealer annual report (Athene Securities, 2014), and a Schedule 13D/A filing. Add your own PDFs to the `data/` folder to use it on other documents.

## Evaluation

I wrote 33 test questions from the documents: 28 that the documents can answer (each with a reference answer and the page where the fact appears) and 5 that they cannot. I used them in two ways.

### 1\. Retrieval tuning (no LLM calls)

A question counts as a hit if one of the retrieved chunks comes from the right file and the right page.

**Chunk size and overlap** (embedding model: all-MiniLM-L6-v2)

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
* 1500-character chunks reached 100% at top-6 but send about three times as much text to the model as 500-character chunks, so I chose 500 / 100 / top-6.



\#### Hybrid search (keyword + vector)



Answer text found in the retrieved chunks (28 answerable questions):



| Method | Top-4 | Top-6 | Top-10 |

|---|---|---|---|

| Vector only | 71% | 75% | 86% |

| Keyword only (BM25) | 82% | 82% | 93% |

| Hybrid (reciprocal rank fusion) | 75% | 79% | 93% |



Keyword search helped on exact terms and numbers. Hybrid gained only 1 to 2 questions over vector-only, which is close to noise with 28 questions. My test questions reuse wording from the documents, which favors keyword matching, so these gains may be smaller on real user questions.

### 2\. End-to-end results (with Gemini)





All 33 questions were run through the full pipeline using the chosen setup (500 / 100 / top-6, `gemini-3.1-flash-lite`).

|Measure|Result|
|-|-|
|Answer accuracy (28 answerable questions)|86% (24/28)|
|Correct refusals (5 unanswerable questions)|100% (5/5)|
|Citation accuracy (a cited page matches an expected page)|82% (23/28), after fixing the parser to read multi-page citations such as "pages 2, 4, 8"|

Answer accuracy was judged by a second Gemini call that compares each answer to the reference answer. I also read a sample of answers myself.

### Failure analysis

Four answers were wrong. For all four, I checked what was retrieved:

* Three of them returned "I couldn't find that in the documents". In each case the text containing the answer was not in the chunks sent to the model, so the refusal was the correct behavior for the context it had. The problem was retrieval, not the model.
* One answer was confidently wrong: a Schedule 13D has nearly identical cover pages for nine related companies, and the model reported the share count of a different company. The correct page was not retrieved.

This also showed that my retrieval metric was too generous. A chunk can come from the right page without containing the answer, so I added a stricter check that measures whether the answer text itself is in the retrieved chunks. It is much lower than the page-level score: 71% (20/28) at top-4, 75% (21/28) at top-6, and 86% (24/28) at top-10. It is a lower bound, because an exact phrase split across two chunks counts as a miss, and three of the seven top-6 misses were still answered correctly. All four wrong answers were among the seven.

## Known limitations

* Tables spread across pages, and near-duplicate pages from different entities, cause retrieval misses and one confidently wrong answer.
* The documents were OCR'd or saved from web pages, so some tables have text far from their labels.
* The test set is small (28 answerable questions) and was written by me from the same documents, so scores are likely higher than they would be on real users' questions.
* Answer accuracy is judged by an LLM, which is a rough measure.
* Citation accuracy only checks that one cited page matches an expected page.

## Next steps

* Hybrid search (keyword plus vector) to handle exact-match items such as numbers and form names.
* Retrieving more chunks (top-10) and chunking across page boundaries for tables that split between pages.
* A larger and more varied test set.
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
|`search.py`|Prints the most similar chunks for a question|
|`rag.py`|Full pipeline: retrieve, prompt Gemini, answer with citations|
|`eval\_questions.json`|The 33 test questions with reference answers and pages|
|`add\_pages.py`|Finds the expected page numbers for each question|
|`eval\_retrieval.py`|Measures retrieval hit rate (right file and page)|
|`eval\_content.py`|Measures whether the answer text itself is in the retrieved chunks<br />|
|`eval\_hybrid.py`|Compares vector, keyword and hybrid retrieval |
|`run\_experiments.py`|Compares chunk sizes, overlaps, top-k and embedding models|
|`eval\_answers.py`|Runs all questions through Gemini and scores answers and refusals|
|`rescore\_citations.py`|Re-scores citations, reading multi-page citations correctly|
|`results\*.md`|Saved experiment tables|



