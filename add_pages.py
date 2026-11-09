import json
import re
from pathlib import Path
from pypdf import PdfReader


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).lower()


pdfs = {}
for p in Path("data").glob("*.pdf"):
    pdfs[p.name] = [norm(pg.extract_text() or "") for pg in PdfReader(p).pages]

with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

for q in questions:
    if not q["answerable"]:
        continue
    match = [n for n in pdfs if q["source"].lower() in n.lower()]
    if len(match) != 1:
        print("SOURCE PROBLEM:", q["question"])
        continue
    ev = norm(q["evidence"])
    pages = [i + 1 for i, t in enumerate(pdfs[match[0]]) if ev in t]
    q["source_file"] = match[0]
    q["pages"] = pages
    if not pages:
        print("NOT FOUND:", q["question"])

with open("eval_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print("Done. Questions:", len(questions))