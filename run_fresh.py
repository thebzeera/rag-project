import json
import re
import time
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from rag import answer, client, MODEL

PAUSE = 6


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).lower()


pdfs = {p.name: [norm(pg.extract_text() or "") for pg in PdfReader(p).pages]
        for p in Path("data").glob("*.pdf")}
questions = json.load(open("eval_fresh.json", encoding="utf-8"))

bad = 0
for q in questions:
    if not q["answerable"]:
        continue
    match = [n for n in pdfs if q["source"].lower() in n.lower()]
    if len(match) != 1:
        print("SOURCE PROBLEM:", q["question"])
        bad += 1
        continue
    ev = norm(q["evidence"])
    q["pages"] = [i + 1 for i, t in enumerate(pdfs[match[0]]) if ev in t]
    if not q["pages"]:
        print("NOT FOUND (fix the evidence phrase):", q["question"])
        bad += 1
if bad:
    raise SystemExit("\nFix the problems above, then run again. No API calls were made.")


def call_with_retry(fn, tries=5):
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            m = str(e)
            if "429" in m or "RESOURCE_EXHAUSTED" in m or "503" in m:
                print("   rate limit or busy, waiting 40s...")
                time.sleep(40)
            else:
                raise
    return None


def judge(question, reference, reply):
    prompt = (
        "You are grading an answer against a reference answer.\n"
        f"Question: {question}\nReference answer: {reference}\nModel answer: {reply}\n\n"
        "Is the model answer factually consistent with the reference answer and does it "
        "contain the key facts? Reply with exactly one word: CORRECT or INCORRECT."
    )
    r = call_with_retry(lambda: client.models.generate_content(model=MODEL, contents=prompt))
    return bool(r) and (r.text or "").strip().upper().startswith("CORRECT")


def refused(text):
    return "find that in the documents" in (text or "").lower().replace("’", "'")


results = []
n_a = ok_a = ok_c = n_u = ok_u = 0
for i, q in enumerate(questions, start=1):
    print(f"[{i}/{len(questions)}] {q['question'][:70]}")
    reply = call_with_retry(lambda: answer(q["question"])) or ""
    time.sleep(PAUSE)
    rec = {"question": q["question"], "answer_given": reply}
    if q["answerable"]:
        n_a += 1
        correct = judge(q["question"], q["answer"], reply)
        time.sleep(PAUSE)
        cited = set()
        for m in re.findall(r"pages?\s*([\d,\s]+)", reply, flags=re.I):
            cited |= {int(x) for x in re.findall(r"\d+", m)}
        cite_ok = bool(cited & set(q["pages"]))
        ok_a += correct
        ok_c += cite_ok
        rec.update({"correct": correct, "citation_ok": cite_ok})
        print("   correct:", correct, "| citation ok:", cite_ok)
    else:
        n_u += 1
        r = refused(reply)
        ok_u += r
        rec["correctly_refused"] = r
        print("   correctly refused:", r)
    results.append(rec)

json.dump(results, open("eval_fresh_results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\n=== FRESH TEST SUMMARY ===")
print(f"Answer accuracy:    {ok_a}/{n_a} = {ok_a / n_a:.0%}")
print(f"Citation accuracy:  {ok_c}/{n_a} = {ok_c / n_a:.0%}")
print(f"Correct refusals:   {ok_u}/{n_u} = {ok_u / n_u:.0%}")
print("Details saved to eval_fresh_results.json")
