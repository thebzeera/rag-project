import json
import re
import time
from rag import answer, client, MODEL

PAUSE = 6  # seconds between API calls, to stay under free-tier limits


def call_with_retry(fn, tries=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg:
                print("   rate limit or busy, waiting 40s...")
                time.sleep(40)
            else:
                raise
    return None


def judge(question, reference, model_answer):
    prompt = (
        "You are grading an answer against a reference answer.\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Model answer: {model_answer}\n\n"
        "Is the model answer factually consistent with the reference answer "
        "and does it contain the key facts? Reply with exactly one word: "
        "CORRECT or INCORRECT."
    )
    r = call_with_retry(lambda: client.models.generate_content(model=MODEL, contents=prompt))
    text = (r.text or "").strip().upper() if r else ""
    return text.startswith("CORRECT")


def refused(text):
    t = (text or "").lower().replace("’", "'")
    return "find that in the documents" in t


questions = json.load(open("eval_questions.json", encoding="utf-8"))
results = []
n_ans = ok_ans = n_cite = ok_cite = n_unans = ok_unans = 0

for i, q in enumerate(questions, start=1):
    print(f"[{i}/{len(questions)}] {q['question'][:70]}")
    reply = call_with_retry(lambda: answer(q["question"])) or ""
    time.sleep(PAUSE)
    rec = {"question": q["question"], "answer_given": reply}

    if q["answerable"]:
        n_ans += 1
        correct = judge(q["question"], q["answer"], reply)
        time.sleep(PAUSE)
        cited = {int(p) for p in re.findall(r"page\s*(\d+)", reply, flags=re.I)}
        cite_ok = bool(cited & set(q.get("pages", [])))
        n_cite += 1
        ok_ans += correct
        ok_cite += cite_ok
        rec.update({"correct": correct, "citation_ok": cite_ok})
        print("   correct:", correct, "| citation ok:", cite_ok)
    else:
        n_unans += 1
        r = refused(reply)
        ok_unans += r
        rec.update({"correctly_refused": r})
        print("   correctly refused:", r)
    results.append(rec)

json.dump(results, open("eval_answers.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("\n=== SUMMARY ===")
print(f"Answer accuracy:    {ok_ans}/{n_ans} = {ok_ans / n_ans:.0%}")
print(f"Citation accuracy:  {ok_cite}/{n_cite} = {ok_cite / n_cite:.0%}")
print(f"Correct refusals:   {ok_unans}/{n_unans} = {ok_unans / n_unans:.0%}")
print("Details saved to eval_answers.json")