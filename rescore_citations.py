import json
import re

qs = {q["question"]: q for q in json.load(open("eval_questions.json", encoding="utf-8"))}
ok = n = 0
for r in json.load(open("eval_answers.json", encoding="utf-8")):
    q = qs[r["question"]]
    if not q["answerable"]:
        continue
    cited = set()
    for m in re.findall(r"pages?\s*([\d,\s]+)", r["answer_given"], flags=re.I):
        cited |= {int(x) for x in re.findall(r"\d+", m)}
    good = bool(cited & set(q.get("pages", [])))
    n += 1
    ok += good
    if not good:
        print("not ok:", q["question"][:60], "| cited", sorted(cited), "| expected", q["pages"])
print(f"\nCitation accuracy (fixed parser): {ok}/{n} = {ok / n:.0%}")