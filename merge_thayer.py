import json

with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)
with open("thayer_questions.json", encoding="utf-8") as f:
    new = json.load(f)

existing = {q["question"] for q in questions}
added = 0
for q in new:
    if q["question"] not in existing:
        questions.append(q)
        added += 1

with open("eval_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print("Added", added, "questions. Total:", len(questions))