import json

for r in json.load(open("eval_answers.json", encoding="utf-8")):
    if r.get("correct") is False or r.get("citation_ok") is False:
        print("\nQ:", r["question"])
        print("   correct:", r.get("correct"), "| citation ok:", r.get("citation_ok"))
        print("   ANSWER:", r["answer_given"][:400].replace("\n", " "))