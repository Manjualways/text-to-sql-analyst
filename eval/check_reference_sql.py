
import json

from eval.eval import EVAL_SET_PATH, run_reference_sql


with EVAL_SET_PATH.open("r", encoding="utf-8") as file:
    questions = json.load(file)

errors = []

for item in questions:
    try:
        run_reference_sql(item["expected_sql"])
        print(f"[OK] Question {item['id']}: {item['question']}")

    except Exception as exc:
        errors.append(
            (item["id"], item["question"], str(exc))
        )
        print(f"[ERROR] Question {item['id']}: {exc}")

print("\n========== REFERENCE SQL VALIDATION ==========")
print(f"Total questions: {len(questions)}")
print(f"Valid queries:   {len(questions) - len(errors)}")
print(f"Failed queries:  {len(errors)}")

if errors:
    print("\nQueries that need fixing:")
    for question_id, question, error in errors:
        print(f"{question_id}. {question}")
        print(f"   Error: {error}")
