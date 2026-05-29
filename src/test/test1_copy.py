import json
import os

json_file = r"D:\PythonProjects\Employ26\output\skill_extraction\regression\prompts\flat_skill_regression_dataset.initial_prompts.jsonl"
print(os.getenv("OPENAI_API_KEY"))

with open(json_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # print(data["user_prompt"])
    # print(data["system_prompt"])