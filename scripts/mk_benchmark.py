"""构建职业细类匹配基准测试。

输出:
    output/match_benchmark/
    ├── data.json       ← 测试数据（无标签，<1MB）
    ├── train_data.json ← 训练数据（含标签，供候选人训练模型）
    ├── eval.py         ← 独立评分脚本（内嵌答案）
    └── README.md       ← 候选人说明
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_JSON = str(PROJECT_ROOT / "data" / "project-4-at-2026-05-27-01-51-7cceb9ba.json")
OUT_DIR = PROJECT_ROOT / "output" / "match_benchmark"


def normalize_choice(c: str) -> str:
    c = c.strip()
    if "都不属于" in c:
        return "NONE"
    return c.replace("候选", "").strip()


def load_all() -> List[Dict]:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = []
    for t in raw:
        td = t.get("data", {})
        choices = Counter()
        for ann in t.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("from_name") != "best_candidate_choice":
                    continue
                for c in r.get("value", {}).get("choices", []):
                    choices[normalize_choice(c)] += 1
        if not choices:
            continue

        cands = []
        for letter in ["a", "b", "c", "d", "e"]:
            cands.append({
                "letter": letter.upper(),
                "code": td.get(f"candidate_{letter}_code", "") or "",
                "title": td.get(f"candidate_{letter}_title", "") or "",
                "desc": td.get(f"candidate_{letter}_desc", "") or "",
            })

        tasks.append({
            "task_id": str(t["id"]),
            "inner_id": t.get("inner_id", 0),
            "job_title": td.get("job_title", "") or "",
            "job_requirements": td.get("job_requirements_clean", "") or "",
            "candidates": cands,
            "label": choices.most_common(1)[0][0],
            "num_annotators": len(t.get("annotations", [])),
        })
    return tasks


def split_train_test(tasks: List[Dict]) -> tuple:
    rng = random.Random(42)

    gold = [t for t in tasks if 1 <= t["inner_id"] <= 30]
    gold_ids = {g["task_id"] for g in gold}

    pool = [t for t in tasks if t["task_id"] not in gold_ids]
    multi = [t for t in pool if t["num_annotators"] > 1]
    multi_ids = {t["task_id"] for t in multi}

    single = [t for t in pool if t["num_annotators"] == 1]
    by_label: Dict[str, List] = {}
    for t in single:
        by_label.setdefault(t["label"], []).append(t)

    sampled = []
    targets = {"NONE": 50, "A": 30, "B": 30, "C": 30, "D": 30, "E": 30}
    for label, n in targets.items():
        available = by_label.get(label, [])
        sampled.extend(rng.sample(available, min(n, len(available))))

    test = gold + multi + sampled
    test_ids: Set[str] = {t["task_id"] for t in test}
    train = [t for t in tasks if t["task_id"] not in test_ids]
    assert len({t["task_id"] for t in train} & test_ids) == 0
    return train, test


def make_data_files(train: List[Dict], test: List[Dict]):
    # 测试数据（无标签）
    test_data = []
    for t in test:
        test_data.append({
            "task_id": t["task_id"],
            "job_title": t["job_title"],
            "job_requirements": t["job_requirements"],
            "candidates": t["candidates"],
        })

    path = OUT_DIR / "data.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"data.json → {path}  ({len(test_data)}条, {size_mb:.1f}MB)")

    # 训练数据（含标签）
    train_data = []
    for t in train:
        train_data.append({
            "task_id": t["task_id"],
            "job_title": t["job_title"],
            "job_requirements": t["job_requirements"],
            "candidates": t["candidates"],
            "label": t["label"],
        })

    path = OUT_DIR / "train_data.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"train_data.json → {path}  ({len(train_data)}条, {size_mb:.1f}MB)")

    # 标签分布
    labels = Counter(t["label"] for t in test)
    print(f"test labels:  {dict(sorted(labels.items()))}")
    print(f"重叠: 0 条")


def make_eval_py(test: List[Dict]):
    answers = {t["task_id"]: t["label"] for t in test}
    answers_json = json.dumps(answers, ensure_ascii=False)

    script = f'''"""职业细类匹配评分脚本。内嵌正确答案，独立运行，不依赖项目代码。

用法:
    python eval.py predictions.json
"""

import json
import sys
import numpy as np

ANSWERS = {answers_json}


def bootstrap_ci(corrects, n_bootstrap=2000):
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(corrects), len(corrects), replace=True)
        means.append(np.array(corrects)[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    if len(sys.argv) < 2:
        print("用法: python eval.py predictions.json")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        preds = json.load(f)

    pred_map = {{p["task_id"]: p["prediction"] for p in preds}}
    classes = ["A", "B", "C", "D", "E", "NONE"]

    correct = 0
    corrects = []
    none_hit = 0
    none_true_count = 0
    none_pred_count = 0
    per_tp = {{c: 0 for c in classes}}
    per_support = {{c: 0 for c in classes}}

    for tid, label in ANSWERS.items():
        per_support[label] += 1
        pred = pred_map.get(tid, "?")
        if pred == label:
            correct += 1
            corrects.append(1)
            per_tp[label] += 1
            if label == "NONE":
                none_hit += 1
        else:
            corrects.append(0)
        if label == "NONE":
            none_true_count += 1
        if pred == "NONE":
            none_pred_count += 1

    total = len(ANSWERS)
    acc = correct / total
    ci_low, ci_high = bootstrap_ci(corrects)
    nr = none_hit / none_true_count if none_true_count else 0
    np_ = none_hit / none_pred_count if none_pred_count else 0

    print(f"Accuracy:       {{acc:.2%}}")
    print(f"95% CI:         {{ci_low:.2%}} - {{ci_high:.2%}}")
    print(f"Total:          {{total}}")
    print(f"")
    print(f"NONE Recall:    {{nr:.2%}} ({{none_hit}}/{{none_true_count}})")
    print(f"NONE Precision: {{np_:.2%}} ({{none_hit}}/{{none_pred_count}})")
    print(f"")
    print("Per-class Recall:")
    for cls in classes:
        s = per_support[cls]
        r = per_tp[cls] / s if s else 0
        print(f"  {{cls}}: {{r:.2%}} ({{per_tp[cls]}}/{{s}})")


if __name__ == "__main__":
    main()
'''

    path = OUT_DIR / "eval.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"eval.py → {path}  (内嵌{len(answers)}条答案)")


def make_readme():
    md = """# 职业细类匹配 模型考核

## 文件说明

| 文件 | 用途 |
|------|------|
| `data.json` | 测试数据，需预测的 495 个岗位 |
| `train_data.json` | 训练数据，18,111 条含标签，供训练模型 |
| `eval.py` | 评分脚本 |

## 数据格式

`data.json` 每条记录包含岗位名称、岗位要求、5 个候选职业（代码+名称+定义）：

```
{
  "task_id": "46437",
  "job_title": "医学信息专员",
  "job_requirements": "1、医学、药学相关专业...",
  "candidates": [
    {"letter": "A", "code": "2-05-01-18", "title": "药学技术人员", "desc": "从事药物..."},
    ...
  ]
}
```

`train_data.json` 格式相同，额外包含 `"label": "B"` 字段。

## 提交格式

对 `data.json` 中每条数据预测一个候选，输出 JSON 数组：

```json
[
  {"task_id": "46437", "prediction": "B"},
  ...
]
```

`prediction` 取值为 A、B、C、D、E、NONE。

## 评分

```bash
python eval.py 你的预测.json
```
"""
    path = OUT_DIR / "README.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"README.md → {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_all()
    print(f"全部任务: {len(tasks)}")

    train, test = split_train_test(tasks)
    print(f"训练: {len(train)}  测试: {len(test)}  重叠: 0\n")

    make_data_files(train, test)
    make_eval_py(test)
    make_readme()

    print(f"\n完成 → {OUT_DIR}")
    print(f"发给候选人: data.json + train_data.json + README.md")
    print(f"评分命令:   python eval.py <name>.json")


if __name__ == "__main__":
    main()
