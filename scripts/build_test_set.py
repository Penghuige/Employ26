"""构建职业细类匹配考核测试集。

从 18611 条标注数据中划分出测试集和参考答案。
测试集（交给候选人）不含标签，参考答案（批改用）含标签。

分层策略:
- Tier 1 (Gold):   30 条验证集 → 最可靠，每位标注员都标过
- Tier 2 (Silver): ~265 条多标注员数据 → 扩展覆盖
- Tier 3 (Bronze): ~200 条分层抽样 → 补充 NONE 和稀有类别

输出:
    output/test_set/test_data.json       ← 发给候选人的考卷（无标签）
    output/test_set/answer_key.json      ← 参考答案（批改用，不公开）
    output/test_set/test_set_manifest.md ← 考卷说明文档

用法:
    python scripts/build_test_set.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_JSON = str(PROJECT_ROOT / "data" / "project-4-at-2026-05-27-01-51-7cceb9ba.json")
OUTPUT_DIR = PROJECT_ROOT / "output" / "test_set"


def normalize_choice(c: str) -> str:
    c = c.strip()
    if "都不属于" in c:
        return "NONE"
    return c.replace("候选", "").strip()


def load_all_tasks() -> List[Dict]:
    """加载全部标注任务并整理为标准格式。"""
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = []
    for t in raw:
        inner = t.get("inner_id", 0)
        ann_count = len(t.get("annotations", []))
        td = t.get("data", {})

        # 多数票（只统计 best_candidate_choice）
        choices = Counter()
        for ann in t.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("from_name") != "best_candidate_choice":
                    continue
                for c in r.get("value", {}).get("choices", []):
                    choices[normalize_choice(c)] += 1
        if not choices:
            continue
        top = choices.most_common(1)[0]
        majority = top[0]
        agreement = top[1] / choices.total()

        # 5 个候选
        candidates = []
        for letter in ["a", "b", "c", "d", "e"]:
            candidates.append({
                "letter": letter.upper(),
                "code": td.get(f"candidate_{letter}_code", "") or "",
                "title": td.get(f"candidate_{letter}_title", "") or "",
                "desc": td.get(f"candidate_{letter}_desc", "") or "",
            })

        tasks.append({
            "task_id": t["id"],
            "inner_id": inner,
            "job_title": td.get("job_title", "") or "",
            "job_requirements": td.get("job_requirements_clean", "") or "",
            "candidates": candidates,
            "label": majority,
            "agreement": agreement,
            "num_annotators": ann_count,
            "sample_source": td.get("sample_source", ""),
        })

    return tasks


def select_test_set(tasks: List[Dict]) -> List[Dict]:
    """按分层策略选取测试集。

    Tier 1: inner_id 1-30 (Gold, 多标注员验证集)
    Tier 2: 非 Gold 且 annotators > 1 (Silver, 2 人以上标注)
    Tier 3: 从剩余数据分层抽样 (Bronze, 补充 NONE 和类别平衡)
    """
    gold = [t for t in tasks if 1 <= t["inner_id"] <= 30]
    gold_ids = {t["task_id"] for t in gold}

    remaining_1 = [t for t in tasks if t["task_id"] not in gold_ids]
    silver = [t for t in remaining_1 if t["num_annotators"] > 1]
    silver_ids = {t["task_id"] for t in silver}

    # Tier 3: 分层抽样
    pool = [t for t in remaining_1 if t["task_id"] not in silver_ids]
    # 按 label 分组
    by_label: Dict[str, List[Dict]] = {}
    for t in pool:
        by_label.setdefault(t["label"], []).append(t)

    bronze = []
    # 每个类别至少 20 条，NONE 至少 40 条
    target_per_class = {"NONE": 50, "A": 25, "B": 25, "C": 25, "D": 25, "E": 25}
    for label, target in target_per_class.items():
        available = by_label.get(label, [])
        import random
        rng = random.Random(42)
        selected = rng.sample(available, min(target, len(available)))
        bronze.extend(selected)

    test_set = gold + silver + bronze

    # 去重
    seen = set()
    unique = []
    for t in test_set:
        if t["task_id"] not in seen:
            seen.add(t["task_id"])
            unique.append(t)

    return unique


def build_test_data(test_set: List[Dict]) -> List[Dict]:
    """构建发给候选人的考卷（无标签）。"""
    return [
        {
            "test_id": f"TSK-{t['task_id']}",
            "job_title": t["job_title"],
            "job_requirements": t["job_requirements"],
            "candidates": [
                {
                    "letter": c["letter"],
                    "code": c["code"],
                    "title": c["title"],
                    "desc": c["desc"],
                }
                for c in t["candidates"]
            ],
            "tier": (
                "gold" if t["inner_id"] >= 1 and t["inner_id"] <= 30
                else "silver" if t["num_annotators"] > 1
                else "bronze"
            ),
        }
        for t in test_set
    ]


def build_training_set(all_tasks: List[Dict], test_set_ids: set) -> List[Dict]:
    """构建训练集（含标签），排除已在测试集中的任务。"""
    train = []
    for t in all_tasks:
        if t["task_id"] in test_set_ids:
            continue
        train.append({
            "task_id": t["task_id"],
            "job_title": t["job_title"],
            "job_requirements": t["job_requirements"],
            "candidates": [
                {
                    "letter": c["letter"],
                    "code": c["code"],
                    "title": c["title"],
                    "desc": c["desc"],
                }
                for c in t["candidates"]
            ],
            "label": t["label"],
        })
    return train


def build_answer_key(test_set: List[Dict]) -> List[Dict]:
    """构建参考答案（批改用，含标签和元数据）。"""
    return [
        {
            "test_id": f"TSK-{t['task_id']}",
            "task_id": t["task_id"],
            "label": t["label"],
            "agreement": t["agreement"],
            "num_annotators": t["num_annotators"],
            "tier": (
                "gold" if t["inner_id"] >= 1 and t["inner_id"] <= 30
                else "silver" if t["num_annotators"] > 1
                else "bronze"
            ),
        }
        for t in test_set
    ]


def build_manifest(test_set: List[Dict], output_path: str) -> None:
    """生成考卷说明文档。"""
    gold = [t for t in test_set if 1 <= t["inner_id"] <= 30]
    silver = [t for t in test_set if t["num_annotators"] > 1 and not (1 <= t["inner_id"] <= 30)]
    bronze = [t for t in test_set if t["num_annotators"] <= 1 and not (1 <= t["inner_id"] <= 30)]

    labels = Counter(t["label"] for t in test_set)
    gold_labels = Counter(t["label"] for t in gold)
    silver_labels = Counter(t["label"] for t in silver)
    bronze_labels = Counter(t["label"] for t in bronze)

    md = f"""# 职业细类匹配 候选人考核

## 测试集概述

| 分层 | 数量 | 标注员 | 可靠性 | 权重 |
|------|------|--------|--------|------|
| Gold | {len(gold)} | 12-20 | 极高 | 40% |
| Silver | {len(silver)} | 2+ | 高 | 35% |
| Bronze | {len(bronze)} | 1 | 参考 | 25% |
| **总计** | **{len(test_set)}** | | | |

## 标签分布

| 类别 | 总数 | Gold | Silver | Bronze |
|------|------|------|--------|--------|
"""
    for cls in ["A", "B", "C", "D", "E", "NONE"]:
        md += f"| {cls} | {labels.get(cls, 0)} | {gold_labels.get(cls, 0)} | {silver_labels.get(cls, 0)} | {bronze_labels.get(cls, 0)} |\n"

    md += f"""
## 考核要求

### 输入
`test_data.json` 中的每条记录包含:
- `test_id`: 任务编号
- `job_title`: 岗位名称
- `job_requirements`: 岗位要求描述
- `candidates`: 5 个候选职业 (A-E)，每个含 `code`, `title`, `desc`

### 输出格式
提交一个 JSON 数组，每条一行：
```json
{{"test_id": "TSK-46437", "prediction": "A", "confidence": 0.95, "reasoning": "理由"}}
```

`prediction` 必须是 "A"、"B"、"C"、"D"、"E"、"NONE" 之一。

## 评分标准

| 指标 | 权重 | 计算方式 |
|------|------|----------|
| Weighted Accuracy | 100% | Σ(tier_weight × correct) / Σ(tier_weight) |
| NONE Recall | 加分项 | 人工NONE中正确预测NONE的比例 |
| NONE Precision | 加分项 | 预测NONE中确实是NONE的比例 |

Tier 权重: Gold=3, Silver=2, Bronze=1

## 公平性承诺

- 所有候选人收到的 `test_data.json` 完全一致
- 参考答案 `answer_key.json` 考核结束前不公开
- 评分脚本 `eval_submission.py` 开源，可独立验证
- 测试集可通过 `build_test_set.py` 重新生成（相同随机种子）
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Manifest saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_all_tasks()
    print(f"全部标注任务: {len(tasks)}")

    test_set = select_test_set(tasks)
    print(f"测试集: {len(test_set)} 条 (Gold={sum(1 for t in test_set if 1<=t['inner_id']<=30)}, "
          f"Silver={sum(1 for t in test_set if t['num_annotators']>1 and not (1<=t['inner_id']<=30))}, "
          f"Bronze={sum(1 for t in test_set if t['num_annotators']<=1 and not (1<=t['inner_id']<=30))})")

    # 标签分布
    labels = Counter(t["label"] for t in test_set)
    print(f"标签分布: {dict(labels.most_common())}")

    # 考卷（无标签）
    test_data = build_test_data(test_set)
    test_path = OUTPUT_DIR / "test_data.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"考卷已生成: {test_path} ({len(test_data)} 条，无标签)")

    # 训练集（含标签，排除测试集）
    test_ids = {t["task_id"] for t in test_set}
    train_set = build_training_set(tasks, test_ids)
    train_path = OUTPUT_DIR / "train_data.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    train_labels = Counter(t["label"] for t in train_set)
    print(f"训练集已生成: {train_path} ({len(train_set)} 条，含标签)")
    print(f"  标签分布: {dict(train_labels.most_common())}")
    print(f"  与测试集重叠: {len(test_ids & {t['task_id'] for t in train_set})} 条 (应为0)")

    # 数据完整性校验
    checksum = hashlib.md5(
        json.dumps(test_data, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    print(f"考卷 MD5: {checksum}")

    # 参考答案（含标签，不公开）
    answer_key = build_answer_key(test_set)
    answer_path = OUTPUT_DIR / "answer_key.json"
    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, ensure_ascii=False, indent=2)
    print(f"参考答案已生成: {answer_path} (含标签，请勿发给候选人)")

    # 考卷说明
    manifest_path = OUTPUT_DIR / "test_set_manifest.md"
    build_manifest(test_set, str(manifest_path))

    # 保存配置（含校验和，防止篡改）
    import datetime
    config = {
        "version": "2.0",
        "created_at": datetime.datetime.now().isoformat(),
        "test_count": len(test_set),
        "train_count": len(train_set),
        "checksum_md5": checksum,
        "random_seed": 42,
        "tiers": {
            "gold": {"count": sum(1 for t in test_set if 1<=t['inner_id']<=30), "weight": 3},
            "silver": {"count": sum(1 for t in test_set if t['num_annotators']>1 and not (1<=t['inner_id']<=30)), "weight": 2},
            "bronze": {"count": sum(1 for t in test_set if t['num_annotators']<=1 and not (1<=t['inner_id']<=30)), "weight": 1},
        },
        "label_distribution": {k: v for k, v in labels.most_common()},
    }
    config_path = OUTPUT_DIR / "test_set_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
