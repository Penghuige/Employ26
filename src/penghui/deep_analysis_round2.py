#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二轮数据集深度分析：per-task majority 和 RAG TopK 命中率指标。

分析维度:
    1. 每任务标注员多数意见（majority vote）
    2. RAG 候选 TopK 命中率（Top-1 至 Top-5）
    3. 多数意见为 "NONE"（以上都不对）的任务比例

用法:
    python -m src.penghui.deep_analysis_round2
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from config.paths import get_project_paths

_project = get_project_paths()
DATA_FILE = os.path.join(str(_project.project_root), "data", "project-4-at-2026-05-27-01-51-7cceb9ba.json")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Parse
task_annotations = {}
for item in raw_data:
    task_id = item["id"]
    ann_list = []
    for ann in item["annotations"]:
        choice = None
        for r in ann["result"]:
            if r["from_name"] == "best_candidate_choice":
                choices = r["value"].get("choices", [])
                if choices:
                    raw = choices[0]
                    if len(raw) >= 2 and raw[-1] in "ABCDE":
                        choice = raw[-1]
                    elif "不" in raw:  # 不
                        choice = "NONE"
        ann_list.append(choice)
    task_annotations[task_id] = {
        "annotations": ann_list,
        "data": item["data"],
    }

def get_majority(task):
    choices = [c for c in task["annotations"] if c is not None]
    if not choices:
        return None
    counter = Counter(choices)
    top, count = counter.most_common(1)[0]
    return top

# == Per-task RAG TopK hit rate ==
topk_hits = defaultdict(int)
tasks_with_majority = 0
tasks_with_none_maj = 0

for tid, task in task_annotations.items():
    maj = get_majority(task)
    if maj is None:
        continue
    if maj == "NONE":
        tasks_with_none_maj += 1
        continue
    tasks_with_majority += 1
    data = task["data"]
    for cand in ["a", "b", "c", "d", "e"]:
        source = data.get(f"candidate_{cand}_source", "")
        for k in [1, 2, 3, 4, 5]:
            if f"top{k}" in source and maj == cand.upper():
                topk_hits[k] += 1
                break

print("=== Per-Task (Majority Vote) RAG TopK Hit Rate ===")
print(f"Tasks with majority (non-NONE): {tasks_with_majority}")
print(f"Tasks with majority=NONE: {tasks_with_none_maj}")
cum = 0
for k in sorted(topk_hits.keys()):
    cum += topk_hits[k]
    pct = cum / tasks_with_majority * 100
    print(f"  Top-{k}: {topk_hits[k]} -> cumulative Top{k}={cum}/{tasks_with_majority} = {pct:.1f}%")

# == Single-annotator tasks only ==
single_tasks = {tid: t for tid, t in task_annotations.items() if len(t["annotations"]) == 1}
single_topk = defaultdict(int)
single_valid = 0
single_none = 0
for tid, task in single_tasks.items():
    choice = task["annotations"][0]
    if choice == "NONE":
        single_none += 1
        continue
    single_valid += 1
    data = task["data"]
    for cand in ["a", "b", "c", "d", "e"]:
        source = data.get(f"candidate_{cand}_source", "")
        if choice == cand.upper():
            for k in [1, 2, 3, 4, 5]:
                if f"top{k}" in source:
                    single_topk[k] += 1
                    break
            break

print(f"\n=== Single-Annotator Tasks RAG TopK ===")
print(f"Valid choices: {single_valid}, NONE: {single_none}")
cum = 0
for k in sorted(single_topk.keys()):
    cum += single_topk[k]
    pct = cum / single_valid * 100
    print(f"  Top-{k}: {single_topk[k]} -> cum Top{k}={cum}/{single_valid} = {pct:.1f}%")

# == Multi-annotator tasks (majority) ==
multi_tasks = {tid: t for tid, t in task_annotations.items() if len(t["annotations"]) >= 2}
multi_topk = defaultdict(int)
multi_valid = 0
multi_none = 0
for tid, task in multi_tasks.items():
    maj = get_majority(task)
    if maj == "NONE":
        multi_none += 1
        continue
    if maj is None:
        continue
    multi_valid += 1
    data = task["data"]
    for cand in ["a", "b", "c", "d", "e"]:
        source = data.get(f"candidate_{cand}_source", "")
        if maj == cand.upper():
            for k in [1, 2, 3, 4, 5]:
                if f"top{k}" in source:
                    multi_topk[k] += 1
                    break
            break

print(f"\n=== Multi-Annotator Tasks (Majority) RAG TopK ===")
print(f"Valid majorities: {multi_valid}, NONE majorities: {multi_none}")
cum = 0
for k in sorted(multi_topk.keys()):
    cum += multi_topk[k]
    if multi_valid > 0:
        pct = cum / multi_valid * 100
        print(f"  Top-{k}: {multi_topk[k]} -> cum Top{k}={cum}/{multi_valid} = {pct:.1f}%")

# == Annotator agreement with majority, excl NONE ==
ann_agree_dist = defaultdict(lambda: {"total": 0, "agree": 0})
for tid, task in multi_tasks.items():
    maj = get_majority(task)
    if maj is None or maj == "NONE":
        continue
    for a in task["annotations"]:
        if a is None or a == "NONE":
            continue
        # We can't track annotator IDs here without re-parsing from raw data
        pass

# == Per-annotation in RAG Top1/2/3 ==
per_ann_topk = defaultdict(int)
per_ann_total = 0
per_ann_none = 0
for tid, task in task_annotations.items():
    data = task["data"]
    for a in task["annotations"]:
        if a is None:
            continue
        if a == "NONE":
            per_ann_none += 1
            continue
        per_ann_total += 1
        for cand in ["a", "b", "c", "d", "e"]:
            if a == cand.upper():
                source = data.get(f"candidate_{cand}_source", "")
                for k in [1, 2, 3, 4, 5]:
                    if f"top{k}" in source:
                        per_ann_topk[k] += 1
                        break
                break

print(f"\n=== Per-Annotation RAG TopK Hit Rate ===")
print(f"Total valid annotations: {per_ann_total}, NONE annotations: {per_ann_none}")
cum = 0
for k in sorted(per_ann_topk.keys()):
    cum += per_ann_topk[k]
    pct = cum / per_ann_total * 100
    print(f"  Top-{k}: {per_ann_topk[k]} -> cum Top{k}={cum}/{per_ann_total} = {pct:.1f}%")

# == Task-level summary ==
total_tasks = len(task_annotations)
print(f"\n=== Task-Level Summary ===")
print(f"Total tasks: {total_tasks}")
print(f"  Multi-annotator: {len(multi_tasks)}")
print(f"  Single-annotator: {len(single_tasks)}")
print(f"Tasks with majority non-NONE: {tasks_with_majority} ({tasks_with_majority/total_tasks*100:.1f}%)")
print(f"Tasks with majority NONE: {tasks_with_none_maj} ({tasks_with_none_maj/total_tasks*100:.1f}%)")

# == Weighted average agreement on multi-annotator tasks ==
# For each task: fraction of annotators agreeing with majority
agree_rates = []
for tid, task in multi_tasks.items():
    choices = [c for c in task["annotations"] if c is not None]
    if len(choices) < 2:
        continue
    counter = Counter(choices)
    top_choice, top_count = counter.most_common(1)[0]
    agree_rates.append(top_count / len(choices))

avg_agree = sum(agree_rates) / len(agree_rates) if agree_rates else 0
print(f"\nAvg majority agreement rate (multi-ann tasks): {avg_agree*100:.1f}%")

# == Key check: what metric is closest to 82.8%? ==
print(f"\n=== CANDIDATES FOR 82.8% ===")
candidates = {}

# A: Per-task majority in Top3 RAG
cum3 = sum(topk_hits[k] for k in [1,2,3])
candidates["A: Task-majority in RAG Top3"] = cum3 / tasks_with_majority * 100

# B: Per-annotation in Top3 RAG
cum3_ann = sum(per_ann_topk[k] for k in [1,2,3])
candidates["B: Per-annotation in RAG Top3"] = cum3_ann / per_ann_total * 100

# C: Single-annotator tasks Top3
single_cum3 = sum(single_topk[k] for k in [1,2,3])
candidates["C: Single-ann tasks in RAG Top3"] = single_cum3 / single_valid * 100

# D: Multi-annotator tasks (majority) Top3
multi_cum3 = sum(multi_topk[k] for k in [1,2,3])
if multi_valid > 0:
    candidates["D: Multi-ann tasks majority in RAG Top3"] = multi_cum3 / multi_valid * 100

# E: Avg majority agreement
candidates["E: Avg majority agreement rate"] = avg_agree * 100

# F: Single-annotator tasks chose A-E (valid)
candidates["F: Single-ann chose A-E (valid)"] = single_valid / (single_valid + single_none) * 100

# G: All annotations chose A-E
total_ann = per_ann_total + per_ann_none
candidates["G: All annotations chose A-E"] = per_ann_total / total_ann * 100

for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - 82.8)):
    diff = abs(val - 82.8)
    marker = " *** CLOSEST ***" if diff < 1.0 else (" **" if diff < 5.0 else "")
    print(f"  {name}: {val:.1f}% (diff={diff:.1f}pp){marker}")
