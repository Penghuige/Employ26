#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复现论文第二轮数据集有效性检测
=================================
对第二轮人工标注数据集进行有效性分析：
1. 标注员间一致性（Inter-Annotator Agreement）
2. RAG候选命中率（TOP1-5）
3. Deepseek-v4-pro 标注准确率对比
4. 验证样本（is_validation_sample=1）专项分析
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime

# ── 路径 ──────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_FILE = os.path.join(BASE, "data", "project-4-at-2026-05-27-01-51-7cceb9ba.json")
DEEPSEEK_FILE = os.path.join(BASE, "output", "deepseek_relabel", "deepseek_relabel_raw.jsonl")
OUTPUT_FILE = os.path.join(BASE, "output", "round2_validity_report.txt")

# ── 辅助函数 ────────────────────────────────────────
def extract_choice(annotation):
    """从标注记录中提取最佳候选选择（返回标准化后的字母A-E或NONE）"""
    for r in annotation.get("result", []):
        if r["from_name"] == "best_candidate_choice":
            choices = r["value"].get("choices", [])
            if not choices:
                return None
            raw = choices[0]
            # 候选A/B/C/D/E -> A/B/C/D/E
            if len(raw) >= 2 and raw[-1] in "ABCDE":
                return raw[-1]
            # 以上选项都不属于 -> NONE
            if "不" in raw:
                return "NONE"
            return raw
    return None


def make_summary(text):
    """生成带时间戳的报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("Round 2 Dataset Validity Reproduction Analysis")
    lines.append(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(text)
    return "\n".join(lines)


# ── 1. 加载数据 ────────────────────────────────────
print("Loading data...")
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

output_lines = []
def log(s):
    print(s)
    output_lines.append(s)

log("=" * 70)
log("Round 2 Dataset Validity Reproduction Analysis")
log(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

# ── 数据集概览 ────────────────────────────────────
log(f"\n[Dataset Overview]")
log(f"  Total tasks: {len(raw_data)}")
total_annotations = sum(len(item["annotations"]) for item in raw_data)
log(f"  Total annotations: {total_annotations}")

# ── 2. 解析标注 ────────────────────────────────────
task_annotations = {}  # task_id -> {annotations: [{annotator, choice}], data: {...}}
validation_tasks = []  # is_validation_sample == 1
for item in raw_data:
    task_id = item["id"]
    data_fields = item["data"]
    ann_list = []
    for ann in item["annotations"]:
        choice = extract_choice(ann)
        ann_list.append({
            "annotator": ann["completed_by"],
            "choice": choice,
        })
    task_annotations[task_id] = {
        "annotations": ann_list,
        "data": data_fields,
    }
    if data_fields.get("is_validation_sample") == "1":
        validation_tasks.append(task_id)

# 分类任务
multi_tasks = {
    tid: t for tid, t in task_annotations.items()
    if len(t["annotations"]) >= 2
}
single_tasks = {
    tid: t for tid, t in task_annotations.items()
    if len(t["annotations"]) == 1
}

log(f"  Tasks with annotations: {len(task_annotations)}")
log(f"  Multi-annotator tasks: {len(multi_tasks)}")
log(f"  Single-annotator tasks: {len(single_tasks)}")
log(f"  Validation-sample tasks (is_validation_sample=1): {len(validation_tasks)}")

# ── 3. 全局选择分布 ──────────────────────────────
all_choices = Counter()
for tid, t in task_annotations.items():
    for a in t["annotations"]:
        all_choices[a["choice"]] += 1

log(f"\n[Global Choice Distribution]")
total = sum(all_choices.values())
for choice in sorted(all_choices.keys(), key=lambda x: all_choices[x], reverse=True):
    log(f"  {choice}: {all_choices[choice]} ({all_choices[choice]/total*100:.1f}%)")

none_count = all_choices.get("NONE", 0)
valid_count = total - none_count
log(f"  Valid choices (A-E): {valid_count} ({valid_count/total*100:.1f}%)")
log(f"  NONE (以上选项都不属于): {none_count} ({none_count/total*100:.1f}%)")

# 单标注任务的选择分布
single_choices = Counter()
for tid, t in single_tasks.items():
    single_choices[t["annotations"][0]["choice"]] += 1
log(f"\n[Single-Annotator Tasks Choice Distribution] ({len(single_tasks)} tasks)")
single_total = sum(single_choices.values())
for choice in sorted(single_choices.keys(), key=lambda x: single_choices[x], reverse=True):
    log(f"  {choice}: {single_choices[choice]} ({single_choices[choice]/single_total*100:.1f}%)")
single_valid = single_total - single_choices.get("NONE", 0)
log(f"  Chose A-E: {single_valid} ({single_valid/single_total*100:.1f}%)")
log(f"  Chose NONE: {single_choices.get('NONE', 0)} ({single_choices.get('NONE',0)/single_total*100:.1f}%)")

# ── 4. 标注员间一致性（仅多标注任务） ──────────────
log(f"\n{'='*70}")
log(f"[Analysis 1: Inter-Annotator Agreement]")
log(f"{'='*70}")

if len(multi_tasks) > 0:
    # 4.1 完全一致
    full_agree = 0
    for tid, t in multi_tasks.items():
        choices = [a["choice"] for a in t["annotations"]]
        if len(set(choices)) == 1:
            full_agree += 1
    log(f"\n  Full agreement: {full_agree}/{len(multi_tasks)} = {full_agree/len(multi_tasks)*100:.1f}%")

    # 4.2 Pairwise
    pairwise_rates = []
    for tid, t in multi_tasks.items():
        choices = [a["choice"] for a in t["annotations"]]
        n = len(choices)
        if n < 2:
            continue
        agrees = sum(1 for i in range(n) for j in range(i+1, n) if choices[i] == choices[j])
        pairwise_rates.append(agrees / (n*(n-1)//2))
    avg_pairwise = sum(pairwise_rates) / len(pairwise_rates) if pairwise_rates else 0
    log(f"  Avg pairwise agreement: {avg_pairwise*100:.1f}%")

    # 4.3 多数一致
    majority_votes = {}
    majority_exists = 0
    for tid, t in multi_tasks.items():
        choices = [a["choice"] for a in t["annotations"]]
        counter = Counter(choices)
        top_choice, top_count = counter.most_common(1)[0]
        if top_count > len(choices) / 2:
            majority_exists += 1
            majority_votes[tid] = top_choice
        else:
            majority_votes[tid] = None
    log(f"  Majority exists (>50%): {majority_exists}/{len(multi_tasks)} = {majority_exists/len(multi_tasks)*100:.1f}%")

    # 4.4 按标注人数分组
    log(f"\n  [Breakdown by Annotation Count]")
    log(f"  {'Count':<8} {'Tasks':<8} {'FullAgree':<12} {'Majority':<12} {'PairwiseAvg':<12}")
    log(f"  {'-'*52}")
    by_count = defaultdict(list)
    for tid, t in multi_tasks.items():
        by_count[len(t["annotations"])].append(t)
    for n_ann in sorted(by_count.keys()):
        tasks = by_count[n_ann]
        full = sum(1 for t in tasks if len(set(a["choice"] for a in t["annotations"])) == 1)
        maj = sum(1 for t in tasks if Counter(a["choice"] for a in t["annotations"]).most_common(1)[0][1] > len(t["annotations"])/2)
        pw_list = []
        for t in tasks:
            choices = [a["choice"] for a in t["annotations"]]
            n = len(choices)
            if n < 2:
                pw_list.append(1.0)
            else:
                ag = sum(1 for i in range(n) for j in range(i+1,n) if choices[i]==choices[j])
                pw_list.append(ag / (n*(n-1)//2))
        avg = sum(pw_list)/len(pw_list)
        log(f"  {n_ann:<8} {len(tasks):<8} {full:<12} {maj:<12} {avg*100:.1f}%")

    # 4.5 仅分析 validation samples 中的多标注任务
    val_multi = {tid: t for tid, t in multi_tasks.items() if tid in validation_tasks}
    if val_multi:
        log(f"\n  [Validation-Sample Multi-Tasks Only] ({len(val_multi)} tasks)")
        val_full = sum(1 for tid, t in val_multi.items() if len(set(a["choice"] for a in t["annotations"])) == 1)
        pw_vals = []
        for tid, t in val_multi.items():
            choices = [a["choice"] for a in t["annotations"]]
            n = len(choices)
            if n >= 2:
                ag = sum(1 for i in range(n) for j in range(i+1,n) if choices[i]==choices[j])
                pw_vals.append(ag / (n*(n-1)//2))
        avg_val_pw = sum(pw_vals)/len(pw_vals) if pw_vals else 0
        log(f"  Full agreement: {val_full}/{len(val_multi)} = {val_full/len(val_multi)*100:.1f}%")
        log(f"  Avg pairwise: {avg_val_pw*100:.1f}%")
else:
    majority_votes = {}
    log("  No multi-annotator tasks found!")

# ── 5. RAG 候选来源命中率 ──────────────────────────
log(f"\n{'='*70}")
log(f"[Analysis 2: RAG Candidate Source Hit Rate]")
log(f"{'='*70}")

source_hits = defaultdict(lambda: {"appears": 0, "chosen": 0})
# 统计每个来源的出现次数和被人选择的次数
for tid, t in task_annotations.items():
    data = t["data"]
    # 记录每个候选的来源
    cand_sources = {}
    for cand in ["a", "b", "c", "d", "e"]:
        source = data.get(f"candidate_{cand}_source", "unknown")
        cand_sources[cand.upper()] = source
        source_hits[source]["appears"] += 1
    # 统计被选择的次数
    for a in t["annotations"]:
        if a["choice"] in cand_sources:
            source = cand_sources[a["choice"]]
            source_hits[source]["chosen"] += 1

log(f"  {'Source':<35} {'Appears':<10} {'Chosen':<10} {'HitRate':<10}")
log(f"  {'-'*65}")
for source in sorted(source_hits.keys()):
    s = source_hits[source]
    rate = s["chosen"] / s["appears"] * 100 if s["appears"] > 0 else 0
    log(f"  {source:<35} {s['appears']:<10} {s['chosen']:<10} {rate:.1f}%")

# RAG Top1 命中率
log(f"\n[Analysis 2b: Human vs RAG Top1 & Top5]")
human_chose_top1 = 0
human_chose_topN = defaultdict(int)
human_chose_valid = 0

for tid, t in task_annotations.items():
    data = t["data"]
    # 找到每个rank对应的candidate
    rank_to_cand = {}
    for cand in ["a", "b", "c", "d", "e"]:
        source = data.get(f"candidate_{cand}_source", "")
        if "top1" in source:
            rank_to_cand[1] = cand.upper()
        elif "top2" in source:
            rank_to_cand[2] = cand.upper()
        elif "top3" in source:
            rank_to_cand[3] = cand.upper()
        elif "top4" in source:
            rank_to_cand[4] = cand.upper()
        elif "top5" in source:
            rank_to_cand[5] = cand.upper()

    for a in t["annotations"]:
        if a["choice"] in ("A", "B", "C", "D", "E"):
            human_chose_valid += 1
            for rank, cand in rank_to_cand.items():
                if a["choice"] == cand:
                    human_chose_topN[rank] += 1
                    if rank == 1:
                        human_chose_top1 += 1
                    break

log(f"  Human chose RAG Top1: {human_chose_top1}/{human_chose_valid} = {human_chose_top1/human_chose_valid*100:.1f}%")
cumulative = 0
for rank in sorted(human_chose_topN.keys()):
    cumulative += human_chose_topN[rank]
    log(f"  Human chose RAG Top1-{rank}: {cumulative}/{human_chose_valid} = {cumulative/human_chose_valid*100:.1f}%")

# ── 6. Deepseek-v4-pro 对比 ────────────────────────
log(f"\n{'='*70}")
log(f"[Analysis 3: Deepseek-v4-pro vs Human]")
log(f"{'='*70}")

deepseek_data = {}
if os.path.exists(DEEPSEEK_FILE):
    with open(DEEPSEEK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                deepseek_data[rec["task_id"]] = rec
    log(f"  Deepseek records loaded: {len(deepseek_data)}")
else:
    log(f"  WARNING: Deepseek file not found: {DEEPSEEK_FILE}")

if deepseek_data:
    # 6.1 Deepseek vs 单标注人类
    ds_vs_single_total = 0
    ds_vs_single_agree = 0
    for tid, t in single_tasks.items():
        if tid not in deepseek_data:
            continue
        human_choice = t["annotations"][0]["choice"]
        if human_choice == "NONE":
            continue
        ds_choice = deepseek_data[tid]["deepseek_choice"]
        ds_vs_single_total += 1
        if ds_choice == human_choice:
            ds_vs_single_agree += 1

    if ds_vs_single_total > 0:
        log(f"\n  [Deepseek vs Single-Annotator Human]")
        log(f"  Compared: {ds_vs_single_total}")
        log(f"  Agreement: {ds_vs_single_agree}/{ds_vs_single_total} = {ds_vs_single_agree/ds_vs_single_total*100:.1f}%")

    # 6.2 Deepseek vs 多数意见
    ds_vs_maj_total = 0
    ds_vs_maj_agree = 0
    for tid, maj in majority_votes.items():
        if tid not in deepseek_data:
            continue
        if maj is None or maj == "NONE":
            continue
        ds_choice = deepseek_data[tid]["deepseek_choice"]
        ds_vs_maj_total += 1
        if ds_choice == maj:
            ds_vs_maj_agree += 1

    if ds_vs_maj_total > 0:
        log(f"\n  [Deepseek vs Majority-Vote]")
        log(f"  Compared: {ds_vs_maj_total}")
        log(f"  Agreement: {ds_vs_maj_agree}/{ds_vs_maj_total} = {ds_vs_maj_agree/ds_vs_maj_total*100:.1f}%")

    # 6.3 Deepseek 综合准确率（有人类参考的所有任务）
    ds_all_total = 0
    ds_all_agree = 0
    for tid, t in task_annotations.items():
        if tid not in deepseek_data:
            continue
        # 优先使用多数意见，其次单标注
        if tid in majority_votes and majority_votes[tid] is not None:
            ref = majority_votes[tid]
        elif len(t["annotations"]) == 1:
            ref = t["annotations"][0]["choice"]
        else:
            continue
        if ref == "NONE":
            continue
        ds_choice = deepseek_data[tid]["deepseek_choice"]
        ds_all_total += 1
        if ds_choice == ref:
            ds_all_agree += 1

    if ds_all_total > 0:
        log(f"\n  [Deepseek vs Human (All Tasks)]")
        log(f"  Compared: {ds_all_total}")
        log(f"  Agreement: {ds_all_agree}/{ds_all_total} = {ds_all_agree/ds_all_total*100:.1f}%")

    # 6.4 按置信度分组
    log(f"\n  [Deepseek Accuracy by Confidence]")
    log(f"  {'Confidence':<14} {'Tasks':<8} {'Correct':<8} {'Accuracy':<10}")
    log(f"  {'-'*40}")
    ds_by_conf = defaultdict(lambda: {"total": 0, "correct": 0})
    for tid, t in task_annotations.items():
        if tid not in deepseek_data:
            continue
        if tid in majority_votes and majority_votes[tid] is not None:
            ref = majority_votes[tid]
        elif len(t["annotations"]) == 1:
            ref = t["annotations"][0]["choice"]
        else:
            continue
        if ref == "NONE":
            continue
        ds_rec = deepseek_data[tid]
        conf = ds_rec["deepseek_confidence"]
        bucket = f"{conf:.1f}"
        ds_by_conf[bucket]["total"] += 1
        if ds_rec["deepseek_choice"] == ref:
            ds_by_conf[bucket]["correct"] += 1

    for bucket in sorted(ds_by_conf.keys()):
        s = ds_by_conf[bucket]
        acc = s["correct"]/s["total"]*100 if s["total"] > 0 else 0
        log(f"  {bucket:<14} {s['total']:<8} {s['correct']:<8} {acc:.1f}%")

    # 6.5 Deepseek NONE-only 分析
    ds_choice_dist = Counter()
    for rec in deepseek_data.values():
        ds_choice_dist[rec["deepseek_choice"]] += 1
    log(f"\n  [Deepseek Choice Distribution]")
    for c in sorted(ds_choice_dist.keys(), key=lambda x: ds_choice_dist[x], reverse=True):
        log(f"  {c}: {ds_choice_dist[c]} ({ds_choice_dist[c]/len(deepseek_data)*100:.1f}%)")

# ── 7. 每个标注员质量 ──────────────────────────────
log(f"\n{'='*70}")
log(f"[Analysis 4: Per-Annotator Quality]")
log(f"{'='*70}")

ann_stats = defaultdict(lambda: {"total": 0, "multi": 0, "agree_maj": 0})
for tid, t in task_annotations.items():
    maj = majority_votes.get(tid)
    for a in t["annotations"]:
        aid = a["annotator"]
        ann_stats[aid]["total"] += 1
        if len(t["annotations"]) >= 2:
            ann_stats[aid]["multi"] += 1
            if maj is not None and a["choice"] == maj:
                ann_stats[aid]["agree_maj"] += 1

log(f"  {'Ann':<8} {'Total':<8} {'Multi':<8} {'AgreeMaj':<10} {'Rate':<8}")
log(f"  {'-'*42}")
for aid in sorted(ann_stats.keys()):
    s = ann_stats[aid]
    if s["multi"] > 0:
        rate = s["agree_maj"]/s["multi"]*100
        log(f"  {aid:<8} {s['total']:<8} {s['multi']:<8} {s['agree_maj']:<10} {rate:.1f}%")
    else:
        log(f"  {aid:<8} {s['total']:<8} {s['multi']:<8} {s['agree_maj']:<10} N/A")

# ── 8. "都不属于" 专项分析 ─────────────────────────
log(f"\n{'='*70}")
log(f"[Analysis 5: 'NONE' (以上选项都不属于) Analysis]")
log(f"{'='*70}")

# 哪些任务的多数意见是NONE
none_majority_tasks = []
for tid, maj in majority_votes.items():
    if maj == "NONE":
        none_majority_tasks.append(tid)
log(f"  Tasks with majority=NONE: {len(none_majority_tasks)}")

# 单标注中选择NONE的任务比例
none_single = sum(1 for tid, t in single_tasks.items() if t["annotations"][0]["choice"] == "NONE")
log(f"  Single-annotator tasks chose NONE: {none_single}/{len(single_tasks)} ({none_single/len(single_tasks)*100:.1f}%)")

# ── 9. 验证样本专项分析 ───────────────────────────
log(f"\n{'='*70}")
log(f"[Analysis 6: Validation Samples (is_validation_sample=1) Deep Dive]")
log(f"{'='*70}")

val_tasks_data = {tid: task_annotations[tid] for tid in validation_tasks if tid in task_annotations}
log(f"  Validation sample tasks: {len(val_tasks_data)}")

if val_tasks_data:
    # 选择分布
    val_choices = Counter()
    for tid, t in val_tasks_data.items():
        for a in t["annotations"]:
            val_choices[a["choice"]] += 1
    log(f"\n  [Validation Set Choice Distribution]")
    vt = sum(val_choices.values())
    for choice in sorted(val_choices.keys(), key=lambda x: val_choices[x], reverse=True):
        log(f"  {choice}: {val_choices[choice]} ({val_choices[choice]/vt*100:.1f}%)")

    # 一致率
    val_full = sum(1 for tid, t in val_tasks_data.items() if len(t["annotations"]) >= 2 and len(set(a["choice"] for a in t["annotations"])) == 1)
    val_multi_count = sum(1 for tid, t in val_tasks_data.items() if len(t["annotations"]) >= 2)
    if val_multi_count > 0:
        log(f"\n  Full agreement (validation, multi-ann): {val_full}/{val_multi_count} = {val_full/val_multi_count*100:.1f}%")

    # Deepseek on validation
    if deepseek_data:
        ds_val_total = 0
        ds_val_agree = 0
        for tid in validation_tasks:
            if tid not in deepseek_data or tid not in task_annotations:
                continue
            t = task_annotations[tid]
            if tid in majority_votes and majority_votes[tid] is not None:
                ref = majority_votes[tid]
            elif len(t["annotations"]) == 1:
                ref = t["annotations"][0]["choice"]
            else:
                continue
            if ref == "NONE":
                continue
            ds_val_total += 1
            if deepseek_data[tid]["deepseek_choice"] == ref:
                ds_val_agree += 1
        if ds_val_total > 0:
            log(f"  Deepseek accuracy on validation: {ds_val_agree}/{ds_val_total} = {ds_val_agree/ds_val_total*100:.1f}%")

# ── 10. 综合结论 ──────────────────────────────────
log(f"\n{'='*70}")
log(f"[SUMMARY: All Metrics]")
log(f"{'='*70}")

# 收集所有可能的准确率指标
metrics = []

# 指标1: 人类选择 A-E 的比例（非NONE）
metrics.append(("Human chose A-E (not NONE) [all tasks]", valid_count/total*100))

# 指标2: 单标注任务中选择 A-E 的比例
metrics.append(("Human chose A-E (single-ann tasks)", single_valid/single_total*100))

# 指标3: 人类选择 = RAG Top1
if human_chose_valid > 0:
    metrics.append(("Human agree with RAG Top1", human_chose_top1/human_chose_valid*100))

# 指标4: 人类选择 ∈ RAG Top1-5
metrics.append(("Human agree with RAG Top1-5", cumulative/human_chose_valid*100 if human_chose_valid > 0 else 0))

# 指标5: 多标注完全一致率
metrics.append(("Multi-ann full agreement", full_agree/len(multi_tasks)*100))

# 指标6: 多标注 pairwise 一致率
metrics.append(("Multi-ann pairwise agreement", avg_pairwise*100))

# 指标7: 多标注多数存在率
metrics.append(("Multi-ann majority exists", majority_exists/len(multi_tasks)*100))

# 指标8-10: 排除NONE后的一致率
if len(multi_tasks) > 0:
    # 排除NONE后的完全一致
    non_none_multi = 0
    non_none_full = 0
    for tid, t in multi_tasks.items():
        non_none_choices = [a["choice"] for a in t["annotations"] if a["choice"] != "NONE"]
        if len(non_none_choices) >= 2:
            non_none_multi += 1
            if len(set(non_none_choices)) == 1:
                non_none_full += 1
    if non_none_multi > 0:
        metrics.append(("Multi-ann full agreement (excl NONE)", non_none_full/non_none_multi*100))

# 指标: Deepseek准确性
if ds_all_total > 0:
    metrics.append(("Deepseek agree with human (all tasks)", ds_all_agree/ds_all_total*100))
if ds_vs_single_total > 0:
    metrics.append(("Deepseek agree with human (single-ann)", ds_vs_single_agree/ds_vs_single_total*100))
if ds_vs_maj_total > 0:
    metrics.append(("Deepseek agree with human (majority)", ds_vs_maj_agree/ds_vs_maj_total*100))

# 指标: 验证集
if val_tasks_data:
    val_multi_count = sum(1 for tid, t in val_tasks_data.items() if len(t["annotations"]) >= 2)
    if val_multi_count > 0:
        pw_vals = []
        for tid, t in val_tasks_data.items():
            if len(t["annotations"]) >= 2:
                choices = [a["choice"] for a in t["annotations"]]
                n = len(choices)
                ag = sum(1 for i in range(n) for j in range(i+1,n) if choices[i]==choices[j])
                pw_vals.append(ag / (n*(n-1)//2))
        avg_val_pw = sum(pw_vals)/len(pw_vals) if pw_vals else 0
        metrics.append(("Validation set pairwise agreement", avg_val_pw*100))

log(f"\n  {'Metric':<55} {'Value':<10}")
log(f"  {'-'*65}")
for name, val in metrics:
    marker = " <-- CLOSE TO 82.8%" if abs(val - 82.8) < 3.0 else ""
    log(f"  {name:<55} {val:.1f}%{marker}")

# 寻找最接近 82.8% 的指标
log(f"\n  Looking for metric closest to 82.8%...")
closest = min(metrics, key=lambda x: abs(x[1] - 82.8))
log(f"  Closest: {closest[0]} = {closest[1]:.1f}% (diff={abs(closest[1]-82.8):.1f}pp)")

log(f"\n{'='*70}")
log("Analysis complete.")
log(f"{'='*70}")

# ── 保存报告 ──────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\nReport saved to: {OUTPUT_FILE}")
