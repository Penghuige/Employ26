#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多指标模型评估：超越 Top-1 准确率
==================================
评估维度：
  1. 候选选择准确率 — 从5个候选(A-E)中选对的比例（实际标注场景）
  2. 候选排序质量 — 人类选择在5个候选中排第几
  3. 分歧仲裁准确率 — 320条分歧中，模型更接近人还是DS
  4. 小类/中类/大类级准确率 — 不要求精确到细类
  5. MRR (Mean Reciprocal Rank) — 全量检索的平均倒数排名
  6. NDCG@5 — 归一化折损累积增益
"""

import json, os, time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_FILE = os.path.join(BASE_DIR, "data", "project-4-at-2026-05-27-01-51-7cceb9ba.json")
DEEPSEEK_FILE = os.path.join(BASE_DIR, "output", "deepseek_relabel", "deepseek_relabel_raw.jsonl")
DICT_FILE = os.path.join(BASE_DIR, "data", "中国职业大典.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "rag_round2_training")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "model_comparison.txt")

MODEL_PATHS = {
    "v1 (全量)":      os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned"),
    "v3 (Silver/Gold)": os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned-v3"),
    "v4 (Medium分歧)":  os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned-v4"),
    "baseline (bge-large)": r"D:\model\bge-large-zh-v1.5",
}


def parse_choice(annotation):
    for r in annotation.get("result", []):
        if r["from_name"] == "best_candidate_choice":
            choices = r["value"].get("choices", [])
            if not choices: return None
            raw = choices[0]
            if len(raw) >= 2 and raw[-1] in "ABCDE": return raw[-1]
            if "不" in raw: return "NONE"
    return None


def load_dict():
    df = pd.read_excel(DICT_FILE, engine="openpyxl")
    df.fillna("", inplace=True)
    c2text, c2title, c2subclass, c2midclass, c2major = {}, {}, {}, {}, {}
    for _, row in df.iterrows():
        code = str(row["code"]).strip()
        title = str(row["title"]).strip()
        desc = str(row.get("desc", ""))
        tasks = str(row.get("tasks", ""))
        if not code or not title: continue
        c2title[code] = title
        parts = [title]
        if desc and desc != "nan": parts.append(f"定义：{desc}")
        if tasks and tasks != "nan": parts.append(f"任务：{tasks}")
        c2text[code] = "。".join(parts)
        parts = code.split("-")
        c2subclass[code] = "-".join(parts[:3]) if len(parts) >= 3 else code
        c2midclass[code] = "-".join(parts[:2]) if len(parts) >= 2 else code
        c2major[code] = parts[0] if parts else code
    return c2text, c2title, c2subclass, c2midclass, c2major


def main():
    # ── Load data ──
    print("Loading data...")
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    ds_records = {}
    with open(DEEPSEEK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ds_records[r["task_id"]] = r
    c2text, c2title, c2subclass, c2midclass, c2major = load_dict()

    # ── Build evaluation samples ──
    print("Building eval samples...")
    eval_samples = []  # Each: {job_title, anchor, candidates: [{letter, code, title, text}], human_choice, ds_choice, n_ann}
    for item in raw_data:
        tid = item["id"]
        data = item["data"]
        jt = str(data.get("job_title", "")).strip()
        jr = str(data.get("job_requirements_clean", "")).strip()
        if not jr: continue
        anchor = f"{jt} {jr}"

        # Human choice
        anns = item["annotations"]
        choices = [c for c in [parse_choice(a) for a in anns] if c and c != "NONE"]
        if not choices: continue
        if len(anns) >= 2:
            from collections import Counter
            ctr = Counter(choices)
            hum_choice, cnt = ctr.most_common(1)[0]
        else:
            hum_choice = choices[0]

        # Candidates
        candidates = []
        for letter in "abcde":
            title = str(data.get(f"candidate_{letter}_title", "")).strip()
            code = str(data.get(f"candidate_{letter}_code", "")).strip()
            source = str(data.get(f"candidate_{letter}_source", "")).strip()
            if not code: continue
            text = c2text.get(code, title)
            candidates.append({
                "letter": letter.upper(),
                "code": code,
                "title": c2title.get(code, title),
                "text": text,
                "source": source,
            })

        if len(candidates) < 3: continue

        ds = ds_records.get(tid)
        ds_choice = ds["deepseek_choice"] if ds else None

        eval_samples.append({
            "task_id": tid,
            "anchor": anchor,
            "candidates": candidates,
            "human_choice": hum_choice,
            "ds_choice": ds_choice,
            "n_ann": len(anns),
        })

    print(f"  Eval samples: {len(eval_samples)}")

    # Also build full-occupation evaluation for MRR
    occ_codes = sorted(c2text.keys())
    occ_texts = [c2text[c] for c in occ_codes]

    # ── Evaluate each model ──
    all_results = {}

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_path, device=device)
        model.max_seq_length = 256

        # Encode all occupations once
        with torch.no_grad():
            occ_emb = model.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=True, convert_to_tensor=True)

        results = {
            "candidate_hit": 0,           # model选对候选(同human)
            "candidate_total": 0,
            "human_rank_in_candidates": [], # 人类选择在5个候选人中的排名
            "subclass_hit": 0,            # 至少小类正确(前3位相同)
            "midclass_hit": 0,
            "major_hit": 0,
            "full_total": 0,
            "reciprocal_ranks": [],       # 全量检索的1/rank
            "ds_side_human": 0,           # 分歧中模型偏向人类
            "ds_side_ds": 0,
            "ds_total": 0,
        }

        # ── Metric 1&2: Candidate selection ──
        for sample in eval_samples:
            candidates = sample["candidates"]
            cand_texts = [c["text"] for c in candidates]

            with torch.no_grad():
                anc_emb = model.encode([sample["anchor"]], batch_size=1,
                                       normalize_embeddings=True,
                                       show_progress_bar=False, convert_to_tensor=True)
                cand_emb = model.encode(cand_texts, batch_size=len(cand_texts),
                                        normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_tensor=True)
                sims = torch.mm(anc_emb, cand_emb.T).squeeze(0)
                sorted_idx = torch.argsort(sims, descending=True).cpu().tolist()

            # Which candidate would the model pick?
            model_pick = candidates[sorted_idx[0]]["letter"]

            results["candidate_total"] += 1
            if model_pick == sample["human_choice"]:
                results["candidate_hit"] += 1

            # Human's rank among candidates
            human_letter = sample["human_choice"]
            human_idx_in_cands = next((i for i, c in enumerate(candidates)
                                       if c["letter"] == human_letter), None)
            if human_idx_in_cands is not None:
                human_rank = sorted_idx.index(human_idx_in_cands) + 1
                results["human_rank_in_candidates"].append(human_rank)

            # Disagreement arbitration
            if sample["ds_choice"] and sample["ds_choice"] != sample["human_choice"]:
                results["ds_total"] += 1
                model_rank_human = sorted_idx.index(human_idx_in_cands) + 1 if human_idx_in_cands is not None else 99
                ds_letter = sample["ds_choice"]
                ds_idx = next((i for i, c in enumerate(candidates) if c["letter"] == ds_letter), None)
                model_rank_ds = sorted_idx.index(ds_idx) + 1 if ds_idx is not None else 99
                if model_rank_human < model_rank_ds:
                    results["ds_side_human"] += 1
                elif model_rank_ds < model_rank_human:
                    results["ds_side_ds"] += 1

        # ── Metric 3-5: Hierarchy-level & MRR (抽样加速) ──
        np.random.seed(42)
        mrr_samples = np.random.choice(len(eval_samples), min(3000, len(eval_samples)), replace=False)
        mrr_anchors = [eval_samples[i]["anchor"] for i in mrr_samples]
        mrr_codes = [eval_samples[i]["candidates"][0]["code"] for i in mrr_samples]

        # Get actual human code for each
        mrr_human_codes = []
        for i in mrr_samples:
            s = eval_samples[i]
            hum_letter = s["human_choice"]
            hum_code = next((c["code"] for c in s["candidates"] if c["letter"] == hum_letter), None)
            mrr_human_codes.append(hum_code)

        with torch.no_grad():
            mrr_anc_emb = model.encode(mrr_anchors, batch_size=64, normalize_embeddings=True,
                                       show_progress_bar=True, convert_to_tensor=True)
            mrr_sims = torch.mm(mrr_anc_emb, occ_emb.T)
            _, mrr_ranked = torch.topk(mrr_sims, k=50, dim=1)
        mrr_ranked = mrr_ranked.cpu().tolist()

        for i, (code, ranked) in enumerate(zip(mrr_human_codes, mrr_ranked)):
            if not code or code not in occ_codes: continue
            target_idx = occ_codes.index(code)
            results["full_total"] += 1
            try:
                rank = ranked.index(target_idx) + 1
            except ValueError:
                rank = 51  # not in top 50
            results["reciprocal_ranks"].append(1.0 / rank)

            # Hierarchy check
            pred_code = occ_codes[ranked[0]]
            if c2subclass.get(code) == c2subclass.get(pred_code):
                results["subclass_hit"] += 1
            if c2midclass.get(code) == c2midclass.get(pred_code):
                results["midclass_hit"] += 1
            if c2major.get(code) == c2major.get(pred_code):
                results["major_hit"] += 1

        # ── Compute final metrics ──
        results["candidate_acc"] = results["candidate_hit"] / results["candidate_total"] * 100
        results["mean_human_rank"] = np.mean(results["human_rank_in_candidates"])
        results["mrr"] = np.mean(results["reciprocal_ranks"])
        results["subclass_acc"] = results["subclass_hit"] / results["full_total"] * 100
        results["midclass_acc"] = results["midclass_hit"] / results["full_total"] * 100
        results["major_acc"] = results["major_hit"] / results["full_total"] * 100
        if results["ds_total"] > 0:
            results["ds_side_human_pct"] = results["ds_side_human"] / results["ds_total"] * 100
            results["ds_side_ds_pct"] = results["ds_side_ds"] / results["ds_total"] * 100
        else:
            results["ds_side_human_pct"] = results["ds_side_ds_pct"] = 0

        all_results[model_name] = results

        del model
        torch.cuda.empty_cache()

    # ── Print comparison ──
    print(f"\n\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")

    metrics = [
        ("候选选择准确率", "candidate_acc", "%", "↑ 从5个候选里选对的概率"),
        ("人类选择在候选中的平均排位", "mean_human_rank", "", "↓ 越低越好, 1=完美"),
        ("分歧中偏向人类的比例", "ds_side_human_pct", "%", "↑ 模型更接近人类判断"),
        ("全量检索 MRR", "mrr", "", "↑ 平均倒数排名"),
        ("细类准确率 (Subclass)", "subclass_acc", "%", "↑ 前3位代码匹配"),
        ("中类准确率 (Midclass)", "midclass_acc", "%", "↑ 前2位代码匹配"),
        ("大类准确率 (Major)", "major_acc", "%", "↑ 第1位代码匹配"),
    ]

    for metric_name, key, unit, desc in metrics:
        print(f"\n  [{metric_name}] {desc}")
        print(f"  {'Model':<30} {'Value':>10}")
        print(f"  {'-'*40}")
        values = [(name, r[key]) for name, r in all_results.items()]
        best = max(values, key=lambda x: x[1]) if "↑" in desc else min(values, key=lambda x: x[1])
        for name, val in values:
            marker = " <-- best" if name == best[0] else ""
            if unit == "%":
                print(f"  {name:<30} {val:>9.1f}%{marker}")
            else:
                print(f"  {name:<30} {val:>9.3f}{marker}")

    # ── Candidate rank distribution ──
    print(f"\n\n  [候选排序分布] (人类选择在5个候选中的排位)")
    print(f"  {'Model':<30} {'Rank#1':>8} {'Rank#2':>8} {'Rank#3':>8} {'Rank#4':>8} {'Rank#5':>8}")
    for name in all_results:
        r = all_results[name]
        ranks = r["human_rank_in_candidates"]
        dist = {i: sum(1 for x in ranks if int(x) == i) for i in range(1, 6)}
        total = len(ranks)
        print(f"  {name:<30} " +
              " ".join(f"{dist[i]/total*100:>7.1f}%" for i in range(1, 6)))

    # ── Save ──
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for metric_name, key, unit, desc in metrics:
            f.write(f"\n[{metric_name}] {desc}\n")
            values = [(name, r[key]) for name, r in all_results.items()]
            best = max(values, key=lambda x: x[1]) if "↑" in desc else min(values, key=lambda x: x[1])
            for name, val in values:
                marker = " <-- best" if name == best[0] else ""
                if unit == "%":
                    f.write(f"  {name:<30} {val:>9.1f}%{marker}\n")
                else:
                    f.write(f"  {name:<30} {val:>9.3f}{marker}\n")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
