#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 训练 v4：中等方案（基于分歧的正负样本）
=============================================
正样本: DS一致 + 语义排名 <= 10         (~2,500对)
负样本: DS分歧 + 语义排名 >= 30         (~600对，作为hard negative验证集)

训练: MultipleNegativesRankingLoss (in-batch negatives) + 正样本对比
      负样本不参与训练，单独保留作为评估指标
"""

import json, os, time, random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_FILE = os.path.join(BASE_DIR, "data", "project-4-at-2026-05-27-01-51-7cceb9ba.json")
DEEPSEEK_FILE = os.path.join(BASE_DIR, "output", "deepseek_relabel", "deepseek_relabel_raw.jsonl")
DICT_FILE = os.path.join(BASE_DIR, "data", "中国职业大典.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "rag_round2_training")
BASE_MODEL_PATH = r"D:\model\bge-large-zh-v1.5"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned-v4")


@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    max_seq_length: int = 256
    warmup_ratio: float = 0.1
    random_seed: int = 42
    # 负样本方案阈值
    pos_semantic_rank_max: int = 10   # DS一致+语义排名<=此值 → 正样本
    neg_semantic_rank_min: int = 30   # DS分歧+语义排名>=此值 → 负样本


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
    c2text, c2title = {}, {}
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
    return c2text, c2title


def compute_semantic_rank(anchor, target_code, occ_codes, occ_emb, model):
    """计算 target_code 在全部职业中的语义排名。"""
    if target_code not in occ_codes:
        return None
    with torch.no_grad():
        anc_emb = model.encode([anchor], batch_size=1, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_tensor=True)
        sims = torch.mm(anc_emb, occ_emb.T).squeeze(0)
        sorted_idx = torch.argsort(sims, descending=True).cpu().tolist()
        target_idx = occ_codes.index(target_code)
        return sorted_idx.index(target_idx) + 1


def main():
    config = Config()
    print("=" * 70)
    print("RAG Training v4: Medium Scheme (DS agree/disagree + semantic)")
    print(f"  Pos: DS agree + sem rank <= {config.pos_semantic_rank_max}")
    print(f"  Neg: DS disagree + sem rank >= {config.neg_semantic_rank_min}")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load ──
    print("\n[1] Loading data...")
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    ds_records = {}
    with open(DEEPSEEK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ds_records[r["task_id"]] = r
    c2text, c2title = load_dict()
    occ_codes = sorted(c2text.keys())
    occ_texts = [c2text[c] for c in occ_codes]
    print(f"  Human: {len(raw_data)}, DS: {len(ds_records)}, Dict: {len(c2text)}")

    # ── Pre-encode occupations ──
    print("\n[2] Encoding occupations...")
    model = SentenceTransformer(BASE_MODEL_PATH, device="cuda")
    model.max_seq_length = config.max_seq_length
    with torch.no_grad():
        occ_emb = model.encode(occ_texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_tensor=True)

    # ── Build positive & negative pairs ──
    print("\n[3] Computing semantic ranks & building pairs...")
    positive_pairs, negative_pairs, test_pairs = [], [], []
    n_skipped_none = n_skipped_no_ds = n_mid_rank = 0
    sem_ranks_pos, sem_ranks_neg, sem_ranks_mid = [], [], []

    for i, item in enumerate(raw_data):
        if i % 2000 == 0:
            print(f"  Progress: {i}/{len(raw_data)}")
        tid = item["id"]
        data = item["data"]
        anns = item["annotations"]
        jt = str(data.get("job_title", "")).strip()
        jr = str(data.get("job_requirements_clean", "")).strip()
        if not jr: continue
        anchor = f"{jt} {jr}"

        choices = [c for c in [parse_choice(a) for a in anns] if c and c != "NONE"]
        if not choices:
            n_skipped_none += 1
            continue

        if len(anns) >= 2:
            counter = Counter(choices)
            hum_choice, cnt = counter.most_common(1)[0]
        else:
            hum_choice = choices[0]

        hum_code = str(data.get(f"candidate_{hum_choice.lower()}_code", "")).strip()
        if not hum_code or hum_code not in c2text: continue

        ds = ds_records.get(tid)
        if ds is None:
            n_skipped_no_ds += 1
            # 无DS的留作测试
            test_pairs.append({"anchor": anchor, "code": hum_code,
                               "positive": c2text[hum_code], "task_id": tid})
            continue

        ds_agrees = (ds["deepseek_choice"] == hum_choice)

        # 计算语义排名
        sem_rank = compute_semantic_rank(anchor, hum_code, occ_codes, occ_emb, model)

        pair = {
            "task_id": tid, "anchor": anchor,
            "code": hum_code, "positive": c2text[hum_code],
            "job_title": jt, "ds_agrees": ds_agrees, "sem_rank": sem_rank,
            "hum_choice": hum_choice, "ds_choice": ds["deepseek_choice"],
        }

        if ds_agrees and sem_rank and sem_rank <= config.pos_semantic_rank_max:
            positive_pairs.append(pair)
            sem_ranks_pos.append(sem_rank)
        elif not ds_agrees and sem_rank and sem_rank >= config.neg_semantic_rank_min:
            negative_pairs.append(pair)
            sem_ranks_neg.append(sem_rank)
        else:
            n_mid_rank += 1
            sem_ranks_mid.append(sem_rank)
            test_pairs.append(pair)

    print(f"\n  Positive pairs (DS agree + sem<={config.pos_semantic_rank_max}): {len(positive_pairs)}")
    print(f"    Avg semantic rank: {np.mean(sem_ranks_pos):.1f}")
    print(f"  Negative pairs (DS disagree + sem>={config.neg_semantic_rank_min}): {len(negative_pairs)}")
    print(f"    Avg semantic rank: {np.mean(sem_ranks_neg):.1f}")
    print(f"  Mid-rank / test pool: {n_mid_rank}")
    print(f"  Skipped: NONE={n_skipped_none}, no_DS={n_skipped_no_ds}")

    # ── Split train/test ──
    print(f"\n[4] Split train/test...")
    random.seed(config.random_seed)
    random.shuffle(positive_pairs)
    n_train = int(len(positive_pairs) * 0.85)
    train_pos = positive_pairs[:n_train]
    test_pos = positive_pairs[n_train:]

    # 添加额外测试样本
    random.shuffle(test_pairs)
    test_all = test_pos + test_pairs[:3000]
    # 去重 train ids
    train_ids = {p["task_id"] for p in train_pos}
    test_all = [p for p in test_all if p["task_id"] not in train_ids]

    print(f"  Train (positive only): {len(train_pos)}")
    print(f"  Test (mixed):          {len(test_all)}")
    print(f"  Negative set (held-out): {len(negative_pairs)}")

    # ── Train ──
    print(f"\n[5] Training...")
    train_examples = [InputExample(texts=[p["anchor"], p["positive"]]) for p in train_pos]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    del model  # release base model
    torch.cuda.empty_cache()
    model = SentenceTransformer(BASE_MODEL_PATH, device=device_str)
    model.max_seq_length = config.max_seq_length

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * config.epochs * config.warmup_ratio)

    print(f"  Batches/epoch: {len(train_dataloader)}, Total: {len(train_dataloader)*config.epochs}")

    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True,
        save_best_model=True,
        use_amp=True,
        evaluator=None,
    )
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")

    # ── Evaluate ──
    print(f"\n[6] Evaluation...")
    occ_codes_list = sorted(c2text.keys())
    occ_texts_list = [c2text[c] for c in occ_codes_list]
    code_to_idx_eval = {c: i for i, c in enumerate(occ_codes_list)}

    with torch.no_grad():
        occ_emb_eval = model.encode(occ_texts_list, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=True, convert_to_tensor=True)

    # Evaluate in chunks
    EVAL_CHUNK = 1000
    anchors_test = [p["anchor"] for p in test_all]

    topk = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    for start in range(0, len(anchors_test), EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, len(anchors_test))
        chunk = anchors_test[start:end]
        with torch.no_grad():
            chunk_emb = model.encode(chunk, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_tensor=True)
            chunk_sim = torch.mm(chunk_emb, occ_emb_eval.T)
            _, chunk_rk = torch.topk(chunk_sim, k=10, dim=1)
        chunk_rk = chunk_rk.cpu().tolist()
        for i, rk in enumerate(chunk_rk):
            pair = test_all[start + i]
            gt = code_to_idx_eval.get(pair["code"])
            if gt is None: continue
            total += 1
            for rank, pred in enumerate(rk, 1):
                if pred == gt:
                    for k in topk:
                        if rank <= k: topk[k] += 1
                    break
        print(f"  Eval: {end}/{len(anchors_test)}")

    print(f"\n  Test N={total}")
    for k in [1, 3, 5, 10]:
        print(f"  Top-{k:>2}: {topk[k]:>5}/{total} = {topk[k]/total*100:.1f}%")

    # ── Negative set analysis ──
    print(f"\n[7] Negative set evaluation ({len(negative_pairs)} disagreements)...")
    neg_anchors = [p["anchor"] for p in negative_pairs]
    if neg_anchors:
        neg_human_codes = [code_to_idx_eval.get(p["code"]) for p in negative_pairs]
        with torch.no_grad():
            neg_emb = model.encode(neg_anchors, batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_tensor=True)
            neg_sim = torch.mm(neg_emb, occ_emb_eval.T)
            _, neg_rk = torch.topk(neg_sim, k=10, dim=1)
        neg_rk = neg_rk.cpu().tolist()

        # How many of the negative samples does the model rank highly?
        neg_hit_top10 = 0
        for (gt, rk) in zip(neg_human_codes, neg_rk):
            if gt is not None and gt in rk[:10]:
                neg_hit_top10 += 1
        print(f"  Model Top10 hits human choice in {neg_hit_top10}/{len(negative_pairs)} = "
              f"{neg_hit_top10/len(negative_pairs)*100:.1f}% of disagreement cases")
        print(f"  (Lower is better - these are suspected annotation errors)")

    # ── Compare v1 ──
    print(f"\n[8] Comparison with v1...")
    del model
    torch.cuda.empty_cache()
    v1_path = os.path.join(OUTPUT_DIR, "bge-large-round2-finetuned")
    v1 = SentenceTransformer(v1_path, device=device_str)
    v1.max_seq_length = 256
    v1_topk = {1: 0, 3: 0, 5: 0}
    v1_total = 0
    with torch.no_grad():
        v1_occ = v1.encode(occ_texts_list, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_tensor=True)
    for start in range(0, len(anchors_test), EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, len(anchors_test))
        chunk = anchors_test[start:end]
        with torch.no_grad():
            chunk_emb = v1.encode(chunk, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_tensor=True)
            chunk_sim = torch.mm(chunk_emb, v1_occ.T)
            _, chunk_rk = torch.topk(chunk_sim, k=5, dim=1)
        chunk_rk = chunk_rk.cpu().tolist()
        for i, rk in enumerate(chunk_rk):
            pair = test_all[start + i]
            gt = code_to_idx_eval.get(pair["code"])
            if gt is None: continue
            v1_total += 1
            for rank, pred in enumerate(rk, 1):
                if pred == gt:
                    for k in v1_topk:
                        if rank <= k: v1_topk[k] += 1
                    break

    print(f"  {'':<15} {'v1':<12} {'v4':<12}")
    print(f"  {'-'*39}")
    for k in [1, 3, 5]:
        v1p = v1_topk[k]/v1_total*100
        v4p = topk[k]/total*100
        diff = v4p - v1p
        print(f"  Top-{k:<12} {v1p:.1f}%{'':6} {v4p:.1f}% ({'+'if diff>0 else ''}{diff:.1f}pp)")

    # Save
    results = {
        "train_pos": len(train_pos),
        "test_total": total,
        "negative_count": len(negative_pairs),
        "v4_top1": topk[1]/total*100 if total else 0,
        "v4_top3": topk[3]/total*100 if total else 0,
        "v4_top5": topk[5]/total*100 if total else 0,
        "v4_top10": topk[10]/total*100 if total else 0,
        "v1_top1": v1_topk[1]/v1_total*100 if v1_total else 0,
        "v1_top3": v1_topk[3]/v1_total*100 if v1_total else 0,
        "v1_top5": v1_topk[5]/v1_total*100 if v1_total else 0,
        "neg_top10_hit_rate": neg_hit_top10/len(negative_pairs)*100 if neg_anchors else 0,
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_v4.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Model: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
