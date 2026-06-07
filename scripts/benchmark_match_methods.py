"""职业细类匹配方法基准测试框架。

在统一测试集上评估各候选方法的匹配准确率，保证公平可量化。

测试集设计:
- Tier 1 (Gold): 30 条验证集 (inner_id 1-30)，12-20 位标注员交叉验证
- Tier 2 (Silver): ~265 条多标注员任务 (2+ annotators)，用于扩展覆盖

候选方法:
- random: 随机选择（下限 baseline）
- bm25_top1: 选 BM25 召回排名第一的候选
- bge_top1: 选 BGE 相似度最高的候选
- deepseek: DeepSeek V4 Pro 直接评判

指标:
- Accuracy (整体)
- Per-class F1
- NONE recall (人工NONE → 方法是否也判NONE)
- NONE precision (方法判NONE → 人工是否也是NONE)

用法:
    python scripts/benchmark_match_methods.py
    python scripts/benchmark_match_methods.py --skip-deepseek  # 只跑快速baseline
    python scripts/benchmark_match_methods.py --tier gold      # 只跑 Gold-30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")

INPUT_JSON = str(PROJECT_ROOT / "data" / "project-4-at-2026-05-27-01-51-7cceb9ba.json")
DUCKDB_PATH = str(PROJECT_ROOT / "output" / "recruit.duckdb")
EMBEDDING_PATH = r"D:\model\bge-large-zh-v1.5"


# ===================================================================
# 测试集加载
# ===================================================================
def normalize_choice(c: str) -> str:
    c = c.strip()
    if "都不属于" in c:
        return "NONE"
    return c.replace("候选", "").strip()


def load_test_set(tier: str = "all") -> List[Dict]:
    """加载测试集。

    Args:
        tier: "gold" (30条), "silver" (多标注员), "all" (全部)
    """
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold = []
    silver = []

    for t in data:
        inner = t.get("inner_id", 0)
        ann_count = len(t.get("annotations", []))
        td = t.get("data", {})

        # 提取多数票
        choices = Counter()
        for ann in t.get("annotations", []):
            for r in ann.get("result", []):
                for c in r.get("value", {}).get("choices", []):
                    choices[normalize_choice(c)] += 1
        if not choices:
            continue
        top = choices.most_common(1)[0]
        majority = top[0]
        agreement = top[1] / choices.total()

        # 提取候选
        candidates = []
        for letter in ["a", "b", "c", "d", "e"]:
            candidates.append({
                "letter": letter.upper(),
                "code": td.get(f"candidate_{letter}_code", "") or "",
                "title": td.get(f"candidate_{letter}_title", "") or "",
                "desc": td.get(f"candidate_{letter}_desc", "") or "",
                "source": td.get(f"candidate_{letter}_source", "") or "",
            })

        item = {
            "task_id": t["id"],
            "inner_id": inner,
            "job_title": td.get("job_title", "") or "",
            "job_requirements": td.get("job_requirements_clean", "") or "",
            "candidates": candidates,
            "label": majority,
            "agreement": agreement,
            "num_annotators": ann_count,
            "all_votes": dict(choices),
            "sample_source": td.get("sample_source", ""),
        }

        if inner >= 1 and inner <= 30:
            gold.append(item)
        elif ann_count > 1:
            silver.append(item)

    gold.sort(key=lambda x: x["inner_id"])

    if tier == "gold":
        result = gold
    elif tier == "silver":
        result = silver
    else:
        result = gold + silver

    logger.info("测试集: %d 条 (Gold=%d, Silver=%d)", len(result), len(gold), len(silver))

    # 打标签分布
    labels = Counter(i["label"] for i in result)
    logger.info("标签分布: %s", dict(labels.most_common()))

    return result


# ===================================================================
# 候选方法实现
# ===================================================================
class RandomBaseline:
    """随机选择。"""
    name = "random"

    def predict(self, items: List[Dict]) -> List[str]:
        rng = np.random.RandomState(42)
        return [rng.choice(["A", "B", "C", "D", "E", "NONE"]) for _ in items]


class BM25Baseline:
    """选 BM25 召回排名第一的候选。候选来源 tier*_prefill_top1 = rank 1。"""
    name = "bm25_top1"

    def predict(self, items: List[Dict]) -> List[str]:
        preds = []
        for item in items:
            best = None
            best_rank = 99
            for cand in item["candidates"]:
                src = cand.get("source", "")
                rank = 99
                if "top1" in src:
                    rank = 1
                elif "top2" in src:
                    rank = 2
                elif "top3" in src:
                    rank = 3
                elif "top4" in src:
                    rank = 4
                elif "top5" in src:
                    rank = 5
                if rank < best_rank:
                    best_rank = rank
                    best = cand["letter"]
            preds.append(best or "A")
        return preds


class BGEBaseline:
    """选 BGE 相似度最高的候选。"""
    name = "bge_top1"

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_PATH)

    def predict(self, items: List[Dict]) -> List[str]:
        # 批量编码所有岗位文本
        job_texts = [
            f"{i['job_title']} {i['job_requirements'][:1500]}"
            for i in items
        ]
        # 批量编码所有候选文本
        all_cand_texts = []
        for item in items:
            for cand in item["candidates"]:
                text = f"{cand['title']} {cand['desc'][:500]}"
                all_cand_texts.append(text)

        logger.info("BGE 编码: %d jobs + %d candidates", len(job_texts), len(all_cand_texts))
        job_embs = self.model.encode(job_texts, normalize_embeddings=True,
                                     show_progress_bar=True, batch_size=64)
        cand_embs = self.model.encode(all_cand_texts, normalize_embeddings=True,
                                      show_progress_bar=True, batch_size=256)

        preds = []
        for i, item in enumerate(items):
            job_emb = job_embs[i:i + 1]
            best_letter = None
            best_sim = -1
            for j in range(5):
                sim = float(np.dot(cand_embs[i * 5 + j], job_emb.T).item())
                if sim > best_sim:
                    best_sim = sim
                    best_letter = item["candidates"][j]["letter"]
            preds.append(best_letter or "A")
        return preds


class DeepSeekBaseline:
    """DeepSeek V4 Pro 直接评判。"""
    name = "deepseek"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.system_prompt = (
            "你是《中华人民共和国职业分类大典》（2022年版）的资深分类专家。\n"
            "你的任务是根据招聘岗位的标题和描述，从 5 个候选职业中选择最匹配的一个。\n"
            "评判原则：\n"
            "1. 以岗位描述中的实际工作内容为主要判断依据，不要只看岗位名称。\n"
            "2. 如果你认为5个候选都不合适，请选择 NONE。\n"
            "3. 输出必须是严格的 JSON，不要附带任何解释性文字。"
        )

    def predict(self, items: List[Dict]) -> List[str]:
        preds = []
        for item in tqdm(items, desc="DeepSeek judging"):
            c = item["candidates"]
            user = (
                f"【招聘岗位】\n岗位名称：{item['job_title']}\n"
                f"岗位要求：\n{item['job_requirements'][:3000]}\n\n"
                f"【候选职业】\n"
                f"候选A: [{c[0]['code']}] {c[0]['title']}\n"
                f"候选B: [{c[1]['code']}] {c[1]['title']}\n"
                f"候选C: [{c[2]['code']}] {c[2]['title']}\n"
                f"候选D: [{c[3]['code']}] {c[3]['title']}\n"
                f"候选E: [{c[4]['code']}] {c[4]['title']}\n\n"
                '请输出 JSON：{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0,"reasoning":"30字内"}'
            )
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=5120,
                    timeout=60,
                )
                message = resp.choices[0].message
                raw = (getattr(message, "content", None) or "").strip()
                if not raw:
                    raw = (getattr(message, "reasoning_content", None) or "").strip()
                parsed = self._parse(raw)
                preds.append(parsed.get("best_candidate", "PARSE_ERROR"))
            except Exception as exc:
                logger.warning("DS error on task %d: %s", item["task_id"], exc)
                preds.append("API_ERROR")
            time.sleep(0.3)
        return preds

    @staticmethod
    def _parse(raw: str) -> Dict:
        text = raw.strip()
        for m in ("```json", "```"):
            text = text.replace(m, "")
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        m = re.search(r'"best_candidate"\s*:\s*"([^"]+)"', text)
        if m:
            return {"best_candidate": m.group(1)}
        return {}


# ===================================================================
# 评测指标
# ===================================================================
def evaluate(predictions: List[str], items: List[Dict], method_name: str) -> Dict:
    """计算所有指标。"""
    y_true = [i["label"] for i in items]
    y_pred = predictions

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)

    # Per-class metrics
    classes = ["A", "B", "C", "D", "E", "NONE"]
    per_class = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1,
                          "support": sum(1 for t in y_true if t == cls)}

    # Macro F1
    macro_f1 = np.mean([per_class[c]["f1"] for c in classes])

    # NONE-specific
    none_recall = per_class["NONE"]["recall"]
    none_precision = per_class["NONE"]["precision"]

    # Weighted by annotator agreement (agreement高的答对了加分)
    weighted_correct = sum(
        items[i]["agreement"] for i in range(len(items))
        if y_true[i] == y_pred[i]
    )
    weighted_acc = weighted_correct / sum(i["agreement"] for i in items)

    return {
        "method": method_name,
        "n": len(y_true),
        "accuracy": acc,
        "weighted_accuracy": weighted_acc,
        "macro_f1": macro_f1,
        "none_recall": none_recall,
        "none_precision": none_precision,
        "per_class": per_class,
    }


def bootstrap_confidence(predictions: List[str], items: List[Dict],
                         n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Bootstrap 95% 置信区间。"""
    n = len(items)
    rng = np.random.RandomState(42)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        samp_pred = [predictions[i] for i in idx]
        samp_true = [items[i]["label"] for i in idx]
        accs.append(sum(1 for t, p in zip(samp_true, samp_pred) if t == p) / n)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


# ===================================================================
# 主流程
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="职业匹配基准测试")
    parser.add_argument("--tier", choices=["gold", "silver", "all"], default="all")
    parser.add_argument("--skip-deepseek", action="store_true")
    parser.add_argument("--skip-bge", action="store_true")
    args = parser.parse_args()

    # 1. 加载测试集
    items = load_test_set(args.tier)
    if not items:
        logger.error("测试集为空")
        return

    # 2. 运行各候选方法
    methods: List[Any] = [
        RandomBaseline(),
        BM25Baseline(),
    ]
    if not args.skip_bge:
        methods.append(BGEBaseline())
    if not args.skip_deepseek:
        methods.append(DeepSeekBaseline())

    all_results = []
    for method in methods:
        logger.info("=" * 50)
        logger.info("运行: %s", method.name)
        t0 = time.time()
        preds = method.predict(items)
        elapsed = time.time() - t0
        logger.info("耗时: %.1fs", elapsed)

        result = evaluate(preds, items, method.name)
        ci_low, ci_high = bootstrap_confidence(preds, items)
        result["ci_95"] = (ci_low, ci_high)
        result["time_seconds"] = elapsed
        all_results.append(result)

    # 3. 输出排行榜
    print(f"\n{'='*70}")
    print(f"  基准测试排行榜 (测试集: {len(items)} 条, Tier={args.tier})")
    print(f"{'='*70}")
    print(f"  {'方法':<15s} {'Accuracy':>8s} {'95% CI':>16s} {'MacroF1':>8s} "
          f"{'NONE-R':>8s} {'NONE-P':>8s} {'时间':>8s}")
    print(f"  {'-'*70}")

    sorted_results = sorted(all_results, key=lambda x: -x["accuracy"])
    for r in sorted_results:
        ci = f"{r['ci_95'][0]:.2f}-{r['ci_95'][1]:.2f}"
        print(f"  {r['method']:<15s} {r['accuracy']:>7.1%} {ci:>16s} "
              f"{r['macro_f1']:>7.3f} {r['none_recall']:>7.1%} "
              f"{r['none_precision']:>7.1%} {r['time_seconds']:>6.1f}s")

    # 4. Per-class 对比
    print(f"\n{'='*70}")
    print(f"  各类别 F1 对比")
    print(f"{'='*70}")
    classes = ["A", "B", "C", "D", "E", "NONE"]
    header = f"  {'方法':<15s} " + " ".join(f"{c:>7s}" for c in classes)
    print(header)
    print(f"  {'-'*58}")
    for r in sorted_results:
        row = f"  {r['method']:<15s} "
        for c in classes:
            row += f"{r['per_class'][c]['f1']:>6.3f} "
        print(row)

    # 5. 置信度分析
    if not args.skip_deepseek:
        print(f"\n{'='*70}")
        print(f"  显著差异检验 (vs DeepSeek)")
        print(f"{'='*70}")
        ds_idx = next(i for i, r in enumerate(all_results) if r["method"] == "deepseek")
        ds_preds = None
        for method in methods:
            if method.name == "deepseek":
                ds_preds = method.predict(items) if 'preds_cache' not in dir() else preds
                break

        for r in sorted_results:
            if r["method"] == "deepseek":
                continue
            diff = r["accuracy"] - all_results[ds_idx]["accuracy"]
            # McNemar-like: 统计两者预测不一致的任务中谁更对
            # (simplified)
            print(f"  {r['method']:<15s} vs deepseek: {diff:+.1%}")

    # 6. 保存结果
    output = {
        "test_set_size": len(items),
        "test_set_tier": args.tier,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {k: v for k, v in r.items() if k != "per_class"}
            for r in sorted_results
        ],
        "per_class": {r["method"]: r["per_class"] for r in sorted_results},
    }
    out_path = PROJECT_ROOT / "output" / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s", out_path)


if __name__ == "__main__":
    main()
