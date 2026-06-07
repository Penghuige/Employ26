# -*- coding: utf-8 -*-
"""
BIO 格式转换与验证模块

功能：
  1. 将 batch_annotator 输出的 .jsonl BIO 样本转换为多种训练格式
  2. 验证标注质量（O 率、实体覆盖率等）
  3. 划分训练集 / 验证集 / 测试集

支持的输出格式：
  - CoNLL 格式（每行 token\tlabel，句子间空行）
  - HuggingFace datasets 兼容的 JSON 格式
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# 读取 JSONL
# ============================================================

def load_bio_jsonl(path: str) -> list[dict]:
    """读取 BIO JSONL 文件，返回样本列表"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败：{e}")
    logger.info(f"加载 {len(samples)} 条 BIO 样本")
    return samples


# ============================================================
# 质量验证
# ============================================================

def validate_bio_samples(samples: list[dict]) -> dict:
    """
    验证 BIO 标注质量，输出统计报告。

    Returns:
        {
            "total": 总样本数,
            "total_tokens": 总 token 数,
            "o_rate": O 标签占比,
            "entity_counts": {实体类型: 出现次数},
            "empty_samples": 完全没有实体的样本数,
            "avg_entities_per_sample": 平均每条样本的实体数
        }
    """
    total_tokens = 0
    o_count = 0
    entity_counts: Counter = Counter()
    empty_count = 0

    for sample in samples:
        tokens = sample.get("tokens", [])
        labels = sample.get("labels", [])
        total_tokens += len(tokens)
        has_entity = False
        for label in labels:
            if label == "O":
                o_count += 1
            elif label.startswith("B-"):
                entity_type = label[2:]
                entity_counts[entity_type] += 1
                has_entity = True
        if not has_entity:
            empty_count += 1

    total_entities = sum(entity_counts.values())
    report = {
        "total": len(samples),
        "total_tokens": total_tokens,
        "o_rate": round(o_count / total_tokens, 4) if total_tokens else 0,
        "entity_counts": dict(entity_counts.most_common()),
        "empty_samples": empty_count,
        "avg_entities_per_sample": round(total_entities / len(samples), 2) if samples else 0,
    }

    logger.info("=== BIO 标注质量报告 ===")
    logger.info(f"  总样本数       : {report['total']}")
    logger.info(f"  总 Token 数    : {report['total_tokens']}")
    logger.info(f"  O 标签占比     : {report['o_rate']:.1%}")
    logger.info(f"  无实体样本数   : {report['empty_samples']}")
    logger.info(f"  平均实体数/条  : {report['avg_entities_per_sample']}")
    logger.info("  各类实体数量:")
    for etype, cnt in report["entity_counts"].items():
        logger.info(f"    {etype:10s}: {cnt}")
    return report


# ============================================================
# 数据集划分
# ============================================================

def split_dataset(
    samples: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    将样本划分为训练集、验证集、测试集。

    Returns:
        (train, dev, test)
    """
    random.seed(seed)
    data = samples.copy()
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    train = data[:train_end]
    dev = data[train_end:dev_end]
    test = data[dev_end:]
    logger.info(f"数据集划分 → 训练:{len(train)} | 验证:{len(dev)} | 测试:{len(test)}")
    return train, dev, test


# ============================================================
# 导出 CoNLL 格式
# ============================================================

def export_conll(samples: list[dict], output_path: str):
    """
    导出为 CoNLL 格式：每行 token\tlabel，句子间空行。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            tokens = sample.get("tokens", [])
            labels = sample.get("labels", [])
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")  # 句子间空行
    logger.info(f"CoNLL 导出完成：{output_path}")


# ============================================================
# 导出 HuggingFace JSON 格式
# ============================================================

def export_hf_json(samples: list[dict], output_path: str):
    """
    导出为 HuggingFace datasets 兼容的 JSON 格式。
    每行一个 {"tokens": [...], "ner_tags": [...]} 对象。
    """
    from src.llm.history.ner_schema import LABEL2ID

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            tokens = sample.get("tokens", [])
            labels = sample.get("labels", [])
            ner_tags = [LABEL2ID.get(label, 0) for label in labels]
            record = {
                "tokens": tokens,
                "ner_tags": ner_tags,
                "labels": labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"HuggingFace JSON 导出完成：{output_path}")


# ============================================================
# 一键转换
# ============================================================

def convert(
    bio_jsonl_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
):
    """
    一键完成：加载 → 验证 → 划分 → 导出 CoNLL + HF JSON。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    samples = load_bio_jsonl(bio_jsonl_path)
    validate_bio_samples(samples)
    train, dev, test = split_dataset(samples, train_ratio, dev_ratio)

    output_dir = Path(output_dir)
    # CoNLL 格式
    export_conll(train, output_dir / "conll" / "train.conll")
    export_conll(dev, output_dir / "conll" / "dev.conll")
    export_conll(test, output_dir / "conll" / "test.conll")
    # HuggingFace JSON 格式
    export_hf_json(train, output_dir / "hf_json" / "train.jsonl")
    export_hf_json(dev, output_dir / "hf_json" / "dev.jsonl")
    export_hf_json(test, output_dir / "hf_json" / "test.jsonl")
    logger.info(f"全部导出完成 → {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BIO 格式转换工具")
    parser.add_argument("--input", required=True, help="ner_bio_samples.jsonl 路径")
    parser.add_argument("--output", default="output/ner_dataset", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()
    convert(args.input, args.output, args.train_ratio, args.dev_ratio)
