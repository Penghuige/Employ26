# -*- coding: utf-8 -*-
"""
批量标注主程序

读取 data/ls_jd_tasks.json，使用本地 Qwen3-8B 对 jd_snippet 字段
进行结构化信息抽取，输出两种格式：
  1. output/llm_annotations/extracted_fields.jsonl  —— 字段抽取结果
  2. output/llm_annotations/ner_bio_samples.jsonl   —— BIO 格式 NER 训练样本

用法：
    python src/llm/batch_annotator.py --input data/ls_jd_tasks.json \\
        --output output/llm_annotations --limit 1000 --mode few_shot
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "llm"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# BIO 转换工具
# ============================================================

def _tokenize_zh(text: str) -> list[str]:
    """简单字符级分词（中文逐字，英文按空格/边界切分）"""
    import re
    tokens = []
    # 将英文单词和数字作为一个 token，中文字符逐字
    for seg in re.findall(r'[A-Za-z0-9_.+#-]+|[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]|[^\s]', text):
        tokens.append(seg)
    return tokens


def _bio_tag(tokens: list[str], span_text: str, bio_prefix: str, bio_seq: list[str]):
    """在 bio_seq 中对 span_text 对应的 token 范围打 BIO 标签（原地修改）"""
    span_tokens = _tokenize_zh(span_text)
    n, m = len(tokens), len(span_tokens)
    for i in range(n - m + 1):
        if tokens[i:i + m] == span_tokens:
            # 只标注尚未被标注的区域（避免重叠）
            if all(bio_seq[i + j] == "O" for j in range(m)):
                bio_seq[i] = f"B-{bio_prefix}"
                for j in range(1, m):
                    bio_seq[i + j] = f"I-{bio_prefix}"
                break


def extracted_to_bio(jd_text: str, extracted: dict) -> dict:
    """
    将抽取结果转换为 BIO 格式的 NER 样本。

    Returns:
        {
            "text": "原始文本",
            "tokens": [...],
            "labels": ["O", "B-SKILL", "I-SKILL", ...]
        }
    """
    from ner_schema import NER_ENTITY_SCHEMA

    tokens = _tokenize_zh(jd_text)
    bio_seq = ["O"] * len(tokens)

    # 字段名 -> schema 中的 bio_prefix 映射
    field_to_prefix = {
        "skills": NER_ENTITY_SCHEMA["SKILL"]["bio_prefix"],
        "tools": NER_ENTITY_SCHEMA["TOOL"]["bio_prefix"],
        "certs": NER_ENTITY_SCHEMA["CERT"]["bio_prefix"],
        "benefits": NER_ENTITY_SCHEMA["BENEFIT"]["bio_prefix"],
        "duties": NER_ENTITY_SCHEMA["DUTY"]["bio_prefix"],
        "headcount": NER_ENTITY_SCHEMA["HEADCOUNT"]["bio_prefix"],
        "job_type": NER_ENTITY_SCHEMA["JOB_TYPE"]["bio_prefix"],
    }

    for field, prefix in field_to_prefix.items():
        val = extracted.get(field)
        if not val:
            continue
        spans = val if isinstance(val, list) else [val]
        for span in spans:
            if span and isinstance(span, str):
                _bio_tag(tokens, span, prefix, bio_seq)

    return {"text": jd_text, "tokens": tokens, "labels": bio_seq}


# ============================================================
# 主流程
# ============================================================

def run_annotation(
    input_path: str,
    output_dir: str,
    limit: int = 0,
    mode: str = "few_shot",
    thinking: bool = False,
    resume: bool = True,
    batch_size: int = 32,
):
    """
    批量标注主函数。使用 extract_batch() 做真正的 batch 推理，
    进度条由 qwen3_extractor 内部的双层 tqdm 负责显示。

    Args:
        input_path : ls_jd_tasks.json 路径
        output_dir : 输出目录
        limit      : 处理条数上限，0 表示全量
        mode       : 'zero_shot' 或 'few_shot'
        thinking   : 是否开启 Qwen3 思维链
        resume     : 是否断点续跑（跳过已处理的条目）
        batch_size : mini-batch 大小，RTX 4090 24G 推荐 32
    """
    from qwen3_extractor import Qwen3Extractor

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields_path = output_dir / "extracted_fields.jsonl"
    bio_path = output_dir / "ner_bio_samples.jsonl"

    # 断点续跑：读取已处理的 row_id
    processed_ids: set = set()
    if resume and fields_path.exists():
        with open(fields_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_ids.add(obj.get("row_id"))
                except Exception:
                    pass
        logger.info(f"断点续跑：已跳过 {len(processed_ids)} 条")

    # 加载数据
    logger.info(f"加载数据：{input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if limit > 0:
        tasks = tasks[:limit]

    # 过滤已处理 + 空文本，保留 (row_id, jd_text) 对
    pending: list[tuple] = []
    for i, task in enumerate(tasks):
        row_id = task.get("data", {}).get("row_id", i)
        if row_id in processed_ids:
            continue
        jd_text = task.get("data", {}).get("jd_snippet", "")
        if jd_text and jd_text.strip():
            pending.append((row_id, jd_text.strip()))

    logger.info(f"待推理条数：{len(pending)}（已跳过 {len(processed_ids)} 条）")
    if not pending:
        logger.info("无待处理数据，退出。")
        return

    # 拆分 row_ids 和 jd_texts
    row_ids = [p[0] for p in pending]
    jd_texts = [p[1] for p in pending]

    # 初始化推理器并执行批量推理（内部含双层进度条）
    extractor = Qwen3Extractor(thinking=thinking, batch_size=batch_size)
    all_results = extractor.extract_batch(jd_texts, mode=mode, show_progress=True)

    # 写出结果
    ok_count, fail_count = 0, 0
    fields_out = open(fields_path, "a", encoding="utf-8")
    bio_out = open(bio_path, "a", encoding="utf-8")

    try:
        for row_id, jd_text, extracted in zip(row_ids, jd_texts, all_results):
            if extracted is None:
                fail_count += 1
                logger.warning(f"row_id={row_id} 抽取失败")
                continue

            # 写入字段抽取结果
            record = {
                "row_id": row_id,
                "jd_snippet": jd_text,
                **extracted,
            }
            fields_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 写入 BIO 格式
            bio_sample = extracted_to_bio(jd_text, extracted)
            bio_sample["row_id"] = row_id
            bio_out.write(json.dumps(bio_sample, ensure_ascii=False) + "\n")

            ok_count += 1

        fields_out.flush()
        bio_out.flush()
    finally:
        fields_out.close()
        bio_out.close()

    logger.info(f"完成！成功 {ok_count} 条，失败 {fail_count} 条")
    logger.info(f"字段抽取结果：{fields_path}")
    logger.info(f"BIO 样本：{bio_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3 批量标注岗位描述")
    parser.add_argument(
        "--input", default="data/ls_jd_tasks.json", help="输入 JSON 文件路径"
    )
    parser.add_argument(
        "--output", default="output/llm_annotations", help="输出目录"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="处理条数上限（0=全量）"
    )
    parser.add_argument(
        "--mode", choices=["zero_shot", "few_shot"], default="few_shot",
        help="Prompt 模式"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="开启 Qwen3 思维链（更准确但更慢）"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="不断点续跑，重新处理所有条目"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="batch 推理大小，RTX 4090 24G 推荐 32（默认 32）"
    )
    args = parser.parse_args()

    run_annotation(
        input_path=args.input,
        output_dir=args.output,
        limit=args.limit,
        mode=args.mode,
        thinking=args.thinking,
        resume=not args.no_resume,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

