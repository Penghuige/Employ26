# -*- coding: utf-8 -*-
"""
快速测试脚本（不加载模型）

验证 Prompt 构建、BIO 转换逻辑是否正确，无需 GPU。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ner_schema import NER_ENTITY_SCHEMA, BIO_LABELS, LABEL2ID
from prompt_builder import build_prompt
from batch_annotator import extracted_to_bio


def test_schema():
    print("\n" + "=" * 60)
    print("[1] NER Schema 验证")
    print("=" * 60)
    print(f"  实体类型数：{len(NER_ENTITY_SCHEMA)}")
    for k, v in NER_ENTITY_SCHEMA.items():
        print(f"  [{k:10s}] {v['label']} | 示例: {v['examples'][:2]}")
    print(f"\n  BIO 标签集（共 {len(BIO_LABELS)} 个）:")
    print(f"  {BIO_LABELS}")


def test_prompt():
    print("\n" + "=" * 60)
    print("[2] Prompt 构建验证（few_shot）")
    print("=" * 60)
    sample_jd = (
        "负责数据分析与可视化，熟练使用Python、Tableau，"
        "具备PMP认证者优先，提供五险一金+餐补，招聘2人，全职。"
    )
    messages = build_prompt(sample_jd, mode="few_shot", thinking=False)
    print(f"  消息轮数：{len(messages)}")
    for m in messages:
        role = m["role"]
        content_preview = m["content"][:80].replace("\n", " ")
        print(f"  [{role:10s}] {content_preview}...")


def test_bio_conversion():
    print("\n" + "=" * 60)
    print("[3] BIO 转换验证")
    print("=" * 60)
    jd_text = "熟练使用Python和SQL，具备CPA证书，提供五险一金，招聘1人，全职。"
    extracted = {
        "skills": ["数据分析"],
        "tools": ["Python", "SQL"],
        "certs": ["CPA"],
        "benefits": ["五险一金"],
        "duties": [],
        "headcount": "1人",
        "job_type": "全职",
    }
    bio_sample = extracted_to_bio(jd_text, extracted)
    print(f"  原文：{jd_text}")
    print(f"  Tokens（共 {len(bio_sample['tokens'])} 个）:")
    pairs = list(zip(bio_sample["tokens"], bio_sample["labels"]))
    for token, label in pairs:
        if label != "O":
            print(f"    [{label:10s}] '{token}'")
    entity_tokens = [(t, l) for t, l in pairs if l != "O"]
    print(f"  标注了 {len(entity_tokens)} 个非O token")


def test_zero_shot_prompt():
    print("\n" + "=" * 60)
    print("[4] Prompt 构建验证（zero_shot）")
    print("=" * 60)
    sample_jd = "负责销售管理，要求本科，2年经验，善沟通。"
    messages = build_prompt(sample_jd, mode="zero_shot", thinking=True)
    print(f"  消息轮数：{len(messages)}（zero_shot 应为 2 条）")
    print(f"  用户消息前缀：{messages[-1]['content'][:20]}")


if __name__ == "__main__":
    test_schema()
    test_prompt()
    test_bio_conversion()
    test_zero_shot_prompt()
    print("\n" + "=" * 60)
    print("所有基础测试通过！（未加载模型）")
    print("=" * 60)
    print("\n下一步：运行真实推理")
    print("  cd d:/pythonProject/leisure/Employ26")
    print("  python src/llm/batch_annotator.py --limit 10 --mode few_shot")
    print("  python src/llm/bio_converter.py --input output/llm_annotations/ner_bio_samples.jsonl")

