#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT-NER 推理脚本

训练完成后，使用此脚本对招聘JD文本提取职位(TITLE)和技能(SKILL)。

使用方式：
  # 单条文本
  python src/bert/ner_predict.py --text "负责Java后端开发，熟悉Spring Boot和MySQL"

  # 批量处理文件（每行一条文本）
  python src/bert/ner_predict.py --input data/jd_texts.txt --output output/ner_results.csv

  # 批量处理原始JSON数据集
  python src/bert/ner_predict.py --json data/ls_jd_tasks.json --output output/ner_results.csv

  # 交互模式
  python src/bert/ner_predict.py --interactive
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
import json
import logging
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent.parent
NER_MODEL_DIR = BASE_DIR / "output" / "models" / "bert_ner"

# 与训练时保持一致
ID2LABEL = {0: "O", 1: "B-TITLE", 2: "I-TITLE", 3: "B-SKILL", 4: "I-SKILL"}
MAX_LEN  = 256


class NERPredictor:
    """
    BERT-NER 推理器

    加载训练好的模型，对输入文本执行命名实体识别，
    提取职位名称(TITLE)和技能词(SKILL)。
    """

    def __init__(self, model_dir=None):
        """
        初始化推理器

        Args:
            model_dir: 模型目录，默认为 output/models/bert_ner
        """
        mdir = Path(model_dir) if model_dir else NER_MODEL_DIR
        if not mdir.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {mdir}\n"
                f"请先运行训练: python src/bert/bert_ner.py --train"
            )
        logger.info("加载NER模型: %s", mdir)
        self.tokenizer = BertTokenizerFast.from_pretrained(str(mdir))
        self.model     = BertForTokenClassification.from_pretrained(str(mdir))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("模型加载完成，设备: %s，标签数: %d",
                    self.device, self.model.config.num_labels)

    def _clean(self, text):
        """清洗文本：去除HTML标签和多余空白"""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:500]

    def _decode_bio(self, chars, char_labels):
        """将BIO标签序列解码为实体列表"""
        entities = []
        i = 0
        while i < len(chars):
            lbl = char_labels[i]
            if lbl.startswith("B-"):
                etype = lbl[2:]
                j = i + 1
                while j < len(chars) and char_labels[j] == "I-" + etype:
                    j += 1
                ent_text = "".join(chars[i:j]).strip()
                if ent_text:
                    entities.append({
                        "text":  ent_text,
                        "type":  etype,
                        "start": i,
                        "end":   j,
                    })
                i = j
            else:
                i += 1
        return entities

    def predict(self, text):
        """
        对单条文本做NER推理

        Args:
            text: 输入文本（招聘JD或职位描述）

        Returns:
            dict:
                titles   - 职位名称列表
                skills   - 技能词列表
                entities - 完整实体列表（含位置信息）
        """
        text  = self._clean(text)
        chars = list(text)
        if not chars:
            return {"titles": [], "skills": [], "entities": []}

        # tokenize（is_split_into_words 按字符处理）
        enc = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        enc_in = {k: v.to(self.device) for k, v in enc.items()}

        # 模型推理
        with torch.no_grad():
            out = self.model(**enc_in)
        pred_ids = out.logits.argmax(-1).squeeze(0).cpu().tolist()

        # word_ids 将token映射回字符索引
        word_ids = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=MAX_LEN,
            truncation=True,
        ).word_ids()

        # 每个字符取第一个子词的标签
        char_labels = [None] * len(chars)
        for ti, wid in enumerate(word_ids):
            if wid is not None and char_labels[wid] is None and ti < len(pred_ids):
                char_labels[wid] = ID2LABEL.get(pred_ids[ti], "O")
        char_labels = [l or "O" for l in char_labels]

        entities = self._decode_bio(chars, char_labels)
        return {
            "titles":   [e["text"] for e in entities if e["type"] == "TITLE"],
            "skills":   [e["text"] for e in entities if e["type"] == "SKILL"],
            "entities": entities,
        }

    def predict_batch(self, texts, batch_size=32):
        """
        批量推理（逐条处理，保持简单）

        Args:
            texts: 文本列表
            batch_size: 暂未使用，保留接口

        Returns:
            list of dict，与 predict() 返回格式相同
        """
        return [self.predict(t) for t in tqdm(texts, desc="NER推理")]


def process_txt_file(predictor, input_path, output_path):
    """处理纯文本文件（每行一条JD）"""
    lines = Path(input_path).read_text(encoding="utf-8").strip().splitlines()
    logger.info("读取 %d 条文本: %s", len(lines), input_path)
    results = predictor.predict_batch(lines)
    rows = []
    for text, r in zip(lines, results):
        rows.append({
            "text":       text[:100],
            "titles":     "|".join(r["titles"]),
            "skills":     "|".join(r["skills"]),
            "n_titles":   len(r["titles"]),
            "n_skills":   len(r["skills"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("结果保存: %s (%d行)", output_path, len(df))
    return df


def process_json_file(predictor, json_path, output_path):
    """
    批量处理 ls_jd_tasks.json 格式数据，输出NER结果到CSV。

    输出列：
      row_id        - 记录唯一ID
      job_title     - 原始职位名称
      jd_snippet    - 原始JD文本（截取前200字符）
      occ_core_gold - 标注的职位核心词（用于对比评估）
      skills_gold   - 标注的技能词（用于对比评估）
      pred_titles   - 模型预测的职位（|分隔）
      pred_skills   - 模型预测的技能（|分隔）
      n_titles      - 预测职位数量
      n_skills      - 预测技能数量
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("读取 %d 条记录: %s", len(data), json_path)

    rows = []
    for item in tqdm(data, desc="批量推理"):
        d = item.get("data", {})
        title_text = str(d.get("clean_title") or d.get("job_title") or "")
        jd_raw     = re.sub(r"<[^>]+>", "", str(d.get("jd_snippet") or ""))
        jd_clean   = re.sub(r"\s+", " ", jd_raw).strip()
        full_text  = (title_text + "\u3002" + jd_clean)[:500]

        r = predictor.predict(full_text)
        rows.append({
            "row_id":        d.get("row_id", ""),
            "job_title":     d.get("job_title", ""),
            "jd_snippet":    jd_clean[:200],
            "occ_core_gold": d.get("occ_core", ""),
            "skills_gold":   d.get("hard_skills", ""),
            "pred_titles":   "|".join(r["titles"]),
            "pred_skills":   "|".join(r["skills"]),
            "n_titles":      len(r["titles"]),
            "n_skills":      len(r["skills"]),
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("结果保存: %s (%d行)", output_path, len(df))

    # 简单统计
    logger.info("提取统计:")
    logger.info("  有职位: %d (%.1f%%)",
                (df["n_titles"] > 0).sum(), 100*(df["n_titles"] > 0).mean())
    logger.info("  有技能: %d (%.1f%%)",
                (df["n_skills"] > 0).sum(), 100*(df["n_skills"] > 0).mean())
    logger.info("  平均技能数: %.2f", df["n_skills"].mean())
    return df


def interactive_mode(predictor):
    """交互模式：逐条输入文本"""
    print("\nBERT-NER 交互模式（输入 quit 退出）")
    print("-" * 50)
    while True:
        text = input("\n输入文本: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        r = predictor.predict(text)
        print("职位:", r["titles"] or "(未识别)")
        print("技能:", r["skills"] or "(未识别)")
        if r["entities"]:
            print("全部实体:")
            for e in r["entities"]:
                print(f"  [{e['type']}] {e['text']} (位置 {e['start']}-{e['end']})")


def main():
    parser = argparse.ArgumentParser(
        description="BERT-NER推理：从招聘JD提取职位和技能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/bert/ner_predict.py --text "负责Java开发，熟悉Spring Boot和Redis"
  python src/bert/ner_predict.py --json data/ls_jd_tasks.json --output output/ner_results.csv
  python src/bert/ner_predict.py --interactive
        """
    )
    parser.add_argument("--model-dir",   type=str, default=None,
                        help="模型目录（默认: output/models/bert_ner）")
    parser.add_argument("--text",        type=str, default=None,
                        help="单条文本推理")
    parser.add_argument("--input",       type=str, default=None,
                        help="输入文本文件（每行一条）")
    parser.add_argument("--json",        type=str, default=None,
                        help="ls_jd_tasks.json 格式文件")
    parser.add_argument("--output",      type=str,
                        default="output/ner_results.csv",
                        help="输出CSV路径")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式")
    args = parser.parse_args()

    predictor = NERPredictor(model_dir=args.model_dir)

    if args.text:
        r = predictor.predict(args.text)
        print("\n输入:", args.text)
        print("职位:", r["titles"] or "(未识别)")
        print("技能:", r["skills"] or "(未识别)")
        print("全部实体:", r["entities"])

    elif args.input:
        process_txt_file(predictor, args.input, args.output)

    elif args.json:
        process_json_file(predictor, args.json, args.output)

    elif args.interactive:
        interactive_mode(predictor)

    else:
        # 默认：对原始JSON数据集批量推理
        json_path = BASE_DIR / "data" / "ls_jd_tasks.json"
        if json_path.exists():
            process_json_file(predictor, str(json_path), args.output)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
