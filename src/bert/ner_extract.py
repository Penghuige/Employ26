#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NER 提取：职位名称（title）和技能（skill）

两种方法：
  方法一：基于规则 + 现有标注数据词典匹配（推荐）
  方法二：使用预训练NER模型推理

使用：
  # 单条文本提取
  python src/bert/ner_extract.py --text "负责Java后端开发，熟悉Spring Boot"

  # 批量处理整个数据集
  python src/bert/ner_extract.py --batch
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import re
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent.parent
JSON_PATH = BASE_DIR / "data" / "ls_jd_tasks.json"
OUT_PATH  = BASE_DIR / "output" / "ner_results.csv"

SKILL_SEP = re.compile(r'[\u3001\uff0c,\uff1b;/|\\\n\t]+')

BLACKLIST = {
    '\uff08\u65e0\uff09', '\u65e0', '', 'none', 'null', '/', '-', '\u2014', '\u6682\u65e0',
    '\u6c9f\u901a\u80fd\u529b', '\u56e2\u961f\u5408\u4f5c', '\u8d23\u4efb\u5fc3',
    '\u6267\u884c\u529b', '\u6297\u538b\u80fd\u529b', '\u79ef\u6781\u4e3b\u52a8',
    '\u8ba4\u771f\u8d23\u4efb', '\u7ec6\u5fc3', '\u8010\u5fc3', '\u70ed\u60c5',
}


class RuleBasedExtractor:
    """基于规则的职位/技能提取器
    
    从 ls_jd_tasks.json 构建词典，对新文本做最长匹配
    """

    # 额外正则：匹配英文技术栈
    TECH_RE = re.compile(
        r"\b(?:Python|Java|C\+\+|C#|Go|Rust|PHP|Ruby|Swift|Kotlin"
        r"|MySQL|Redis|MongoDB|PostgreSQL|Oracle|Elasticsearch"
        r"|Docker|Kubernetes|K8s|Git|Linux|AWS|Azure|GCP"
        r"|React|Vue|Angular|Node\.js|Django|Flask|FastAPI|Spring"
        r"|TensorFlow|PyTorch|BERT|GPT|SQL|NLP|API|SDK)\b",
        re.IGNORECASE
    )

    def __init__(self, json_path):
        self.title_set = set()
        self.skill_set = set()
        self._build_dicts(json_path)

    def _build_dicts(self, json_path):
        logger.info("构建词典: %s", json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        title_cnt, skill_cnt = defaultdict(int), defaultdict(int)
        for item in data:
            d = item.get("data", {})
            t = str(d.get("occ_core", "") or "").strip()
            if t and t not in BLACKLIST and len(t) >= 2:
                title_cnt[t] += 1
            for sk in SKILL_SEP.split(str(d.get("hard_skills", "") or "")):
                sk = sk.strip()
                if sk and sk not in BLACKLIST and len(sk) >= 2:
                    skill_cnt[sk] += 1
        self.title_set = {t for t, c in title_cnt.items() if c >= 1}
        self.skill_set = {s for s, c in skill_cnt.items() if c >= 2}
        logger.info("词典: titles=%d  skills=%d", len(self.title_set), len(self.skill_set))
        top20 = sorted(skill_cnt, key=skill_cnt.get, reverse=True)[:20]
        logger.info("高频技能 Top20: %s", top20)

    def extract(self, text):
        """返回 {titles: [...], skills: [...]}"""
        text = re.sub(r"<[^>]+>", "", text)  # 去HTML
        text = re.sub(r"\s+", " ", text).strip()
        tl = text.lower()

        # 职位：最长匹配，只取第一个
        titles = []
        for t in sorted(self.title_set, key=len, reverse=True):
            if t in text:
                titles.append(t)
                break

        # 技能：最长匹配，不重叠
        skills = []
        matched = []  # [(start, end)]
        for sk in sorted(self.skill_set, key=len, reverse=True):
            idx = tl.find(sk.lower())
            while idx != -1:
                end = idx + len(sk)
                overlap = any(not (end <= ms or idx >= me) for ms, me in matched)
                if not overlap:
                    skills.append(sk)
                    matched.append((idx, end))
                idx = tl.find(sk.lower(), idx + 1)

        # 正则补充英文技术栈
        for m in self.TECH_RE.finditer(text):
            w = m.group()
            if w not in skills:
                skills.append(w)

        # 技能大小写归一化：统一为首字母大写（保留全大写缩写）
        def normalize(s):
            if s.isupper() and len(s) <= 5: return s  # 保留 SQL/API/NLP
            return s[0].upper() + s[1:] if s else s
        skills_norm = []
        seen = set()
        for sk in skills:
            key = sk.lower()
            if key not in seen:
                seen.add(key)
                skills_norm.append(normalize(sk))
        return {"titles": list(dict.fromkeys(titles)),
                "skills": skills_norm}

    def batch_extract(self, records):
        rows = []
        for item in tqdm(records, desc="NER提取"):
            d = item.get("data", {})
            txt = str(d.get("clean_title", "") or "") + " " + str(d.get("jd_snippet", "") or "")
            res = self.extract(txt)
            rows.append({
                "row_id":        d.get("row_id", ""),
                "job_title":     d.get("job_title", ""),
                "occ_core_gold": d.get("occ_core", ""),
                "skills_gold":   d.get("hard_skills", ""),
                "pred_titles":   "|".join(res["titles"]),
                "pred_skills":   "|".join(res["skills"]),
                "n_skills":      len(res["skills"]),
            })
        return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="NER提取职位和技能")
    parser.add_argument("--text", type=str, default=None, help="单条文本")
    parser.add_argument("--batch", action="store_true", help="批量处理全数据集")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（0=全部）")
    args = parser.parse_args()

    extractor = RuleBasedExtractor(JSON_PATH)

    if args.text:
        # 单条文本提取
        result = extractor.extract(args.text)
        print("\n输入文本:", args.text)
        print("职位 (titles):", result['titles'])
        print("技能 (skills):", result['skills'])

    elif args.batch:
        # 批量处理
        logger.info("读取 %s ...", JSON_PATH)
        with open(JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if args.limit > 0:
            data = data[:args.limit]
        logger.info("处理 %d 条记录...", len(data))
        df = extractor.batch_extract(data)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        logger.info("结果保存: %s  (%d行)", OUT_PATH, len(df))

        # 简单评估
        total = len(df)
        has_skill = (df["n_skills"] > 0).sum()
        avg_skills = df["n_skills"].mean()
        logger.info("\n评估结果:")
        logger.info("  总记录数: %d", total)
        logger.info("  有技能提取: %d (%.1f%%)", has_skill, 100*has_skill/total)
        logger.info("  平均技能数: %.2f", avg_skills)
        logger.info("  Top技能示例:")
        all_skills = []
        for skills_str in df["pred_skills"].dropna():
            all_skills.extend([s for s in skills_str.split("|") if s])
        from collections import Counter
        top_skills = Counter(all_skills).most_common(20)
        for sk, cnt in top_skills:
            logger.info("    %-20s %d", sk, cnt)
    else:
        # 默认：演示模式
        demo_texts = [
            "负责Java后端开发，熟悉Spring Boot、Redis、MySQL，有Docker部署经验",
            "Python数据分析工程师，要求掌握Pandas、NumPy、SQL，了解机器学习",
            "产品经理，负责需求分析和产品规划，熟悉Axure、Figma",
            "前端开发，精通React/Vue，熟悉TypeScript，了解Node.js",
        ]
        print("\n=== NER提取演示 ===")
        for text in demo_texts:
            result = extractor.extract(text)
            print(f"\n文本: {text}")
            print(f"职位: {result['titles']}")
            print(f"技能: {result['skills']}")


if __name__ == "__main__":
    main()