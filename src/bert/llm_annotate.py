#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_annotate.py - 使用本地大模型（Ollama）对招聘JD进行NER标注

功能：
  1. 调用本地 Ollama 服务，对 jd_snippet 进行 NER 标注
  2. 支持多实体类型：TITLE / SKILL / CERT / LANG / EDU / EXP / MAJOR / DOMAIN
  3. 输出三种格式：
     - JSONL（Label Studio 导入格式，断点续传）
     - CoNLL（BERT fine-tune 标准格式，字符级BIO）
     - CSV（人工审核用）
  4. 断点续传：已标注的 row_id 自动跳过
  5. 解析容错：自动提取模型输出中的 JSON 块

用法：
  # 确保 Ollama 服务已启动并拉取模型
  ollama serve
  ollama pull qwen2.5:7b

  # 预览前20条（不写磁盘）
  python src/bert/llm_annotate.py --limit 20 --preview

  # 全量标注
  python src/bert/llm_annotate.py --model qwen2.5:7b

  # 仅导出 CoNLL 格式（从已有 JSONL 转换，不重新调用LLM）
  python src/bert/llm_annotate.py --export-conll

  # 指定模型和并发数
  python src/bert/llm_annotate.py --model deepseek-r1:7b --limit 500

实体类型：
  TITLE  职位核心词        销售经理、风险政策岗
  SKILL  硬技能/工具/框架  Python、SQL、Spring Boot
  CERT   证书/资质         CPA、PMP、驾驶证C1
  LANG   语言能力          英语CET-6、日语N2
  EDU    学历要求          本科、硕士及以上
  EXP    经验要求          2年以上、3-5年工作经验
  MAJOR  专业要求          计算机科学、统计学
  DOMAIN 行业领域知识      互联网贷款风险管理
"""

import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── 路径配置 ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent.parent
JSON_PATH     = BASE_DIR / "data" / "ls_jd_tasks.json"
OUT_DIR       = BASE_DIR / "output" / "ner_annotation"
ANNO_JSONL    = OUT_DIR / "ner_llm_annotations.jsonl"
CONLL_PATH    = OUT_DIR / "ner_train.conll"
REVIEW_CSV    = OUT_DIR / "ner_review.csv"

# ── 合法实体类型 ────────────────────────────────────────────────────────
VALID_TYPES = {"TITLE", "SKILL", "CERT", "LANG", "EDU", "EXP", "MAJOR", "DOMAIN"}

# ── System Prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是专业的招聘信息NER标注专家。从招聘JD中识别以下实体类型：

TITLE  : 职位核心词，如：销售经理、数据分析师、风险政策岗
SKILL  : 硬技能/工具/框架，如：Python、SQL、Spring Boot、Docker、Tableau
          【仅标注可传授、可验证的具体技术，排除"沟通能力""责任心"等软素质】
CERT   : 证书/资质，如：CPA、PMP、驾驶证C1、CFA、建造师证
LANG   : 语言能力，如：英语CET-6、日语N2、英语口语
EDU    : 学历要求，如：本科、硕士及以上、大专以上
EXP    : 工作经验要求，如：2年以上、3-5年工作经验、应届生
MAJOR  : 专业要求，如：计算机科学、统计学、金融学、医学检验
DOMAIN : 行业/领域知识，如：互联网贷款风险管理、快消品市场推广

【必须排除，不标注】
软素质: 责任心、沟通能力、抗压能力、团队合作、执行力、积极主动
泛动作: 负责、管理、执行、推进、协作、配合
福利:   五险一金、餐补、年终奖、带薪年假
条件:   加班、出差、驻外、轮班
特质:   认真、细心、耐心、热情

【输出要求】
- 只输出纯JSON，不要任何解释
- text必须是原文中的完整字符串（不要修改或截断）
- entities按原文出现顺序排列"""

USER_TEMPLATE = """职位名称：{job_title}
JD内容：{jd_text}

输出格式：
{{"entities": [{{"text": "原文字符串", "type": "实体类型"}}]}}"""


# ════════════════════════════════════════════════════════════════════════
# Ollama 调用封装
# ════════════════════════════════════════════════════════════════════════

class OllamaAnnotator:
    """
    调用本地 Ollama 服务进行 NER 标注。

    推荐模型（按质量/速度排序）：
      qwen2.5:7b     中文理解强，速度快（推荐首选）
      qwen2.5:14b    质量更高，需要更多显存
      deepseek-r1:7b 推理强，速度较慢
      qwen3:8b       支持思维链，适合复杂标注
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: int = 90,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model       = model
        self.host        = host.rstrip("/")
        self.timeout     = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._check_service()

    def _check_service(self):
        """检查 Ollama 服务连通性，并验证目标模型已拉取。"""
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(self.host, timeout=5) as r:
                logger.info("Ollama 服务正常: %s", self.host)
        except Exception as e:
            raise RuntimeError(
                f"无法连接 Ollama ({self.host})。\n"
                f"请先运行: ollama serve\n原始错误: {e}"
            )
        # 检查模型是否存在
        try:
            url = f"{self.host}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            names = [m["name"] for m in data.get("models", [])]
            # 模糊匹配（忽略tag部分）
            base = self.model.split(":")[0]
            found = any(base in n for n in names)
            if not found:
                logger.warning(
                    "模型 %s 未在本地找到，可用模型: %s\n"
                    "请先运行: ollama pull %s",
                    self.model, names, self.model
                )
            else:
                logger.info("模型 %s 已就绪", self.model)
        except Exception:
            pass  # tags 接口不影响主流程

    def _call(self, user_prompt: str) -> Optional[str]:
        """向 Ollama /api/chat 发送请求，返回模型输出字符串。"""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p":       0.9,
                "num_predict": 1024,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["message"]["content"]
            except Exception as e:
                logger.warning("Ollama 请求失败 (%d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return None

    @staticmethod
    def _parse(raw: str) -> list:
        """
        从模型输出中提取 entities 列表。
        容错顺序：直接解析 → 提取代码块 → 提取首个{}块
        """
        if not raw:
            return []
        for candidate in [
            raw.strip(),
            # 提取 ```json...``` 或 ```...```
            (re.search(r"```(?:json)?\s*([\s\S]+?)```", raw) or type('', (), {'group': lambda s, i: None})()).group(1),
            # 提取最外层 {...}
            (re.search(r"\{[\s\S]+\}", raw) or type('', (), {'group': lambda s, i: None})()).group(0),
        ]:
            if not candidate:
                continue
            try:
                obj = json.loads(candidate.strip())
                if isinstance(obj, dict):
                    return obj.get("entities", [])
            except (json.JSONDecodeError, TypeError):
                pass
        logger.debug("无法解析模型输出: %.200s", raw)
        return []

    @staticmethod
    def _validate(entities: list, full_text: str) -> list:
        """
        过滤不合法实体：
          - type 必须在 VALID_TYPES 中
          - text 必须在原文中真实出现
          - 去重
        """
        seen, result = set(), []
        for e in entities:
            etype = str(e.get("type", "")).strip().upper()
            etext = str(e.get("text", "")).strip()
            if not etext:
                continue
            if etype not in VALID_TYPES:
                continue
            if etext not in full_text:
                logger.debug("实体未在原文中找到，跳过: [%s]", etext)
                continue
            key = (etext, etype)
            if key in seen:
                continue
            seen.add(key)
            result.append({"text": etext, "type": etype})
        return result

    def annotate(self, row_id: int, job_title: str, jd_text: str) -> dict:
        """
        对单条 JD 进行标注。

        Returns:
            {
              "row_id":    int,
              "job_title": str,
              "jd_text":   str,   # 清洗后的文本
              "entities":  list,  # [{text, type}, ...]
              "raw_model_output": str,
              "parse_ok":  bool,
            }
        """
        # 清洗文本
        jd_clean = re.sub(r"<[^>]+>", "", jd_text)
        jd_clean = re.sub(r"\s+", " ", jd_clean).strip()[:600]
        full_text = job_title + "。" + jd_clean

        prompt = USER_TEMPLATE.format(job_title=job_title, jd_text=jd_clean)
        raw    = self._call(prompt)
        parsed = self._parse(raw or "")
        valid  = self._validate(parsed, full_text)

        return {
            "row_id":           row_id,
            "job_title":        job_title,
            "jd_text":          jd_clean,
            "entities":         valid,
            "raw_model_output": (raw or "")[:500],
            "parse_ok":         bool(valid),
        }


# ════════════════════════════════════════════════════════════════════════
# 格式转换：annotations -> CoNLL
# ════════════════════════════════════════════════════════════════════════

def entities_to_bio(text: str, entities: list) -> list:
    """
    将实体列表转为字符级 BIO 标签序列。

    策略：最长实体优先，不允许重叠标注。

    Returns:
        [(char, label), ...]
    
    """
    labels = ['O'] * len(text)
    for ent in sorted(entities, key=lambda e: len(e['text']), reverse=True):
        etext = ent['text']
        etype = ent['type']
        start = 0
        while True:
            idx = text.find(etext, start)
            if idx == -1:
                break
            end = idx + len(etext)
            if all(labels[i] == 'O' for i in range(idx, end)):
                labels[idx] = f'B-{etype}'
                for i in range(idx + 1, end):
                    labels[i] = f'I-{etype}'
            start = idx + 1
    return list(zip(text, labels))


def annotation_to_conll(record: dict) -> str:
    """
    Convert one annotation record to CoNLL format string.
    Each line: char TAB label. Sentences separated by blank lines.
    """
    text     = record["job_title"] + "." + record["jd_text"]
    entities = record["entities"]
    bio      = entities_to_bio(text, entities)
    SENT_SEP = set(".!?\n")
    lines, buf = [], []
    for char, label in bio:
        buf.append(char + "\t" + label)
        if char in SENT_SEP and len(buf) >= 10:
            lines.extend(buf)
            lines.append("")
            buf = []
    if buf:
        lines.extend(buf)
        lines.append("")
    return "\n".join(lines)


def load_done_ids(jsonl_path: Path) -> set:
    """Load completed row_ids from existing JSONL for resume support."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["row_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run_annotation(
    model:   str  = "qwen2.5:7b",
    host:    str  = "http://localhost:11434",
    limit:   int  = 0,
    preview: bool = False,
    timeout: int  = 90,
):
    """
    Main annotation loop.
    Reads ls_jd_tasks.json, calls LLM for each record, appends to JSONL.
    Supports resume: already-annotated row_ids are skipped.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Reading data: %s", JSON_PATH)
    with open(JSON_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)
    if limit > 0:
        raw_data = raw_data[:limit]
    logger.info("Total records: %d", len(raw_data))
    done_ids = load_done_ids(ANNO_JSONL)
    if done_ids:
        logger.info("Resume: skipping %d already-done records", len(done_ids))
    annotator = OllamaAnnotator(model=model, host=host, timeout=timeout)
    jsonl_fh  = None if preview else open(ANNO_JSONL, "a", encoding="utf-8")
    stats = {"total": 0, "ok": 0, "fail": 0, "skip": 0}
    t0 = time.time()
    try:
        pbar = tqdm(raw_data, desc="LLM-NER")
        preview_count = 0
        for item in pbar:
            d      = item.get("data", {})
            row_id = int(d.get("row_id", -1))
            title  = str(d.get("clean_title") or d.get("job_title") or "").strip()
            jd_raw = str(d.get("jd_snippet") or "").strip()
            if row_id in done_ids:
                stats["skip"] += 1
                continue
            stats["total"] += 1
            record = annotator.annotate(row_id, title, jd_raw)
            if record["parse_ok"]:
                stats["ok"] += 1
            else:
                stats["fail"] += 1
            pbar.set_postfix(ok=stats["ok"], fail=stats["fail"])
            if preview:
                preview_count += 1
                print("\n" + "=" * 60)
                print(f"row_id={row_id}  title={title}")
                print("JD: " + record["jd_text"][:150])
                print("Entities:")
                for e in record["entities"]:
                    print(f"  [{e['type']:6s}] {e['text']}")
                if preview_count >= 5:
                    break
            else:
                jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fh.flush()
    finally:
        if jsonl_fh:
            jsonl_fh.close()
    elapsed = time.time() - t0
    logger.info(
        "Done: total=%d ok=%d fail=%d skip=%d  elapsed=%.1fs",
        stats["total"], stats["ok"], stats["fail"], stats["skip"], elapsed,
    )
    if stats["total"] > 0:
        logger.info(
            "Success rate: %.1f%%  Speed: %.2f records/min",
            100 * stats["ok"] / stats["total"],
            stats["total"] / elapsed * 60,
        )


def export_conll():
    """
    Export CoNLL format (char-level BIO) from existing JSONL.
    Can be re-run without calling LLM again.
    Output: output/ner_annotation/ner_train.conll
    """
    from collections import Counter
    if not ANNO_JSONL.exists():
        logger.error("Annotation file missing: %s", ANNO_JSONL)
        return
    records = []
    with open(ANNO_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info("Loaded %d annotation records", len(records))
    conll_blocks = []
    for rec in tqdm(records, desc="to-CoNLL"):
        block = annotation_to_conll(rec)
        if block.strip():
            conll_blocks.append(block)
    with open(CONLL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(conll_blocks))
    logger.info("CoNLL saved: %s", CONLL_PATH)
    label_cnt = Counter()
    for block in conll_blocks:
        for line in block.splitlines():
            if "\t" in line:
                label_cnt[line.split("\t")[1]] += 1
    logger.info("Label distribution: %s", dict(label_cnt.most_common()))
    total_t = sum(label_cnt.values())
    non_o   = total_t - label_cnt.get("O", 0)
    logger.info("Total tokens: %d  Entity ratio: %.1f%%",
                total_t, 100 * non_o / max(total_t, 1))


def export_review_csv():
    """
    Export human-review CSV: each entity type in its own column.
    Output: output/ner_annotation/ner_review.csv
    """
    if not ANNO_JSONL.exists():
        logger.error("Annotation file missing: %s", ANNO_JSONL)
        return
    rows = []
    with open(ANNO_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_type = {t: [] for t in VALID_TYPES}
            for e in rec.get("entities", []):
                by_type.get(e["type"], []).append(e["text"])
            rows.append({
                "row_id":    rec["row_id"],
                "job_title": rec["job_title"],
                "jd_text":   rec["jd_text"][:150],
                "parse_ok":  rec["parse_ok"],
                **{t: " | ".join(v) for t, v in by_type.items()},
            })
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")
    logger.info("Review CSV saved: %s  (%d rows)", REVIEW_CSV, len(df))
    logger.info("Entity type coverage:")
    for t in sorted(VALID_TYPES):
        if t in df.columns:
            pct = (df[t].str.len() > 0).mean() * 100
            logger.info("  %-8s %.1f%%", t, pct)


def print_sample(n: int = 5):
    """Print first N annotated records for quick quality check."""
    if not ANNO_JSONL.exists():
        logger.error("Annotation file missing: %s", ANNO_JSONL)
        return
    records = []
    with open(ANNO_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            if len(records) >= n:
                break
    print("\n" + "=" * 70)
    print(f"Sample annotations (first {n})")
    print("=" * 70)
    for rec in records:
        print(f"\nrow_id={rec['row_id']}  title={rec['job_title']}")
        print("JD: " + rec["jd_text"][:120])
        print("Entities:")
        by_type: dict = {}
        for e in rec.get("entities", []):
            by_type.setdefault(e["type"], []).append(e["text"])
        for t, vs in sorted(by_type.items()):
            print(f"  [{t:6s}] {' | '.join(vs)}")
        print(f"  parse_ok={rec['parse_ok']}")


# ════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Annotate recruitment JDs with local Ollama LLM for NER training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/bert/llm_annotate.py --limit 20 --preview
  python src/bert/llm_annotate.py --model qwen2.5:7b
  python src/bert/llm_annotate.py --model qwen2.5:3b --limit 1000
  python src/bert/llm_annotate.py --export-conll
  python src/bert/llm_annotate.py --export-csv
  python src/bert/llm_annotate.py --sample 10
  python src/bert/llm_annotate.py --all-formats"""
    )
    parser.add_argument("--model",        default="qwen2.5:7b",
                        help="Ollama model name (default: qwen2.5:7b)")
    parser.add_argument("--host",         default="http://localhost:11434",
                        help="Ollama service URL")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Max records to process (0=all)")
    parser.add_argument("--timeout",      type=int, default=90,
                        help="Per-request timeout in seconds")
    parser.add_argument("--preview",      action="store_true",
                        help="Preview mode: print first 5, no disk write")
    parser.add_argument("--export-conll", action="store_true",
                        help="Export CoNLL from existing JSONL (no LLM call)")
    parser.add_argument("--export-csv",   action="store_true",
                        help="Export review CSV from existing JSONL")
    parser.add_argument("--sample",       type=int, default=0,
                        help="Print first N annotated samples")
    parser.add_argument("--all-formats",  action="store_true",
                        help="After annotation export both CoNLL and CSV")
    args = parser.parse_args()

    if args.export_conll:
        export_conll()
    elif args.export_csv:
        export_review_csv()
    elif args.sample > 0:
        print_sample(args.sample)
    else:
        run_annotation(
            model=args.model,
            host=args.host,
            limit=args.limit,
            preview=args.preview,
            timeout=args.timeout,
        )
        if args.all_formats and not args.preview:
            export_conll()
            export_review_csv()


if __name__ == "__main__":
    main()
