#!/usr/bin/env python
# -*- coding: utf-8 -*-
# bert_ner.py - BERT命名实体识别训练脚本
#
# 功能：从招聘JD文本中提取职位名称(TITLE)和技能词(SKILL)
#
# 模块结构：
#   STEP 1  auto_bio_label / build_bio_dataset  自动生成字符级BIO标注
#   STEP 2  NERDataset                          子词对齐，转换为BERT输入
#   STEP 3  evaluate_ner / train_ner            微调训练 + seqeval评估
#   STEP 4  NERPredictor / check_data / main    推理和工具函数
#
# 标签体系(BIO格式):
#   O=非实体  B-TITLE/I-TITLE=职位  B-SKILL/I-SKILL=技能
#
# 用法:
#   python src/bert/bert_ner.py --check-data  # 验证BIO标注
#   python src/bert/bert_ner.py --train       # 训练
#   python src/bert/bert_ner.py --predict "文本"  # 推理
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import json
import re
import logging
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast, BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent.parent
JSON_PATH     = BASE_DIR / "data" / "ls_jd_tasks.json"
NER_MODEL_DIR = BASE_DIR / "output" / "models" / "bert_ner"
NER_DATA_DIR  = BASE_DIR / "output" / "ner_data"
_LOCAL        = BASE_DIR / "output" / "pretrained" / "chinese-bert-wwm-ext"
PRETRAINED    = str(_LOCAL) if _LOCAL.exists() else "hfl/chinese-bert-wwm-ext"

# ── 标签映射 ──────────────────────────────────────────────
LABEL2ID  = {"O": 0, "B-TITLE": 1, "I-TITLE": 2, "B-SKILL": 3, "I-SKILL": 4}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ── 训练超参数 ─────────────────────────────────────────────
CFG = {
    "pretrained":   PRETRAINED,  # 预训练模型路径
    "max_len":      256,         # 最大token长度
    "batch_size":   64,          # 批大小
    "epochs":       10,          # 最大训练轮数
    "lr":           3e-5,        # 学习率
    "warmup_ratio": 0.1,         # 预热比例
    "patience":     3,           # 早停耐心值
    "seed":         42,          # 随机种子
}
SKILL_SEP = re.compile(r"[\u3001\uff0c,\uff1b;/|]+")  # 技能分隔符
BLACKLIST = {"\uff08\u65e0\uff09", "\u65e0", "", "none", "null", "nan"}


# ════════════════════════════════════════════════
# STEP 1: 自动BIO标注
# 作用：将结构化标注字段对齐到原文字符，生成BIO序列
# ════════════════════════════════════════════════

def auto_bio_label(text, entities):
    # 对原文每个字符生成BIO标签（最长匹配优先）
    # 参数：text=原始文本, entities=[{text,type}, ...]
    # 返回：[(char, label), ...] 字符级标注列表
    labels = ["O"] * len(text)
    for ent in sorted(entities, key=lambda e: len(e["text"]), reverse=True):
        et = ent["text"].strip()
        if not et or len(et) < 2:
            continue
        start = 0
        while True:
            idx = text.lower().find(et.lower(), start)
            if idx == -1:
                break
            end = idx + len(et)
            if all(labels[i] == "O" for i in range(idx, end)):
                labels[idx] = "B-" + ent["type"]
                for i in range(idx + 1, end):
                    labels[i] = "I-" + ent["type"]
            start = idx + 1
    return list(zip(text, labels))


def build_bio_dataset(json_path, max_samples=0):
    # 从ls_jd_tasks.json构建字符级BIO标注数据集
    # 数据来源：clean_title+jd_snippet -> 输入文本
    #           occ_core -> TITLE实体
    #           hard_skills -> SKILL实体（按分隔符拆分）
    # 过滤：文本<10字符、标注后全O的样本跳过
    # 返回：[{row_id, text, chars, labels}, ...]
    logger.info("读取标注数据: %s", json_path)
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    if max_samples > 0:
        raw = raw[:max_samples]
    samples, skip = [], 0
    for item in tqdm(raw, desc="生成BIO标注"):
        d = item.get("data", {})
        title_text = str(d.get("clean_title") or d.get("job_title") or "")
        jd_text    = re.sub(r"<[^>]+>", "", str(d.get("jd_snippet") or ""))
        jd_text    = re.sub(r"\s+", " ", jd_text).strip()
        full_text  = (title_text + "\u3002" + jd_text)[:500]
        if len(full_text) < 10:
            skip += 1
            continue
        entities = []
        occ = str(d.get("occ_core") or "").strip()
        if occ and len(occ) >= 2:
            entities.append({"text": occ, "type": "TITLE"})
        for sk in SKILL_SEP.split(str(d.get("hard_skills") or "")):
            sk = sk.strip()
            if sk and sk not in BLACKLIST and len(sk) >= 2:
                entities.append({"text": sk, "type": "SKILL"})
        char_labels = auto_bio_label(full_text, entities)
        chars  = [c for c, _ in char_labels]
        labels = [l for _, l in char_labels]
        if not any(l != "O" for l in labels):
            skip += 1
            continue
        samples.append({"row_id": d.get("row_id", len(samples)),
                        "text": full_text, "chars": chars, "labels": labels})
    logger.info("有效样本: %d / %d (跳过 %d)", len(samples), len(raw), skip)
    logger.info("标签分布: %s",
                dict(Counter([l for s in samples for l in s["labels"]])))
    return samples


# ════════════════════════════════════════════════
# STEP 2: Dataset（子词对齐）
# 作用：字符级BIO标注 -> BERT token级输入
# 核心：word_ids()将sub-word token映射回字符索引
#       规则：CLS/SEP/padding=-100, 首子词=原始标签, 续子词=-100
# ════════════════════════════════════════════════

class NERDataset(Dataset):
    # BERT-NER数据集，处理中文字符级BIO标注的子词对齐
    #
    # 对齐示例（英文技能词）：
    #   输入: [P,y,t,h,o,n] 标签: [B-SKILL,I-SKILL,I-SKILL,...]
    #   BERT: [CLS, Py, ##thon, SEP]
    #   对齐: [-100, 3,  -100,  -100]  (3=B-SKILL)

    def __init__(self, samples, tokenizer, max_len):
        # samples: build_bio_dataset()返回的列表
        # tokenizer: BertTokenizerFast实例
        # max_len: 最大序列长度（含CLS/SEP）
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 返回 {input_ids, attention_mask, token_type_ids, labels}
        # labels中-100表示忽略（不参与loss计算）
        s = self.samples[idx]
        label_ids = [LABEL2ID.get(l, 0) for l in s["labels"]]
        enc = self.tokenizer(
            s["chars"], is_split_into_words=True,
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)
        aligned, prev_wid = [], None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_wid:
                aligned.append(label_ids[wid] if wid < len(label_ids) else -100)
            else:
                aligned.append(-100)
            prev_wid = wid
        ttype = enc.get("token_type_ids",
            torch.zeros(self.max_len, dtype=torch.long)).squeeze(0)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": ttype,
            "labels":         torch.tensor(aligned, dtype=torch.long),
        }


# STEP 3: 训练与评估
# BertForTokenClassification在每个token输出NUM_LABELS类概率
# 内置CrossEntropyLoss自动忽略label=-100的token
# seqeval进行实体级评估（完整实体匹配，比token级更严格）

def evaluate_ner(model, loader, device):
    # 评估NER模型：优先seqeval实体级F1，降级sklearn token级F1
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tids = batch["token_type_ids"].to(device)
            labs = batch["labels"].to(device)
            out  = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
            preds = out.logits.argmax(-1)
            for pred_row, true_row, mask_row in zip(
                    preds.cpu().tolist(), labs.cpu().tolist(), mask.cpu().tolist()):
                ts, ps = [], []
                for p, t, m in zip(pred_row, true_row, mask_row):
                    if t == -100 or m == 0:
                        continue
                    ts.append(ID2LABEL.get(t, "O"))
                    ps.append(ID2LABEL.get(p, "O"))
                all_true.append(ts)
                all_pred.append(ps)
    try:
        from seqeval.metrics import f1_score, classification_report
        f1 = f1_score(all_true, all_pred)
        report = classification_report(all_true, all_pred)
    except Exception:
        from sklearn.metrics import f1_score as skf1
        flat_t = [l for s in all_true for l in s]
        flat_p = [l for s in all_pred for l in s]
        f1 = skf1(flat_t, flat_p, average="macro", zero_division=0)
        report = "F1(macro)=%.4f" % f1
    return f1, report, all_true, all_pred


def train_ner():
    # BERT-NER训练主函数
    # 流程：加载/缓存BIO数据->8:1:1划分->初始化模型->AdamW+预热->训练+早停->test评估
    import random
    from sklearn.model_selection import train_test_split
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)
    NER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = NER_DATA_DIR / "bio_samples.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            samples = json.load(f)
        logger.info("加载缓存: %d 条", len(samples))
    else:
        samples = build_bio_dataset(JSON_PATH)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)
        logger.info("BIO数据已缓存: %s", cache)
    train_s, tmp_s = train_test_split(samples, test_size=0.2, random_state=42)
    val_s,  test_s = train_test_split(tmp_s,   test_size=0.5, random_state=42)
    logger.info("train=%d val=%d test=%d", len(train_s), len(val_s), len(test_s))
    tokenizer = BertTokenizerFast.from_pretrained(CFG["pretrained"])
    model = BertForTokenClassification.from_pretrained(
        CFG["pretrained"], num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)
    train_loader = DataLoader(NERDataset(train_s, tokenizer, CFG["max_len"]),
        batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(NERDataset(val_s,   tokenizer, CFG["max_len"]),
        batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(NERDataset(test_s,  tokenizer, CFG["max_len"]),
        batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    total_steps  = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, patience_cnt = 0.0, 0
    logger.info("开始训练...")
    for epoch in range(1, CFG["epochs"] + 1):
        model.train(); total_loss = 0.0
        for batch in tqdm(train_loader, desc="Epoch %d" % epoch):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tids = batch["token_type_ids"].to(device)
            labs = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask,
                        token_type_ids=tids, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += out.loss.item()
        train_loss = total_loss / len(train_loader)
        val_f1, val_report, _, _ = evaluate_ner(model, val_loader, device)
        logger.info("Epoch %d: loss=%.4f val_f1=%.4f", epoch, train_loss, val_f1)
        logger.info("Val: %s", val_report)
        if val_f1 > best_f1:
            best_f1 = val_f1; patience_cnt = 0
            model.save_pretrained(str(NER_MODEL_DIR))
            tokenizer.save_pretrained(str(NER_MODEL_DIR))
            logger.info("  -> 保存最佳 f1=%.4f", val_f1)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]: logger.info("早停"); break
    best = BertForTokenClassification.from_pretrained(str(NER_MODEL_DIR)).to(device)
    test_f1, test_report, _, _ = evaluate_ner(best, test_loader, device)
    logger.info("Test F1=%.4f", test_f1)
    logger.info("Test: %s", test_report)


# STEP 4: 推理
# 清洗文本->tokenize->推理->word_ids映射->BIO解码->实体列表

class NERPredictor:
    # BERT-NER推理器，封装完整推理流程

    def __init__(self, model_dir=None):
        # 加载训练好的tokenizer和模型
        mdir = model_dir or NER_MODEL_DIR
        logger.info("加载NER模型: %s", mdir)
        self.tokenizer = BertTokenizerFast.from_pretrained(str(mdir))
        self.model     = BertForTokenClassification.from_pretrained(str(mdir))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        # 单条文本推理，返回 {titles, skills, entities}
        text  = re.sub(r"<[^>]+>", "", text)
        text  = re.sub(r"\s+", " ", text).strip()[:500]
        chars = list(text)
        enc   = self.tokenizer(chars, is_split_into_words=True,
            max_length=CFG["max_len"], truncation=True, return_tensors="pt")
        enc_in = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc_in)
        pred_ids = out.logits.argmax(-1).squeeze(0).cpu().tolist()
        word_ids = self.tokenizer(chars, is_split_into_words=True,
            max_length=CFG["max_len"], truncation=True).word_ids()
        char_labels = [None] * len(chars)
        for ti, wid in enumerate(word_ids):
            if wid is not None and char_labels[wid] is None and ti < len(pred_ids):
                char_labels[wid] = ID2LABEL.get(pred_ids[ti], "O")
        char_labels = [l or "O" for l in char_labels]
        entities, i = [], 0
        while i < len(chars):
            lbl = char_labels[i]
            if lbl.startswith("B-"):
                etype = lbl[2:]; j = i + 1
                while j < len(chars) and char_labels[j] == "I-" + etype: j += 1
                ent_text = "".join(chars[i:j]).strip()
                if ent_text: entities.append({"text": ent_text, "type": etype,
                                               "start": i, "end": j})
                i = j
            else: i += 1
        return {"titles": [e["text"] for e in entities if e["type"] == "TITLE"],
                "skills": [e["text"] for e in entities if e["type"] == "SKILL"],
                "entities": entities}

    def batch_predict(self, texts):
        # 批量推理
        return [self.predict(t) for t in tqdm(texts, desc="NER推理")]


def check_data(n=5):
    # 检查BIO标注质量，打印前n条样例
    samples = build_bio_dataset(JSON_PATH, max_samples=200)
    print("\n=== BIO标注样例 ===")
    for s in samples[:n]:
        chars, labels = s["chars"], s["labels"]
        entities, i = [], 0
        while i < len(chars):
            if labels[i].startswith("B-"):
                t = labels[i][2:]; j = i + 1
                while j < len(chars) and labels[j] == "I-" + t: j += 1
                entities.append("[" + "".join(chars[i:j]) + "](" + t + ")")
                i = j
            else: i += 1
        print("文本:", s["text"][:80])
        print("实体:", entities)
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BERT-NER 职位技能提取")
    parser.add_argument("--train",      action="store_true", help="训练模型")
    parser.add_argument("--predict",    type=str, default=None, help="推理单条文本")
    parser.add_argument("--check-data", action="store_true", help="检查BIO标注质量")
    args = parser.parse_args()
    if args.train:
        train_ner()
    elif args.predict:
        p = NERPredictor()
        r = p.predict(args.predict)
        print("职位:", r["titles"])
        print("技能:", r["skills"])
    elif args.check_data:
        check_data()
    else:
        check_data(3)


if __name__ == "__main__":
    main()
