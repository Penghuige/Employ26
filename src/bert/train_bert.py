#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import json
import logging
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.model_platform.torch_runtime import resolve_model_path, resolve_torch_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR  = Path(__file__).parent.parent.parent
DB_PATH   = BASE_DIR / "output" / "recruit.duckdb"
MODEL_DIR = BASE_DIR / "output" / "models" / "bert_occ_category"
_LOCAL = Path("output/pretrained/chinese-bert-wwm-ext")
_PRETRAINED = str(_LOCAL) if _LOCAL.exists() else str(resolve_model_path("bert"))
CFG = {"pretrained": _PRETRAINED, "max_len": 128, "batch_size": 32,
       "epochs": 10, "lr": 2e-5, "warmup_ratio": 0.1, "patience": 3, "seed": 42}


class JDDataset(__import__('torch').utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        import torch
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt")
        ttype = enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long))
        if hasattr(ttype, "squeeze"): ttype = ttype.squeeze(0)
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": ttype,
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def set_seed(seed):
    import torch, numpy as np
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_db_data(db_path):
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    num_labels = con.execute("SELECT COUNT(*) FROM label_map").fetchone()[0]
    label_map = {r[0]: r[1] for r in con.execute(
        "SELECT label_id, occ_category FROM label_map ORDER BY label_id").fetchall()}
    train_df = con.execute("SELECT text, label, row_id FROM train_set").df()
    val_df   = con.execute("SELECT text, label, row_id FROM val_set").df()
    test_df  = con.execute("SELECT text, label, row_id FROM test_set").df()
    con.close()
    return num_labels, label_map, train_df, val_df, test_df


def write_metric(db_path, epoch, split, loss, acc, f1):
    import duckdb
    con = duckdb.connect(str(db_path))
    con.execute(
        "INSERT INTO train_metrics (epoch, split, loss, accuracy, f1_macro) VALUES (?,?,?,?,?)",
        [epoch, split, float(loss), float(acc), float(f1)])
    con.close()


def write_predictions(db_path, pred_df):
    import duckdb
    con = duckdb.connect(str(db_path))
    con.execute("DELETE FROM predictions")
    con.execute("INSERT INTO predictions SELECT * FROM pred_df")
    con.close()


def evaluate(model, loader, device):
    import torch
    from torch import nn
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, all_true, all_pred = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tids = batch["token_type_ids"].to(device)
            labs = batch["label"].to(device)
            out  = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
            loss = criterion(out.logits, labs)
            total_loss += loss.item()
            all_pred.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(labs.cpu().tolist())
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1, all_true, all_pred


def main():
    import torch, json, pandas as pd
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
    from tqdm import tqdm

    set_seed(CFG["seed"])
    device = torch.device(resolve_torch_device())
    logger.info("Using device: %s", device)

    logger.info("Reading data from DuckDB...")
    num_labels, label_map, train_df, val_df, test_df = read_db_data(DB_PATH)
    logger.info("Classes: %d  train=%d val=%d test=%d",
                num_labels, len(train_df), len(val_df), len(test_df))

    train_texts  = train_df["text"].tolist(); train_labels = train_df["label"].tolist()
    val_texts    = val_df["text"].tolist();   val_labels   = val_df["label"].tolist()
    test_texts   = test_df["text"].tolist();  test_labels  = test_df["label"].tolist()
    test_row_ids = test_df["row_id"].tolist()

    logger.info("Loading model: %s", CFG["pretrained"])
    tokenizer = BertTokenizerFast.from_pretrained(CFG["pretrained"])
    model = BertForSequenceClassification.from_pretrained(
        CFG["pretrained"], num_labels=num_labels, ignore_mismatched_sizes=True).to(device)

    train_loader = DataLoader(JDDataset(train_texts, train_labels, tokenizer, CFG["max_len"]),
        batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(JDDataset(val_texts, val_labels, tokenizer, CFG["max_len"]),
        batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(JDDataset(test_texts, test_labels, tokenizer, CFG["max_len"]),
        batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    total_steps  = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1, patience_cnt = 0.0, 0
    logger.info("Training...")

    for epoch in range(1, CFG["epochs"] + 1):
        model.train(); total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            ids  = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            tids = batch["token_type_ids"].to(device); labs = batch["label"].to(device)
            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
            loss = criterion(out.logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        logger.info("Epoch %d: train=%.4f val_loss=%.4f acc=%.4f f1=%.4f",
                    epoch, train_loss, val_loss, val_acc, val_f1)
        write_metric(DB_PATH, epoch, "train", train_loss, 0.0, 0.0)
        write_metric(DB_PATH, epoch, "val", val_loss, val_acc, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; patience_cnt = 0
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(MODEL_DIR))
            logger.info("  -> Saved best model (f1=%.4f)", val_f1)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                logger.info("Early stop"); break

    logger.info("Testing best model...")
    best_model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    test_loss, test_acc, test_f1, true_labs, pred_labs = evaluate(best_model, test_loader, device)
    logger.info("Test: loss=%.4f acc=%.4f f1=%.4f", test_loss, test_acc, test_f1)
    write_metric(DB_PATH, -1, "test", test_loss, test_acc, test_f1)

    pred_df = pd.DataFrame({
        "row_id": test_row_ids, "true_label": true_labs, "pred_label": pred_labs,
        "true_category": [label_map.get(l, str(l)) for l in true_labs],
        "pred_category":  [label_map.get(l, str(l)) for l in pred_labs],
        "confidence": [1.0] * len(pred_labs)})
    write_predictions(DB_PATH, pred_df)
    logger.info("Predictions written: %d rows", len(pred_df))

    with open(MODEL_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({**CFG, "num_labels": num_labels, "best_val_f1": best_val_f1,
                   "test_acc": test_acc, "test_f1": test_f1}, f, ensure_ascii=False, indent=2)
    logger.info("Done. Model: %s", MODEL_DIR)


if __name__ == "__main__":
    main()
