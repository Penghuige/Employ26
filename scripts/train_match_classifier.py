"""训练职业细类匹配分类器（XGBoost）。

从 recruit.main.match_training_features 读取特征，
训练 6 类分类器 (A/B/C/D/E/NONE)。

用法:
    python scripts/train_match_classifier.py
    python scripts/train_match_classifier.py --no-test  # 跳过 test set 评估
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_classifier")

DUCKDB_PATH = str(PROJECT_ROOT / "output" / "recruit.duckdb")
FEATURE_TABLE = "recruit.main.match_training_features"
TEST_SIZE = 0.2
RANDOM_STATE = 42
PER_CAND_COLS = ["bge_sim", "bm25_rank", "cat_id"]
AGG_COLS = ["agg_max_sim", "agg_min_sim", "agg_mean_sim",
            "agg_std_sim", "agg_sim_gap", "agg_unique_cats"]


# ===================================================================
# Feature Builder
# ===================================================================
class FeatureBuilder:
    def __init__(self):
        self.feature_cols: List[str] = []
        self.le: LabelEncoder = LabelEncoder()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.feature_cols = []
        for letter in "ABCDE":
            for feat in PER_CAND_COLS:
                self.feature_cols.append(f"cand_{letter}_{feat}")
        for col in AGG_COLS:
            if col in df.columns:
                self.feature_cols.append(col)
        for pos in "ABCDE":
            col = f"bge_top1_{pos}"
            df[col] = (df.get("agg_top1_bge_pos", "") == pos).astype(int)
            self.feature_cols.append(col)
        for letter in "ABCDE":
            src_col = f"cand_{letter}_source"
            if src_col in df.columns:
                df[f"{letter}_src_tier2"] = df[src_col].str.startswith("tier2").astype(int)
                df[f"{letter}_src_tier3"] = df[src_col].str.startswith("tier3").astype(int)
                df[f"{letter}_src_other"] = (
                    (~df[src_col].str.startswith("tier2")) &
                    (~df[src_col].str.startswith("tier3"))
                ).astype(int)
            self.feature_cols.extend([
                f"{letter}_src_tier2", f"{letter}_src_tier3", f"{letter}_src_other",
            ])
        X = self._build_matrix(df)
        y = self.le.fit_transform(df["label"].values)
        self._fitted = True
        logger.info("特征: X=%s, 类别=%d", X.shape, len(self.le.classes_))
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("请先调用 fit_transform")
        for pos in "ABCDE":
            col = f"bge_top1_{pos}"
            if col not in df.columns:
                df[col] = (df.get("agg_top1_bge_pos", "") == pos).astype(int)
        for letter in "ABCDE":
            src_col = f"cand_{letter}_source"
            for st in ["tier2", "tier3", "other"]:
                fname = f"{letter}_src_{st}"
                if fname not in df.columns:
                    if src_col in df.columns:
                        if st == "tier2":
                            df[fname] = df[src_col].str.startswith("tier2").astype(int)
                        elif st == "tier3":
                            df[fname] = df[src_col].str.startswith("tier3").astype(int)
                        else:
                            df[fname] = (
                                (~df[src_col].str.startswith("tier2")) &
                                (~df[src_col].str.startswith("tier3"))
                            ).astype(int)
                    else:
                        df[fname] = 0
        valid = df["label"].isin(self.le.classes_)
        X = self._build_matrix(df)
        y = self.le.transform(df.loc[valid, "label"].values)
        return X[valid.values], y

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0
        return df[self.feature_cols].fillna(0).values.astype(np.float32)


# ===================================================================
# Evaluation
# ===================================================================
def evaluate(model, X: np.ndarray, y: np.ndarray, le: LabelEncoder,
             name: str) -> float:
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n{'='*60}")
    print(f"  {name} (n={len(y)})")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    for line in classification_report(y, y_pred, target_names=le.classes_, digits=4).split("\n"):
        print(f"  {line}")
    cm = confusion_matrix(y, y_pred)
    labels = le.classes_
    header = "       " + "".join(f"{l:>6s}" for l in labels)
    print(f"  {header}")
    for i, label in enumerate(labels):
        row = f"  {label:>5s} " + "".join(f"{cm[i,j]:>6d}" for j in range(len(labels)))
        print(row)
    return acc


def compute_baselines(df: pd.DataFrame, fb: FeatureBuilder) -> None:
    X, y = fb.transform(df)
    n = len(y)
    results = {}
    # always pick majority
    results["majority"] = max(np.mean(y == c) for c in range(len(fb.le.classes_)))
    # always A
    if "A" in fb.le.classes_:
        results["always_A"] = np.mean(y == fb.le.transform(["A"])[0])
    # always NONE
    if "NONE" in fb.le.classes_:
        results["always_NONE"] = np.mean(y == fb.le.transform(["NONE"])[0])
    # BGE max sim
    sim_cols = [f"cand_{l}_bge_sim" for l in "ABCDE"]
    sims = df[sim_cols].values
    best = [chr(65 + i) for i in np.argmax(sims, axis=1)]
    valid = [b in fb.le.classes_ for b in best]
    if all(valid):
        results["bge_max"] = accuracy_score(y, fb.le.transform(best))
    # BM25 top1
    rank_cols = [f"cand_{l}_bm25_rank" for l in "ABCDE"]
    ranks = df[rank_cols].fillna(99).values
    best_r = [chr(65 + i) for i in np.argmin(ranks, axis=1)]
    valid_r = [b in fb.le.classes_ for b in best_r]
    if all(valid_r):
        results["bm25_top1"] = accuracy_score(y[valid_r], fb.le.transform(
            [b for b, v in zip(best_r, valid_r) if v]))
    print(f"\n  Baselines:")
    for k, v in results.items():
        print(f"    {k:15s}: {v:.4f}")


# ===================================================================
# XGBoost Training
# ===================================================================
def train_xgboost(X_train, y_train, X_val, y_val, num_classes):
    import xgboost as xgb
    counts = np.bincount(y_train, minlength=num_classes)
    weights = np.array([max(counts) / max(c, 1) for c in counts])
    sw = weights[y_train]
    model = xgb.XGBClassifier(
        objective="multi:softmax", num_class=num_classes,
        max_depth=6, learning_rate=0.05, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)
    return model


# ===================================================================
# Main
# ===================================================================
def load_old_test_labels(conn) -> Dict[int, str]:
    """加载旧 5170 标注的多数票 label。"""
    rows = conn.execute("""
        SELECT task_id, best_candidate, COUNT(*) as cnt
        FROM recruit.main.label_studio_annotations_v2
        WHERE best_candidate != ''
        GROUP BY task_id, best_candidate
    """).fetchall()
    tid_to_best = defaultdict(Counter)
    for tid, choice, cnt in rows:
        c = choice.replace("候选", "").strip().replace("以上选项都不属于", "NONE")
        tid_to_best[tid][c] += cnt
    return {tid: cntr.most_common(1)[0][0] for tid, cntr in tid_to_best.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    conn = duckdb.connect(DUCKDB_PATH)

    # --- 加载旧 5170 的 task_id (作为独立 test set，不参与训练) ---
    old_test_ids: set = set()
    raw_file = str(PROJECT_ROOT / "output" / "deepseek_relabel" /
                   "deepseek_relabel_raw.jsonl")
    if os.path.exists(raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    old_test_ids.add(json.loads(line)["task_id"])

    old_labels = load_old_test_labels(conn) if not args.no_test else {}

    # --- 加载全部特征 ---
    df_all = conn.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
    logger.info("加载 %d 行", len(df_all))

    # 分离 train (新数据) 和 test (旧 5170)
    mask_train = ~df_all["task_id"].isin(old_test_ids)
    df_train = df_all[mask_train].copy()
    mask_test = df_all["task_id"].isin(old_test_ids)
    df_test_raw = df_all[mask_test].copy()
    logger.info("Train (新数据): %d, Test (旧5170): %d",
                 len(df_train), len(df_test_raw))

    # --- Train/Val split ---
    fb = FeatureBuilder()
    X_train_all, y_train_all = fb.fit_transform(df_train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_train_all,
    )
    logger.info("Train: %d, Val: %d", len(X_train), len(X_val))
    logger.info("LabelEncoder classes: %s", list(fb.le.classes_))

    # Baselines (on val)
    n_train = len(X_train)
    val_indices_in_df = df_train.index[-len(X_val):]
    val_df = df_train.loc[val_indices_in_df].copy()
    val_df["label"] = fb.le.inverse_transform(y_val)
    compute_baselines(val_df, fb)

    # --- Train ---
    t0 = time.time()
    model = train_xgboost(X_train, y_train, X_val, y_val, len(fb.le.classes_))
    logger.info("训练完成 (%.1fs)", time.time() - t0)

    # --- Evaluate ---
    evaluate(model, X_train, y_train, fb.le, "Train Set")
    val_acc = evaluate(model, X_val, y_val, fb.le, "Validation Set (新数据)")

    # Feature importance
    print(f"\n{'='*60}")
    print(f"  Top-20 特征重要性")
    print(f"{'='*60}")
    for name, imp in sorted(zip(fb.feature_cols, model.feature_importances_),
                            key=lambda x: -x[1])[:20]:
        print(f"  {name:30s}: {imp:.4f}")

    # --- Test on old 5170 ---
    if not args.no_test and not df_test_raw.empty and old_labels:
        df_test_raw["label"] = df_test_raw["task_id"].map(old_labels)
        df_test = df_test_raw.dropna(subset=["label"])
        if len(df_test) > 0:
            X_test, y_test = fb.transform(df_test)
            evaluate(model, X_test, y_test, fb.le,
                     f"旧 5170 Test Set (独立, n={len(y_test)})")

    conn.close()
    print(f"\nDone. Val Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
