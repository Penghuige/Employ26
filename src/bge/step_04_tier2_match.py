import pandas as pd
import re
import os
import time
import json
import random
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ================= 配置区域 =================
# 1. 输入输出路径配置
INPUT_PENDING_FILE = r"src\bge\data5\Tier2_Pending_Data.csv"
DICT_FILE = r"data\中国职业大典.xlsx"

# 【新增】中间结果缓存路径（保存所有计算完的分数，用于快速修改阈值）
INTERMEDIATE_CACHE_FILE = r"src\bge\data5\Tier2_Intermediate_Cache.csv"

OUTPUT_MATCHED_FILE = r"src\bge\data5\Tier2_Matched_Data.csv"
OUTPUT_PENDING_FILE = r"src\bge\data5\Tier3_Pending_Data.csv"
OUTPUT_INSPECTION_FILE = r"src\bge\data5\Tier2_Threshold_Inspection.xlsx"
OUTPUT_SUBCLASS_COVERAGE_FILE = r"src\bge\data5\Tier2_Subclass_Coverage_10each.csv"
OUTPUT_LABEL_STUDIO_JSON = r"src\bge\data5\Tier2_Subclass_Coverage_10each.label_studio.json"

# 2. 模型与推理参数
FINETUNED_MODEL_PATH = r"D:\model\bge-base-zh-finetuned"
MATCH_THRESHOLD = 0.7  # 🏆 您可在此处随意修改阈值（如0.75, 0.78, 0.80等）
FORCE_RECALCULATE = True  # 如果更新了原始数据或模型，将其改为 True 即可重新计算向量
INSPECTION_SAMPLE_SIZE = 30  # 每个分数段抽取的样本量
TOP_K_CANDIDATES = 5
SUBCLASS_SAMPLE_SIZE = 100
RANDOM_SEED = 42
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 512

# 3. 严格复用微调阶段的清洗规则以保证特征对齐
NOISE_REGEX = r'(月薪|底薪|综合薪资|提成|绩效|奖金|包吃包住|包食宿|五险一金|带薪年假|周末双休|长白班|提供早午餐|月入过万|上不封顶|年底双薪|\d{3,5}-\d{3,5}元|\d+[kKwW](-\d+[kKwW])?|[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+|【.*?】|\[.*?\])'


def _required_cache_columns():
    """定义当前缓存版本需要包含的列。"""
    required = {
        'tier2_matched_title',
        'tier2_matched_code',
        'tier2_match_score',
        'tier2_top2_title',
        'tier2_top2_code',
        'tier2_top2_score',
        'tier2_score_margin',
    }
    for rank in range(1, TOP_K_CANDIDATES + 1):
        required.update(
            {
                f'tier2_top{rank}_title',
                f'tier2_top{rank}_code',
                f'tier2_top{rank}_score',
                f'tier2_top{rank}_desc',
                f'tier2_top{rank}_tasks',
            }
        )
    return required


def _extract_subclass_code(code):
    """从职业代码中提取小类，例如 05-03-04-01 -> 05-03-04。"""
    code = str(code).strip()
    if not code:
        return ""
    parts = [part for part in code.split('-') if part]
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return code


def _protect_excel_text(value):
    """避免 Excel 打开 CSV 时把 20-01-02 这类代码自动识别成日期。"""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return f"'{text}"


def _safe_text(value):
    """安全转字符串，兼容 NaN。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _unprotect_excel_text(value):
    """把导出到 CSV 时附加的 Excel 文本保护前缀去掉。"""
    text = _safe_text(value)
    if text.startswith("'"):
        return text[1:]
    return text


def _load_csv_with_fallback(file_path):
    """兼容多种编码读取 CSV。"""
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法解码文件: {file_path}")


def _pick_clean_requirements(row):
    """优先使用任职要求，其次岗位职责，最后退回清洗后的岗位描述。"""
    for col in ("任职要求_items_text", "岗位职责_items_text", "岗位描述_清洗", "岗位描述"):
        text = _safe_text(row.get(col, ""))
        if text:
            return text
    return ""


def _shuffle_top5_candidates(row, seed_value):
    """将 top1~top5 候选打乱后映射为 candidate_a~e。"""
    candidates = []
    for rank in range(1, TOP_K_CANDIDATES + 1):
        title = _safe_text(row.get(f"tier2_top{rank}_title", ""))
        code = _unprotect_excel_text(row.get(f"tier2_top{rank}_code", ""))
        desc = _safe_text(row.get(f"tier2_top{rank}_desc", ""))
        if not any([title, code, desc]):
            continue
        candidates.append(
            {
                "orig_rank": rank,
                "title": title,
                "code": code,
                "desc": desc,
            }
        )

    rng = random.Random(seed_value)
    rng.shuffle(candidates)

    slots = ["a", "b", "c", "d", "e"]
    payload = {}
    for slot, candidate in zip(slots, candidates):
        payload[f"candidate_{slot}_title"] = candidate["title"]
        payload[f"candidate_{slot}_desc"] = candidate["desc"]
        payload[f"candidate_{slot}_code"] = candidate["code"]
        payload[f"candidate_{slot}_orig_rank"] = candidate["orig_rank"]

    for slot in slots[len(candidates):]:
        payload[f"candidate_{slot}_title"] = ""
        payload[f"candidate_{slot}_desc"] = ""
        payload[f"candidate_{slot}_code"] = ""
        payload[f"candidate_{slot}_orig_rank"] = ""

    return payload


def export_subclass_coverage_to_label_studio_json(
    input_csv=OUTPUT_SUBCLASS_COVERAGE_FILE,
    output_json=OUTPUT_LABEL_STUDIO_JSON,
):
    """将小类覆盖 CSV 直接转换为可导入 Label Studio 的 JSON。"""
    if not os.path.exists(input_csv):
        print(f"[WARN] 找不到覆盖文件，无法导出 Label Studio JSON: {input_csv}")
        return

    df = _load_csv_with_fallback(input_csv)
    tasks = []

    for idx, row in df.iterrows():
        row_id = _safe_text(row.get("row_id", "")) or _safe_text(row.get("sample_id", "")) or str(idx)
        source_seed = f"{row_id}|{_safe_text(row.get('岗位名称', ''))}|{_safe_text(row.get('tier2_subclass_code', ''))}"
        task_data = {
            "row_id": row_id,
            "job_title": _safe_text(row.get("岗位名称", "")),
            "job_requirements_clean": _pick_clean_requirements(row),
            "tier2_subclass_code": _unprotect_excel_text(row.get("tier2_subclass_code", "")),
            "tier2_subclass_source_rank": _safe_text(row.get("tier2_subclass_source_rank", "")),
            "tier2_subclass_source_score": _safe_text(row.get("tier2_subclass_source_score", "")),
            "company_name": _safe_text(row.get("公司名称", "")),
            "work_city": _safe_text(row.get("工作城市", "")),
            "job_desc_clean": _safe_text(row.get("岗位描述_清洗", "")),
            "matched_title": _safe_text(row.get("tier2_matched_title", "")),
            "matched_code": _unprotect_excel_text(row.get("tier2_matched_code", "")),
        }
        task_data.update(_shuffle_top5_candidates(row, seed_value=source_seed))
        tasks.append({"data": task_data})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"✅ Label Studio 导入 JSON 已生成: {output_json}")


def _build_subclass_coverage_file(df):
    """按 top1 -> top2 -> top3 递补覆盖每个小类，生成人工检查文件。"""
    if df.empty:
        print("[WARN] 无法生成小类覆盖文件：输入为空。")
        return

    candidate_ranks = [1, 2, 3]
    coverage_base = df.copy().reset_index(drop=False).rename(columns={'index': '__source_row_id'})
    expanded_parts = []

    for rank in candidate_ranks:
        code_col = f'tier2_top{rank}_code'
        score_col = f'tier2_top{rank}_score'
        if code_col not in coverage_base.columns:
            continue

        rank_df = coverage_base.copy()
        rank_df['tier2_subclass_code'] = rank_df[code_col].apply(_extract_subclass_code)
        rank_df = rank_df[rank_df['tier2_subclass_code'].astype(str).str.len() > 0].copy()
        if rank_df.empty:
            continue

        rank_df['tier2_subclass_source_rank'] = rank
        rank_df['tier2_subclass_source_score'] = pd.to_numeric(
            rank_df.get(score_col, ""),
            errors='coerce',
        ).fillna(-1.0)
        expanded_parts.append(rank_df)

    if not expanded_parts:
        print("[WARN] 无法生成小类覆盖文件：缺少 top1/top2/top3 候选代码。")
        return

    coverage_df = pd.concat(expanded_parts, ignore_index=True)
    coverage_df = coverage_df.drop_duplicates(
        subset=['__source_row_id', 'tier2_subclass_code'],
        keep='first',
    )
    coverage_df = coverage_df.drop(columns=['search_query', 'score_bin'], errors='ignore')

    sampled_parts = []
    for subclass_code, group in coverage_df.groupby('tier2_subclass_code', sort=True):
        remaining = SUBCLASS_SAMPLE_SIZE
        selected_parts = []
        used_row_ids = set()

        for rank in candidate_ranks:
            if remaining <= 0:
                break

            rank_group = group[group['tier2_subclass_source_rank'] == rank].copy()
            if rank_group.empty:
                continue

            rank_group = rank_group[~rank_group['__source_row_id'].isin(used_row_ids)]
            if rank_group.empty:
                continue

            rank_group = rank_group.sort_values(
                by=['tier2_subclass_source_score', 'tier2_match_score'],
                ascending=[False, False],
            )
            chosen = rank_group.head(remaining).copy()
            if chosen.empty:
                continue

            selected_parts.append(chosen)
            used_row_ids.update(chosen['__source_row_id'].tolist())
            remaining -= len(chosen)

        if selected_parts:
            sampled_parts.append(pd.concat(selected_parts, ignore_index=True))

    if not sampled_parts:
        print("[WARN] 没有可用的小类样本，跳过覆盖文件生成。")
        return

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sort_values(
        by=['tier2_subclass_code', 'tier2_subclass_source_rank', 'tier2_subclass_source_score'],
        ascending=[True, True, False],
    )
    code_like_columns = ['tier2_subclass_code', 'tier2_matched_code']
    for rank in range(1, TOP_K_CANDIDATES + 1):
        code_like_columns.append(f'tier2_top{rank}_code')
    for col in code_like_columns:
        if col in sampled_df.columns:
            sampled_df[col] = sampled_df[col].apply(_protect_excel_text)
    sampled_df = sampled_df.drop(columns=['__source_row_id'], errors='ignore')
    sampled_df.to_csv(OUTPUT_SUBCLASS_COVERAGE_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ 小类覆盖检查文件已生成: {OUTPUT_SUBCLASS_COVERAGE_FILE}")


def clean_and_concat(row):
    """清洗职位描述并与岗位名称进行拼接，构建最终的检索 Query"""
    job_title = str(row.get('岗位名称', '')).strip()
    desc = str(row.get('岗位描述', ''))

    if pd.isna(desc) or not desc:
        desc = ""
    else:
        desc = re.sub(r'<[^>]+>', '', desc)
        try:
            desc = re.sub(NOISE_REGEX, ' ', desc, flags=re.IGNORECASE)
        except Exception:
            desc = desc.replace('【', ' ').replace('】', ' ')
        desc = re.sub(r'[^\w\u4e00-\u9fa5，。、；：！]', ' ', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()

    combined_text = f"{job_title} {desc}".strip()
    return combined_text[:MAX_SEQ_LENGTH]


def load_official_dictionary():
    print(f"-> 正在加载大典官方标准库...")
    try:
        df_dict = pd.read_excel(DICT_FILE, engine='openpyxl')
    except Exception as e:
        print(f"读取大典文件失败: {e}")
        return []

    df_dict.fillna('', inplace=True)
    title_col = next((col for col in ['title', '职业名称'] if col in df_dict.columns), 'title')
    code_col = next((col for col in ['code', '职业代码'] if col in df_dict.columns), 'code')
    desc_col = next((col for col in ['desc', '职业定义'] if col in df_dict.columns), 'desc')
    task_col = next((col for col in ['tasks', '主要工作任务'] if col in df_dict.columns), 'tasks')

    official_records = []
    for _, row in df_dict.iterrows():
        code = str(row[code_col]).strip()
        title = str(row[title_col]).strip()
        desc = str(row[desc_col]).strip()
        tasks = str(row[task_col]).strip()

        if code and title:
            search_text = f"{title}。定义：{desc} 任务：{tasks}"
            official_records.append(
                {
                    'code': code,
                    'title': title,
                    'desc': desc,
                    'tasks': tasks,
                    'search_text': search_text,
                }
            )
    return official_records


def generate_or_load_scores():
    """核心控制逻辑：存在缓存则直接加载，不存在则执行模型推理"""
    if os.path.exists(INTERMEDIATE_CACHE_FILE) and not FORCE_RECALCULATE:
        print(f"\n✅ 发现中间结果缓存，正在检查列完整性...\n缓存路径: {INTERMEDIATE_CACHE_FILE}")
        cached_df = pd.read_csv(INTERMEDIATE_CACHE_FILE, encoding='utf-8-sig')
        missing_cols = sorted(_required_cache_columns() - set(cached_df.columns))
        if not missing_cols:
            print("-> 缓存列完整，直接复用。")
            return cached_df
        print(f"-> 旧缓存缺少新列，将自动重算。缺失列数: {len(missing_cols)}")

    print("\n>>> 未发现缓存或强制重算，开始执行完整推理流程...")
    if not os.path.exists(INPUT_PENDING_FILE):
        print(f"❌ 找不到输入文件: {INPUT_PENDING_FILE}")
        return None

    df = pd.read_csv(INPUT_PENDING_FILE, encoding='utf-8-sig')
    tqdm.pandas(desc="清洗与特征拼接")
    df['search_query'] = df.progress_apply(clean_and_concat, axis=1)

    unique_queries = df['search_query'].unique().tolist()
    official_records = load_official_dictionary()
    official_texts = [rec['search_text'] for rec in official_records]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> 加载微调模型 [{FINETUNED_MODEL_PATH}] 至 {device.upper()}...")
    model = SentenceTransformer(FINETUNED_MODEL_PATH, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("-> 开始执行高维向量化与相似度计算...")
    start_time = time.time()
    with torch.no_grad():
        dict_embeddings = model.encode(official_texts, batch_size=BATCH_SIZE, convert_to_tensor=True,
                                       normalize_embeddings=True, show_progress_bar=True)
        query_embeddings = model.encode(unique_queries, batch_size=BATCH_SIZE, convert_to_tensor=True,
                                        normalize_embeddings=True, show_progress_bar=True)
        cosine_scores = torch.mm(query_embeddings, dict_embeddings.T)
        topk_size = min(TOP_K_CANDIDATES, len(official_records))
        if topk_size == 0:
            print("❌ 官方职业库为空，无法计算候选职业。")
            return None
        topk_scores, topk_indices = torch.topk(cosine_scores, k=topk_size, dim=1)

    topk_scores = topk_scores.cpu().tolist()
    topk_indices = topk_indices.cpu().tolist()

    query_result_rows = []
    for i, query in enumerate(unique_queries):
        ranked_candidates = []
        for score, idx in zip(topk_scores[i], topk_indices[i]):
            record = official_records[idx]
            ranked_candidates.append(
                {
                    "title": record["title"],
                    "code": record["code"],
                    "score": float(score),
                    "desc": record["desc"],
                    "tasks": record["tasks"],
                }
            )

        top1 = ranked_candidates[0]
        top2 = ranked_candidates[1] if len(ranked_candidates) > 1 else ranked_candidates[0]

        result_row = {
            "search_query": query,
            "tier2_matched_title": top1["title"],
            "tier2_matched_code": top1["code"],
            "tier2_match_score": top1["score"],
            "tier2_top2_title": top2["title"],
            "tier2_top2_code": top2["code"],
            "tier2_top2_score": top2["score"],
            "tier2_score_margin": top1["score"] - top2["score"],
        }

        for rank in range(1, TOP_K_CANDIDATES + 1):
            if rank <= len(ranked_candidates):
                candidate = ranked_candidates[rank - 1]
            else:
                candidate = {"title": "", "code": "", "score": "", "desc": "", "tasks": ""}
            result_row[f"tier2_top{rank}_title"] = candidate["title"]
            result_row[f"tier2_top{rank}_code"] = candidate["code"]
            result_row[f"tier2_top{rank}_score"] = candidate["score"]
            result_row[f"tier2_top{rank}_desc"] = candidate["desc"]
            result_row[f"tier2_top{rank}_tasks"] = candidate["tasks"]

        query_result_rows.append(result_row)

    print(f"向量推理计算完成，耗时 {time.time() - start_time:.2f} 秒。")

    # 映射回原表并保存为缓存
    result_df = pd.DataFrame(query_result_rows)
    df = df.merge(result_df, on='search_query', how='left')

    print(f"-> 正在生成并保存全量分数缓存至: {INTERMEDIATE_CACHE_FILE}")
    df.to_csv(INTERMEDIATE_CACHE_FILE, index=False, encoding='utf-8-sig')
    return df


def main_tier2_retrieval():
    # 1. 加载或计算全量带分数的 DataFrame
    df = generate_or_load_scores()
    if df is None:
        return

    # 2. 分层抽样检验
    print(f"\n>>> 执行分层抽样检验 (当前基准阈值: {MATCH_THRESHOLD})...")
    bins = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    labels = ['0.65-0.70', '0.70-0.75', '0.75-0.80', '0.80-0.85', '0.85-0.90', '0.90-0.95', '0.95-1.0']
    df['score_bin'] = pd.cut(df['tier2_match_score'], bins=bins, labels=labels, right=False)

    inspection_samples = []
    for label in labels:
        bin_df = df[df['score_bin'] == label]
        if not bin_df.empty:
            sample_n = min(INSPECTION_SAMPLE_SIZE, len(bin_df))
            inspection_samples.append(bin_df.sample(n=sample_n, random_state=42))

    if inspection_samples:
        df_inspection = pd.concat(inspection_samples)
        inspect_cols = ['score_bin', 'tier2_match_score', 'tier2_score_margin', '岗位名称', 'tier2_matched_title', 'tier2_top2_title',
                        '岗位描述', 'tier2_matched_code', 'tier2_top2_code']
        available_cols = [c for c in inspect_cols if c in df_inspection.columns]
        df_inspection[available_cols].to_excel(OUTPUT_INSPECTION_FILE, index=False)
        print(f"✅ 抽样检验表已更新: {OUTPUT_INSPECTION_FILE}")

    _build_subclass_coverage_file(df)
    export_subclass_coverage_to_label_studio_json()

    # 3. 执行最终截断与分流
    print(f">>> 执行最终截断与数据分流...")
    df_matched = df[df['tier2_match_score'] >= MATCH_THRESHOLD].copy()
    df_pending = df[df['tier2_match_score'] < MATCH_THRESHOLD].copy()

    # pending 文件保留匹配结果列，方便人工检查
    # 仅删除中间运算辅助列，保留 match_title/code/score 等供检查
    columns_to_drop_pending = ['search_query', 'score_bin']
    df_pending = df_pending.drop(columns=columns_to_drop_pending, errors='ignore')

    # matched 文件同样清理辅助列（保持干净输出）
    columns_to_drop_matched = ['search_query', 'score_bin']
    df_matched = df_matched.drop(columns=columns_to_drop_matched, errors='ignore')

    df_matched.to_csv(OUTPUT_MATCHED_FILE, index=False, encoding='utf-8-sig')
    df_pending.to_csv(OUTPUT_PENDING_FILE, index=False, encoding='utf-8-sig')

    total_len = len(df)
    matched_len = len(df_matched)

    print("\n" + "=" * 50)
    print(f"第二级漏斗统计 (当前阈值: {MATCH_THRESHOLD})")
    print("=" * 50)
    print(f"处理数据总量: {total_len} 条")
    print(f"成功匹配入库 (>= {MATCH_THRESHOLD}): {matched_len} 条 (拦截率: {matched_len / total_len * 100:.2f}%)")
    print(f"流入下一级漏斗 (< {MATCH_THRESHOLD}): {len(df_pending)} 条")
    print("=" * 50)


if __name__ == "__main__":
    main_tier2_retrieval()
