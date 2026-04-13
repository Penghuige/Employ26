import pandas as pd
import os
import re
import yaml
import duckdb

# ================= 配置区域 =================
# 从 config/database.yaml 读取 DuckDB 路径与 jobs_table
_CONFIG_FILE = r"config/database.yaml"
with open(_CONFIG_FILE, encoding='utf-8') as _f:
    _cfg = yaml.safe_load(_f)

DUCKDB_PATH = _cfg['database']['duckdb_path']
JOBS_TABLES = [t.strip() for t in _cfg['job_title_parsing']['jobs_table'].split(',')]
OUTPUT_FILE = r"src\bge\data5\Step1_Data_Deduplicated_Strict.csv"


def clean_text_for_dedup(text):
    """
    针对岗位描述的严格清洗函数，仅用于生成去重比对基准。
    剥离HTML标签及所有空白字符，消除格式差异带来的误判。
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)  # 剔除HTML标签
    text = re.sub(r'\s+', '', text)  # 剔除所有空格、换行符和制表符
    return text


def main_step1_strict():
    print(">>> 正在从 DuckDB 加载 jobs_table 数据...")
    if not os.path.exists(DUCKDB_PATH):
        print(f"找不到 DuckDB 文件: {DUCKDB_PATH}")
        return

    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    parts = []
    for table in JOBS_TABLES:
        print(f"  读取表: {table}")
        parts.append(con.execute(f'SELECT * FROM "{table}"').df())
    con.close()
    df = pd.concat(parts, ignore_index=True)

    original_len = len(df)
    print(f"成功读取数据，原始总量: {original_len} 条")

    # ================= 缺失值处理与时间格式化 =================
    print("\n>>> 正在处理缺失值与时间字段...")

    # 剔除核心标识缺失的样本
    df = df.dropna(subset=['公司名称', '岗位名称', '发布时间'])

    # 填充其他用于去重的维度，防止因空值导致比对失败
    df['工作城市'] = df['工作城市'].fillna('未知')
    df['公司行业'] = df['公司行业'].fillna('未知')
    df['岗位描述'] = df['岗位描述'].fillna('')

    # 解析发布时间并生成年月标识
    df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')
    df = df.dropna(subset=['发布时间'])
    df['YearMonth'] = df['发布时间'].dt.to_period('M')

    # ================= 构建严格去重基准 =================
    print("\n>>> 正在生成岗位描述的去重比对指纹...")
    df['dedup_desc_fingerprint'] = df['岗位描述'].apply(clean_text_for_dedup)

    # ================= 六维联合去重 =================
    print("\n>>> 正在执行六维联合去重...")
    # 按时间升序排列，确保保留该月内最早发布的记录
    df = df.sort_values(by='发布时间', ascending=True)

    dedup_subset = ['岗位名称', '工作城市', 'dedup_desc_fingerprint', '公司名称', '公司行业', 'YearMonth']
    df_dedup = df.drop_duplicates(subset=dedup_subset, keep='first')

    dedup_len = len(df_dedup)
    print("去重处理完成。")
    print(f" - 原始有效样本: {len(df)} 条")
    print(f" - 严格去重后样本: {dedup_len} 条")
    print(f" - 剔除冗余样本: {len(df) - dedup_len} 条 (占比 {(len(df) - dedup_len) / len(df) * 100:.2f}%)")

    # ================= 统计验证与输出 =================
    print("\n>>> 去重后各月份样本分布统计:")
    monthly_stats = df_dedup['YearMonth'].value_counts().sort_index()

    print("-" * 30)
    print(f"{'月份 (Year-Month)':<20} | {'数量 (Count)'}")
    print("-" * 30)
    for month, count in monthly_stats.items():
        print(f"{str(month):<20} | {count}")
    print("-" * 30)

    # 剔除辅助比对列，恢复数据原貌，并将YearMonth移至前列
    df_dedup = df_dedup.drop(columns=['dedup_desc_fingerprint'])
    cols = df_dedup.columns.tolist()
    cols.insert(1, cols.pop(cols.index('YearMonth')))
    df_dedup = df_dedup[cols]

    df_dedup.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n步骤1完成，清洗后的数据已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main_step1_strict()
