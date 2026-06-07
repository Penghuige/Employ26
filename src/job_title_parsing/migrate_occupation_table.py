"""
迁移脚本：从 chinese_occupational_dictionary_joined_preprocessed 创建优化后的表

源表从 Parquet 导出（绕开 DB 锁定），输出到新 DuckDB 文件。
完成后由用户手动替换原表名或切换数据库文件。

用法：
    python src/job_title_parsing/migrate_occupation_table.py

输出：output/recruit_optimized.duckdb（仅含优化表 v2）
"""
import duckdb
import re
import os

PARQUET_PATH = "output/_migrate_temp.parquet"
DST_DB = "output/recruit_optimized.duckdb"
DST_TABLE = "main.chinese_occupational_dictionary_joined_preprocessed_v2"

# 源表列名映射（确保从 Parquet 读取时列名正确）
SRC_COLS = [
    "code", "title", "desc", "tasks",
    "级别", "分类代码", "职业代码",
    "大类", "中类", "小类", "细类",
    "task_list", "task_text_joined",
    "title_clean", "desc_clean",
    "hierarchy_text", "aliases",
    "retrieval_title_text", "retrieval_desc_text", "retrieval_task_text",
]


def log(msg: str):
    print(f"[migrate] {msg}")


def run_migration():
    if os.path.exists(DST_DB):
        os.remove(DST_DB)

    con = duckdb.connect(DST_DB)
    con.execute("CREATE SCHEMA IF NOT EXISTS main")

    # ============================================================
    # STEP 1: 从 Parquet 加载并创建优化表
    # ============================================================
    log("STEP 1: 从 Parquet 创建优化表...")

    con.execute(f"""
        CREATE TABLE {DST_TABLE} AS
        SELECT
            code,
            title,
            "desc",
            tasks,
            "级别",
            "分类代码",
            "职业代码",
            "大类",
            "中类",
            "小类",
            "细类",
            task_list,
            title_clean,
            desc_clean,

            -- 新增 occupation_label
            CASE
                WHEN title_clean LIKE '%L/S' THEN 'L/S'
                WHEN title_clean LIKE '%S' THEN 'S'
                WHEN title_clean LIKE '%L' THEN 'L'
                ELSE ''
            END AS occupation_label,

            -- 新增 title_core
            TRIM(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(title_clean, '[SL](/S)?$', ''),
                    '\\([^)]*\\)', ''
                )
            ) AS title_core,

            -- 新增 hier_l1~l4（清理内部空格，'g' 全局替换）
            NULLIF(TRIM(REGEXP_REPLACE("大类", '\\s+', '', 'g')), '') AS hier_l1,
            NULLIF(TRIM(REGEXP_REPLACE("中类", '\\s+', '', 'g')), '') AS hier_l2,
            NULLIF(TRIM(REGEXP_REPLACE("小类", '\\s+', '', 'g')), '') AS hier_l3,
            NULLIF(TRIM(REGEXP_REPLACE("细类", '\\s+', '', 'g')), '') AS hier_l4,

            -- hierarchy_text 重建（'g' 全局替换所有内部空格）
            TRIM(REGEXP_REPLACE(
                TRIM(
                    COALESCE(NULLIF(TRIM(REGEXP_REPLACE("大类", '\\s+', '', 'g')), ''), '') || ' ' ||
                    COALESCE(NULLIF(TRIM(REGEXP_REPLACE("中类", '\\s+', '', 'g')), ''), '') || ' ' ||
                    COALESCE(NULLIF(TRIM(REGEXP_REPLACE("小类", '\\s+', '', 'g')), ''), '') || ' ' ||
                    COALESCE(NULLIF(TRIM(REGEXP_REPLACE("细类", '\\s+', '', 'g')), ''), '')
                ),
                '\\s+', ' ', 'g'
            )) AS hierarchy_text,

            -- aliases 先原样复制，后续 Python 重建
            aliases,

            -- retrieval_title_text 占位
            TRIM(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(title_clean, '[SL](/S)?$', ''),
                    '\\([^)]*\\)', ''
                )
            ) AS retrieval_title_text,

            -- retrieval_task_text 带分段标记
            TRIM(REGEXP_REPLACE(
                TRIM(
                    CASE WHEN task_list IS NOT NULL AND task_list != [] THEN
                        '[TASKS] ' || ARRAY_TO_STRING(task_list, '; ') || ' '
                    ELSE '' END ||
                    '[DESC] ' || COALESCE(NULLIF(desc_clean, ''), title_clean) || ' ' ||
                    '[HIER] ' || TRIM(REGEXP_REPLACE(
                        TRIM(
                            COALESCE(NULLIF(TRIM(REGEXP_REPLACE("大类", '\\s+', '', 'g')), ''), '') || ' ' ||
                            COALESCE(NULLIF(TRIM(REGEXP_REPLACE("中类", '\\s+', '', 'g')), ''), '') || ' ' ||
                            COALESCE(NULLIF(TRIM(REGEXP_REPLACE("小类", '\\s+', '', 'g')), ''), '') || ' ' ||
                            COALESCE(NULLIF(TRIM(REGEXP_REPLACE("细类", '\\s+', '', 'g')), ''), '')
                        ),
                        '\\s+', ' ', 'g'
                    ))
                ),
                '\\s+', ' ', 'g'
            )) AS retrieval_task_text

        FROM read_parquet('{PARQUET_PATH}')
    """)

    count = con.execute(f"SELECT COUNT(*) FROM {DST_TABLE}").fetchone()[0]
    log(f"  表创建完成: {count} 行")

    # ============================================================
    # STEP 2: 清洗 task_list 编号前缀
    # ============================================================
    log("STEP 2: 清洗 task_list 编号前缀...")
    rows = con.execute(f"""
        SELECT code, task_list FROM {DST_TABLE}
        WHERE task_list IS NOT NULL AND task_list != []
    """).fetchall()
    prefix_pattern = re.compile(r'^\d+[\.\、\s]+')
    cleaned = 0
    for code, task_arr in rows:
        new_arr = [prefix_pattern.sub('', item).strip() for item in task_arr]
        if new_arr != task_arr:
            cleaned += 1
            escaped = [s.replace("'", "''") for s in new_arr]
            list_lit = "[" + ", ".join(f"'{s}'" for s in escaped) + "]"
            con.execute(f"UPDATE {DST_TABLE} SET task_list = {list_lit} WHERE code = '{code}'")
    log(f"  清洗了 {cleaned} 行")

    # ============================================================
    # STEP 3: 重建 aliases（从括号解析 + 去机械截断）
    # ============================================================
    log("STEP 3: 重建 aliases...")

    # 3a. 括号解析
    bracket_rows = con.execute(f"""
        SELECT code, title_clean, aliases FROM {DST_TABLE}
        WHERE title_clean LIKE '%(%)%'
    """).fetchall()
    bracket_re = re.compile(r'\(([^)]+)\)')
    rebuilt = 0
    total_added = 0
    for code, title, existing_aliases in bracket_rows:
        matches = bracket_re.findall(title)
        new_aliases = set()
        for m in matches:
            parts = re.split(r'[、，,）)]', m)
            for p in parts:
                p = p.strip()
                if len(p) >= 2 and not re.match(r'^[\d\s\-\+\.]+$', p):
                    new_aliases.add(p)
        if not new_aliases:
            continue

        if existing_aliases and len(existing_aliases) > 0:
            for a in existing_aliases:
                a = a.strip()
                if a not in new_aliases and len(a) >= 2:
                    new_aliases.add(a)

        rebuilt += 1
        total_added += len(new_aliases)
        escaped = [s.replace("'", "''") for s in sorted(new_aliases)]
        list_lit = "[" + ", ".join(f"'{s}'" for s in escaped) + "]"
        con.execute(f"UPDATE {DST_TABLE} SET aliases = {list_lit} WHERE code = '{code}'")

    log(f"  括号解析: 更新 {rebuilt} 行, 共 {total_added} 别名条目")

    # 3b. 清理无括号行的机械截断别名
    bad_rows = con.execute(f"""
        SELECT code, title_clean, title_core, aliases FROM {DST_TABLE}
        WHERE aliases IS NOT NULL AND aliases != []
          AND title_clean NOT LIKE '%(%)%'
    """).fetchall()
    cleared = 0
    for code, title, title_core, aliases in bad_rows:
        good = []
        for a in aliases:
            a = a.strip()
            if a == title or a == title_core or len(a) < 2:
                continue
            # 机械截断：去掉末尾后缀字符（员人员师等）后相等
            truncated = re.sub(r'[人员师士工者生手匠家]$', '', title)
            if a == truncated or a == truncated[:-1]:
                continue
            good.append(a)
        if good != aliases:
            cleared += 1
            if good:
                escaped = [s.replace("'", "''") for s in good]
                list_lit = "[" + ", ".join(f"'{s}'" for s in escaped) + "]"
                con.execute(f"UPDATE {DST_TABLE} SET aliases = {list_lit} WHERE code = '{code}'")
            else:
                con.execute(f"UPDATE {DST_TABLE} SET aliases = [] WHERE code = '{code}'")
    log(f"  清理机械截断: {cleared} 行")

    # 覆盖率
    alias_stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN aliases IS NOT NULL AND aliases != [] THEN 1 END) as with_aliases
        FROM {DST_TABLE}
    """).fetchone()
    log(f"  别名覆盖率: {alias_stats[1]}/{alias_stats[0]} ({round(alias_stats[1]/alias_stats[0]*100, 1)}%)")

    # ============================================================
    # STEP 4: 重新生成 retrieval_title_text
    # ============================================================
    log("STEP 4: 重新生成 retrieval_title_text...")
    con.execute(f"""
        UPDATE {DST_TABLE}
        SET retrieval_title_text = TRIM(
            title_core ||
            CASE WHEN aliases IS NOT NULL AND aliases != [] THEN
                ' ' || ARRAY_TO_STRING(aliases, ' ')
            ELSE '' END
        )
    """)
    rtt = con.execute(f"""
        SELECT ROUND(AVG(LENGTH(retrieval_title_text)), 1)
        FROM {DST_TABLE}
    """).fetchone()[0]
    log(f"  retrieval_title_text 平均长度: {rtt}")

    # ============================================================
    # STEP 5: 验证输出
    # ============================================================
    log("STEP 5: 验证...")
    final_cols = con.execute(f"DESCRIBE {DST_TABLE}").fetchall()
    log(f"最终列数: {len(final_cols)} (原表20列 → 去2冗余 + 增4层级 + 增2标记 = 24列)")
    log("列清单:")
    for i, row in enumerate(final_cols):
        name, dtype = row[0], row[1]
        null_count = con.execute(f"SELECT COUNT(*) FROM {DST_TABLE} WHERE \"{name}\" IS NULL").fetchone()[0]
        empty_count = 0
        if "VARCHAR" in dtype.upper() and "[]" not in dtype.upper():
            empty_count = con.execute(f"SELECT COUNT(*) FROM {DST_TABLE} WHERE \"{name}\" = ''").fetchone()[0]
        notes = ""
        if name == "task_list":
            notes = " ← 已清洗编号前缀"
        elif name == "aliases":
            notes = " ← 已重建（括号解析）"
        elif name == "hierarchy_text":
            notes = " ← 已重建（去内部空格）"
        elif name in ("title_core", "occupation_label", "hier_l1", "hier_l2", "hier_l3", "hier_l4"):
            notes = " ← 新增"
        elif name in ("retrieval_title_text", "retrieval_task_text"):
            notes = " ← 已重建（带分段标记）"
        log(f"  {i:2d}: {name:32s} {dtype:12s} null={null_count:4d} empty={empty_count:4d}{notes}")

    # 抽样
    log("\n抽样（3行关键列）:")
    samples = con.execute(f"""
        SELECT code, title_core, occupation_label,
               hier_l1, hier_l2, hier_l3, hier_l4,
               aliases,
               LEFT(retrieval_title_text, 60) as rtt,
               LEFT(retrieval_task_text, 120) as rta
        FROM {DST_TABLE}
        WHERE occupation_label != '' OR aliases != []
        LIMIT 3
    """).fetchall()
    for row in samples:
        log(f"  [{row[0]}] {row[1]}")
        log(f"    label={row[2]!r}, hier={row[3]}|{row[4]}|{row[5]}|{row[6]}")
        log(f"    aliases={row[7]}")
        log(f"    retrieval_title={row[8]}")
        log(f"    retrieval_task={row[9]}...")

    # hierarchy 深度分布
    h_depth = con.execute(f"""
        SELECT
            LENGTH(hierarchy_text) - LENGTH(REPLACE(hierarchy_text, ' ', '')) + 1 as depth,
            COUNT(*)
        FROM {DST_TABLE}
        GROUP BY depth ORDER BY depth
    """).fetchall()
    log("\nhierarchy 深度分布（修复后）:")
    for depth, cnt in h_depth:
        log(f"  depth={depth}: {cnt}")

    con.close()

    log(f"\n优化表已写入: {DST_DB}")
    log("源表连接问题解决后，可执行:")
    log(f"  DROP TABLE recruit.main.chinese_occupational_dictionary_joined_preprocessed;")
    log(f"  IMPORT DATABASE '{DST_DB}';  -- 或其他合并方式")


if __name__ == "__main__":
    run_migration()
