# =============================================================================
# 模块：src/test/parse_job_desc.py
# 功能：从 DuckDB 读取招聘样本数据，清洗岗位描述并提取结构化模块
#
# 提取的模块类别（合并同义后）：
#   job_duties    : 岗位职责 / 工作职责 / 工作内容 / 职位职责 / 主要职责
#   job_require   : 任职要求 / 岗位要求 / 职位要求 / 任职资格 / 基本要求 / 招聘要求
#   job_welfare   : 福利待遇 / 薪资待遇 / 薪酬福利 / 待遇 / 员工福利 / 职位福利
#   job_other     : 其余无法归类的段落
#
# 模块分隔检测策略（优先级从高到低）：
#   1. 【模块名】 或 [模块名] 显式括号标题（独立行或行内）
#   2. 序号+汉字标题：「一、岗位职责」「（一）任职要求」等
#   3. 模块名: 或 模块名： 作为独立纯标题行
#   4. 精确关键词标题：支持标题出现在任意位置（含句子中间）
#   5. 连续换行切段，段落首句识别为标题
#   6. 兜底：全文归入 job_duties
# =============================================================================

import json
import re
import os
import sys
from typing import Dict, List, Tuple

import duckdb
import pandas as pd
import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

CONFIG_FILE = os.path.join(_PROJECT_ROOT, "config", "database.yaml")
SAMPLE_SIZE = 200
RANDOM_SEED = 42
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "src", "test", "parsed_job_desc_sample.csv")

MODULE_KEYWORDS: Dict[str, List[str]] = {
    "job_duties": [
        "岗位职责", "工作职责", "工作内容", "职位职责", "主要职责",
        "职责描述", "工作描述", "主要工作", "职责要求",
        "工作范围", "职责范围", "工作职能", "岗位内容",
        "职位描述", "岗位描述", "工作职位",
    ],
    "job_require": [
        "任职要求", "岗位要求", "职位要求", "任职资格", "基本要求",
        "招聘要求", "应聘条件", "资质要求", "申请条件", "岗位条件",
        "候选人要求", "职位资质", "技能要求", "其他要求", "学历要求",
        "任职条件", "岗位资格", "职位资格",
    ],
    "job_welfare": [
        "福利待遇", "薪资待遇", "薪酬福利", "薪酬待遇", "员工福利",
        "职位福利", "待遇福利", "公司福利", "岗位福利", "薪资福利",
        "福利保障", "薪资范围", "薪资福利待遇", "薪酬福利待遇", "薪资标准", "薪酬标准",
    ],
}

_ALL_KEYWORDS: List[str] = sorted(
    [kw for kws in MODULE_KEYWORDS.values() for kw in kws],
    key=len, reverse=True
)

# 末尾结构化字段提取：「职能类别」「关键字/关键词」「工作地点/地址」「交通指引」等
# 每个字段单独提取为 key-value，存入 job_other JSON
_STRUCTURED_TAIL_FIELDS = [
    "职能类别", "关键字", "关键词", "工作地点", "工作地址", "上班地点",
    "交通指引", "年龄要求", "联系人", "联系方式", "简历投递",
]
# 找到第一个结构化字段出现的位置（作为正文与尾部的分界）
# 使用简单字符串搜索代替正则，避免回溯
def _find_tail_start(text: str) -> int:
    """找到文本中第一个结构化尾部字段的起始位置，返回 -1 表示未找到。

    规则：搜索「字段名：」或「字段名:」，取最早出现的位置。
    为避免误识别正文中的「联系人」「工作地址」等词，要求该字段名
    出现在文本后半段（pos > len(text) * 0.4）或行首/非正文句中。
    """
    best = -1
    text_len = len(text)
    for field in _STRUCTURED_TAIL_FIELDS:
        for sep in (field + '：', field + ':'):
            idx = text.find(sep)
            if idx == -1:
                continue
            # 要求出现在文本后40%，或前面是换行/空白
            in_tail = idx > text_len * 0.4
            prev_ok = idx == 0 or text[idx - 1] in ('\n', ' ', '\t', '\u3000')
            if not (in_tail or prev_ok):
                continue
            if best == -1 or idx < best:
                best = idx
    return best
# 单个字段分隔符：匹配「字段名：」用于 split，不用 DOTALL
_TAIL_SEP_RE = re.compile(
    r"(" + "|".join(re.escape(f) for f in _STRUCTURED_TAIL_FIELDS) + r")[：:]"
)

_BRACKET_INLINE_RE = re.compile(
    r"[【\[](.{1,20}?)[】\]][\s\u3000:：]*",
)
_BRACKET_RE = re.compile(
    r"^[\s\u3000]*[【\[](.{1,20}?)[】\]][\s\u3000:：]*$",
    re.MULTILINE,
)
# 序号+汉字标题：「一、岗位职责」「（一）任职资格」等
_ORDINAL_TITLE_RE = re.compile(
    r"(?:^|\n)[\s\u3000]*(?:[（(][一二三四五六七八九十\d][）)]|[一二三四五六七八九十][、．]|\d+[、．][^\d\s]?)[\s\u3000]*"
    r"([\u4e00-\u9fa5]{2,12})[\s\u3000]*[：:]?",
    re.MULTILINE,
)
_COLON_TITLE_ONLY_RE = re.compile(
    r"^[\s\u3000]*([\u4e00-\u9fa5]{2,12})[\s\u3000]*[：:][\s\u3000]*$",
    re.MULTILINE,
)
# 匹配任意位置出现的「已知关键词：」（带冒号）
_kw_pattern = "|".join(re.escape(kw) for kw in _ALL_KEYWORDS)
_KNOWN_TITLE_RE = re.compile(
    r"(" + _kw_pattern + r")[\s\u3000]*[：:]"
)
# 匹配「关键词」独立行（后接换行，无需冒号），用于「任职要求\n1.xxx」格式
_KNOWN_TITLE_LINE_RE = re.compile(
    r"(?:^|\n)[\s\u3000]*(" + _kw_pattern + r")[\s\u3000]*\n",
    re.MULTILINE,
)
# 匹配「序号前缀 + 已知关键词」（可无冒号），用于「（一）工作职责」「一、岗位职责」
# 允许在行中间出现（前面是换行或非汉字字符），以支持「一、xxx 二、yyy」连续格式
_ORDINAL_KW_RE = re.compile(
    r"(?:^|\n|(?<=[^\u4e00-\u9fa5]))[\s\u3000]*"
    r"(?:[（(][一二三四五六七八九十\d][）)]|[一二三四五六七八九十][、．])[\s\u3000]*"
    r"(" + _kw_pattern + r")[\s\u3000]*[：:]?",
    re.MULTILINE,
)
_ORDINAL_RE = re.compile(
    r"^[\s\u3000]*(?:[①②③④⑤⑥⑦⑧⑨⑩]|[一二三四五六七八九十][、。]|\d+[、。\.）)][^\d])"
)


def _clean_desc(text: str) -> str:
    """清洗岗位描述：去除乱码，统一换行，压缩多余空白。
    注意：末尾结构化字段（职能类别/关键字/工作地点等）不在此截断，
    由 parse_job_desc 单独提取到 job_other JSON 中。
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("?", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def _match_module(title: str) -> str:
    """将标题字符串匹配到标准模块名，无法识别返回 'job_other'。"""
    title_clean = title.strip().replace(" ", "").replace("\u3000", "")
    for module, keywords in MODULE_KEYWORDS.items():
        for kw in keywords:
            if kw in title_clean:
                return module
    return "job_other"


def _split_by_explicit_titles(text: str) -> List[Tuple[str, str]]:
    """策略1+2+2.5：按【】/[]（独立行或同行内容）、纯标题行、或关键词独立行切分。"""
    title_spans: List[Tuple[int, int, str]] = []
    for m in _BRACKET_RE.finditer(text):
        title_spans.append((m.start(), m.end(), m.group(1)))
    if not title_spans:
        inline = [(m.start(), m.end(), m.group(1))
                  for m in _BRACKET_INLINE_RE.finditer(text)
                  if _match_module(m.group(1)) != "job_other"]
        if len({_match_module(t) for _, _, t in inline}) >= 2:
            title_spans = inline
    for m in _COLON_TITLE_ONLY_RE.finditer(text):
        overlap = any(m.start() < e and m.end() > s for s, e, _ in title_spans)
        if not overlap:
            title_spans.append((m.start(), m.end(), m.group(1)))
    # 策略2.5：关键词独立行（无冒号），如「任职要求\n1.xxx」
    for m in _KNOWN_TITLE_LINE_RE.finditer(text):
        title = m.group(1)
        if _match_module(title) == "job_other":
            continue
        overlap = any(m.start() < e and m.end() > s for s, e, _ in title_spans)
        if not overlap:
            # end 指向换行符之后（即内容起始），start 指向行首
            line_start = m.start() if m.start() == 0 else m.start() + 1
            title_spans.append((line_start, m.end(), title))
    if not title_spans:
        return []
    title_spans.sort(key=lambda x: x[0])
    sections: List[Tuple[str, str]] = []
    pre = text[:title_spans[0][0]].strip()
    if pre:
        sections.append(("job_other", pre))
    for i, (start, end, title) in enumerate(title_spans):
        nxt = title_spans[i + 1][0] if i + 1 < len(title_spans) else len(text)
        sections.append((_match_module(title), text[end:nxt].strip()))
    # 若切分结果中无 job_duties，将首段 job_other（前置文本）升级为 job_duties
    has_duties = any(m == "job_duties" for m, _ in sections)
    if not has_duties and sections and sections[0][0] == "job_other":
        sections[0] = ("job_duties", sections[0][1])
    return sections


def _split_by_ordinal_titles(text: str) -> List[Tuple[str, str]]:
    """策略2b：按「一、岗位职责」「（二）任职资格」等序号+已知关键词切分。"""
    matches = list(_ORDINAL_KW_RE.finditer(text))
    if not matches:
        return []
    known: List[Tuple[int, int, str]] = []
    for m in matches:
        title = m.group(1)
        # 找到匹配串中序号「（」或汉字的真实起始位置
        # m.start() 可能指向换行符或非汉字字符（后顾断言），需跳过
        raw_start = m.start()
        matched_str = m.group(0)
        # 在匹配串中找第一个「（」「一二三...」或数字的位置
        seq_offset = re.search(r'[（(一二三四五六七八九十\d]', matched_str)
        if seq_offset:
            title_start = raw_start + seq_offset.start()
        else:
            title_start = raw_start if raw_start == 0 else raw_start + 1
        known.append((title_start, m.end(), title))
    modules_found = {_match_module(t) for _, _, t in known}
    if len(modules_found) < 2:
        return []
    sections: List[Tuple[str, str]] = []
    pre = text[:known[0][0]].strip()
    if pre:
        sections.append(("job_other", pre))
    for i, (start, end, title) in enumerate(known):
        nxt = known[i + 1][0] if i + 1 < len(known) else len(text)
        sections.append((_match_module(title), text[end:nxt].strip()))
    return sections


def _split_by_known_titles(text: str) -> List[Tuple[str, str]]:
    """策略3：精确关键词标题切分，支持标题出现在任意位置（含句子中间）。

    条件：至少找到1个已知模块标题，且标题前有实质内容（>20字）才启用。
    这样可以处理「任职要求：...」前面已经是职责内容的情况。
    """
    # 只用带冒号正则匹配
    matches = list(_KNOWN_TITLE_RE.finditer(text))
    # 无冒号+数字格式：只在文本中搜索「关键词」紧跟数字，限制搜索范围避免性能问题
    # 策略：只检查带冒号匹配未覆盖的区域，且只检查每个关键词首次出现
    colon_spans = [(m.start(), m.end()) for m in matches]
    _num_extra: List[Tuple[int, int, str]] = []
    if not matches or max(e for _, e in colon_spans) < len(text) * 0.8:
        # 只在未被冒号覆盖的区域搜索，且限制每个kw只找第一个
        for kw in _ALL_KEYWORDS:
            if _match_module(kw) == "job_other":
                continue
            idx = text.find(kw)
            if idx == -1:
                continue
            after = idx + len(kw)
            end = after
            while end < len(text) and text[end] in (' ', '\u3000') and end - after < 2:
                end += 1
            if end < len(text) and text[end].isdigit():
                overlap = any(idx < e and end > s for s, e in colon_spans)
                if not overlap:
                    _num_extra.append((idx, end, kw))
    if not matches:
        return []
    known: List[Tuple[int, int, str]] = []
    for m in matches:
        title = m.group(1)
        if _match_module(title) == "job_other":
            continue
        if known and m.start() < known[-1][1]:
            continue
        known.append((m.start(), m.end(), title))
    # 合并无冒号+数字格式的额外匹配
    for entry in _num_extra:
        s, e, t = entry
        overlap = any(s < ke and e > ks for ks, ke, _ in known)
        if not overlap:
            known.append(entry)
    known.sort(key=lambda x: x[0])
    if not known:
        return []
    modules_found = {_match_module(t) for _, _, t in known}
    # 若只有1个模块且标题前有实质内容(>20字)，把前置文本归 job_duties
    # 若只有1个模块且标题从头开始（无前置内容），说明全文都是该模块，直接返回
    pre_text = text[:known[0][0]].strip()
    if len(modules_found) < 2:
        if len(pre_text) < 20 and known[0][0] < 5:
            # 全文只有一类标题且无前置内容，直接合并返回
            sections: List[Tuple[str, str]] = []
            for i, (start, end, title) in enumerate(known):
                nxt = known[i + 1][0] if i + 1 < len(known) else len(text)
                sections.append((_match_module(title), text[end:nxt].strip()))
            return sections
        elif len(pre_text) < 20:
            return []
    sections: List[Tuple[str, str]] = []
    if pre_text:
        sections.append(("job_duties", pre_text))
    for i, (start, end, title) in enumerate(known):
        nxt = known[i + 1][0] if i + 1 < len(known) else len(text)
        sections.append((_match_module(title), text[end:nxt].strip()))
    return sections


def _split_by_inline_markers(text: str) -> List[Tuple[str, str]]:
    """策略3.5：无显式标题时，按行内关键词标记切分。

    用于修复「职责+要求连写但无换行标题」场景，例如：
    「...... 任职要求 1....」或「...... 岗位要求：...」。
    """
    require_markers = [
        "任职要求", "岗位要求", "职位要求", "任职资格", "任职条件", "岗位条件", "应聘条件",
    ]
    welfare_markers = [
        "福利待遇", "薪资待遇", "薪酬福利", "员工福利", "职位福利", "薪资范围",
    ]

    def _find_pos(markers: List[str]) -> int:
        best = -1
        for mk in markers:
            # 要求前面是边界（行首/空白/常见中文标点），降低误切风险
            pat = re.compile(r"(?:^|[\n\s\u3000。；;，,、])" + re.escape(mk) + r"(?:[\s\u3000]*[：:]?)")
            m = pat.search(text)
            if not m:
                continue
            pos = m.start() if m.start() == 0 else m.start() + 1
            if best == -1 or pos < best:
                best = pos
        return best

    req_pos = _find_pos(require_markers)
    wel_pos = _find_pos(welfare_markers)

    markers: List[Tuple[int, str]] = []
    if req_pos >= 0:
        markers.append((req_pos, "job_require"))
    if wel_pos >= 0:
        markers.append((wel_pos, "job_welfare"))
    if not markers:
        return []

    markers.sort(key=lambda x: x[0])
    first_pos = markers[0][0]

    # 太靠前（几乎从开头就是要求/福利）时，不做「职责前缀」拆分
    sections: List[Tuple[str, str]] = []
    if first_pos > 20:
        pre = text[:first_pos].strip()
        if pre:
            sections.append(("job_duties", pre))

    for i, (pos, module) in enumerate(markers):
        nxt = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        seg = text[pos:nxt].strip()
        if seg:
            sections.append((module, seg))

    # 至少要有两段，或有明确职责前缀+要求/福利，才认为有效
    if len(sections) >= 2:
        return sections
    return []


def _split_by_blank_lines(text: str) -> List[Tuple[str, str]]:
    """策略4：按双换行切段，段落首句作为标题尝试匹配。"""
    paragraphs = [p.strip() for p in re.split(r"\n{2}", text) if p.strip()]
    if len(paragraphs) <= 1:
        return []
    sections: List[Tuple[str, str]] = []
    for para in paragraphs:
        lines = para.split("\n")
        first_line = lines[0].strip()
        if len(first_line) <= 15 and not _ORDINAL_RE.match(first_line):
            module = _match_module(first_line)
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            sections.append((module, content or first_line))
        else:
            sections.append(("job_other", para))
    return sections


def _extract_tail_fields(text: str) -> Tuple[str, Dict[str, str]]:
    """从原文中提取结构化字段（职能类别/关键字/工作地点/交通指引等）。

    策略：用纯字符串搜索找到第一个字段出现的位置，把该位置之后的内容
    全部视为尾部字段区域，用 split 逐一提取 key-value（避免正则回溯）。

    返回：
        (剩余正文, {字段名: 字段值} 字典)
    如果无结构化字段，返回 (原文, {})
    """
    split_pos = _find_tail_start(text)
    if split_pos == -1:
        return text, {}
    body = text[:split_pos].strip()
    tail = text[split_pos:]
    # 用 split 把 tail 按「字段名：/字段名:」切开
    parts = _TAIL_SEP_RE.split(tail)
    fields: Dict[str, str] = {}
    # parts[0] 是第一个字段名之前的内容（通常为空），跳过
    i = 1
    while i + 1 <= len(parts) - 1:
        key = parts[i].strip()
        val = parts[i + 1].strip()
        if key and val:
            fields[key] = val
        i += 2
    return body, fields


def parse_job_desc(raw_desc: str) -> Dict[str, str]:
    """主解析函数：清洗 -> 提取末尾字段 -> 切分 -> 归类 -> 合并同类段落。

    job_other 输出为 JSON 字符串，格式：
        {"职能类别": "...", "关键字": "...", "其他内容": "...", ...}
    其中 "其他内容" key 存放无法归类的段落文本（若有）。
    """
    empty = {"job_duties": "", "job_require": "", "job_welfare": "", "job_other": ""}
    text = _clean_desc(raw_desc)
    if not text:
        return empty
    # 提取末尾结构化字段
    text, tail_fields = _extract_tail_fields(text)
    if not text and not tail_fields:
        return empty
    sections = _split_by_explicit_titles(text)
    if not sections:
        sections = _split_by_ordinal_titles(text)
    if not sections:
        sections = _split_by_known_titles(text)
    if not sections:
        sections = _split_by_inline_markers(text)
    if not sections:
        sections = _split_by_blank_lines(text)
    if not sections:
        sections = [("job_duties", text)]
    buckets: Dict[str, List[str]] = {
        "job_duties": [], "job_require": [], "job_welfare": [], "job_other": []
    }
    for module, content in sections:
        if content:
            buckets[module].append(content)
    result: Dict[str, str] = {k: "\n\n".join(v).strip() for k, v in buckets.items()}
    # 构建 job_other JSON：末尾结构化字段 + 无法归类的段落文本
    other_dict: Dict[str, str] = {}
    if tail_fields:
        other_dict.update(tail_fields)
    if result["job_other"]:
        other_dict["其他内容"] = result["job_other"]
    result["job_other"] = json.dumps(other_dict, ensure_ascii=False) if other_dict else ""
    return result


def load_sample_from_duckdb(config_path: str, sample_size: int, seed: int) -> pd.DataFrame:
    """从 DuckDB 的 jobs_table 读取样本数据。"""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_path = os.path.join(_PROJECT_ROOT, cfg["database"]["duckdb_path"])
    raw_tables = cfg["job_title_parsing"]["jobs_table"]
    tables = [t.strip() for t in raw_tables.split(",")]
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB 文件不存在: {db_path}")
    con = duckdb.connect(db_path, read_only=True)
    union_parts = [f'SELECT * FROM "{t}"' for t in tables]
    union_sql = " UNION ALL ".join(union_parts)
    sql = f"""
        SELECT *
        FROM ({union_sql})
        WHERE 岗位描述 IS NOT NULL
          AND LENGTH(TRIM(岗位描述)) > 10
        ORDER BY md5(
            COALESCE(CAST(岗位名称 AS VARCHAR), '') ||
            COALESCE(LEFT(CAST(岗位描述 AS VARCHAR), 50), '') ||
            '{seed}'
        )
        LIMIT {sample_size}
    """
    df = con.execute(sql).df()
    con.close()
    print(f"从 DuckDB 读取样本: {len(df)} 条（来自 {len(tables)} 张表）")
    return df


def main() -> None:
    print("[1/3] 从 DuckDB 读取样本数据...")
    df = load_sample_from_duckdb(CONFIG_FILE, SAMPLE_SIZE, RANDOM_SEED)
    print("[2/3] 清洗并解析岗位描述...")
    parsed_rows = [parse_job_desc(str(row.get("岗位描述", "")))
                   for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)
    keep_cols = [c for c in ["岗位名称", "工作城市", "公司名称", "岗位描述"] if c in df.columns]
    out_df = pd.concat([df[keep_cols].reset_index(drop=True), parsed_df], axis=1)
    print("\n解析统计：")
    total = len(out_df)
    for col in ["job_duties", "job_require", "job_welfare", "job_other"]:
        hit = int((out_df[col].str.strip() != "").sum())
        print(f"  {col:<15}: {hit}/{total} ({hit / total * 100:.1f}%)")
    print(f"\n[3/3] 保存结果至: {OUTPUT_FILE}")
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("完成。")


if __name__ == "__main__":
    main()
