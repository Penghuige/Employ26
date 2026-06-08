"""岗位描述结构化切分工具。"""

import re
import json
import html
import math
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

TITLE_ALIASES = {
    "岗位职责": [
        "岗位职责", "职责描述", "工作职责", "工作内容", "职位描述", "主要职责", "职责",
        "岗位描述", "岗位内容", "主要工作内容", "主要工作职责", "职位职责", "工作描述", "主要工作",
        "职责要求", "工作范围", "职责范围", "工作职能", "工作总责", "总责",
        "Responsibilities", "Responsibility", "Main Responsibilities", "Job Responsibilities",
        "Key Responsibilities", "Roles and Responsibilities", "Role and Responsibility",
        "Primary Responsibilities", "Primary and Secondary Responsibilities", "Job Description",
        "Position Objective", "Role Purpose", "What you will do", "Job Profile"
    ],
    "任职要求": [
        "任职要求", "岗位要求", "任职资格", "职位要求", "任职条件", "资格要求", "招聘要求",
        "技能要求", "工作要求", "能力要求", "基本要求", "人员要求", "任职资格要求", "职责要求",
        "要求", "应聘条件", "资质要求", "申请条件", "岗位条件", "候选人要求", "职位资质", "其他要求",
        "学历要求", "岗位资格", "职位资格", "具体要求", "专业要求", "工作技能", "技能",
        "Requirements", "Job Requirements", "Qualification", "Qualifications", "Candidate Profile"
    ],
    "福利待遇": [
        "福利待遇", "公司福利", "薪酬福利", "薪资待遇", "福利", "薪资福利", "员工福利", "待遇",
        "薪酬待遇", "员工福利", "职位福利", "待遇福利", "岗位福利", "福利保障", "薪资范围",
        "薪资福利待遇", "薪酬福利待遇", "薪资标准", "薪酬标准"
    ],
    "其他信息": [
        "工作地点", "上班地点", "工作时间", "上班时间", "联系方式", "联系地址", "工作地址", "公司地址",
        "地址", "时间", "备注", "加分项", "我们提供", "你将获得", "公司简介", "应聘方式",
        "住宿环境", "食堂", "宿舍", "附近地铁站", "附近公交站", "节日活动", "交通地址", "社保",
        "培训", "假期福利", "社会保障", "人文关怀", "职业规划", "薪资结构", "职能类别", "关键字",
        "关键词", "交通指引", "年龄要求", "联系人", "简历投递", "职位信息", "基本信息", "重要提示",
        "公司介绍", "项目背景介绍", "职业介绍", "校招岗位", "招聘岗位", "招聘岗位和专业", "公司特色"
        , "About the Company", "Who we are", "Brand Introduction", "Position Title", "Job Title",
        "友情提醒", "提醒"
    ]
}
MIDLINE_EXCLUDE = {"福利", "待遇", "地址", "时间", "路线"}

ALIAS_TO_STD = sorted(
    [(alias, std) for std, aliases in TITLE_ALIASES.items() for alias in aliases],
    key=lambda x: len(x[0]),
    reverse=True,
)
ALIAS_PATTERNS = [
    (re.compile(r'[\s\?？·•●▪◆★※~_]*'.join(map(re.escape, alias)), re.IGNORECASE), alias, std)
    for alias, std in ALIAS_TO_STD
]
MIDLINE_ALIASES = [alias for alias, _ in ALIAS_TO_STD if len(alias) >= 2 and alias not in MIDLINE_EXCLUDE]
MIDLINE_PATTERN = "|".join(re.escape(x) for x in sorted(set(MIDLINE_ALIASES), key=len, reverse=True))

ITEM_TOKEN = r'(?:\d+\s*[、．,，)）]|\d+\.(?!\d)|[（(]\d+[)）]|[一二三四五六七八九十]+\s*[、.．]|[（(][一二三四五六七八九十]+[)）]|[-•●▪◆★])'
ITEM_START_RE = re.compile(rf'(?:(?<=^)|(?<=[\n;；]))\s*(?=(?:"|“)?{ITEM_TOKEN})')
ITEM_LEAD_RE = re.compile(rf'^\s*(?:"|“)?{ITEM_TOKEN}\s*')
OPTIONAL_NOTE_RE = r'(?:[（(][^\n:：)]{{1,12}}[)）])?'
PREFIX_ENUM_RE = re.compile(r'^\s*(?:[（(]?[一二三四五六七八九十0-9]+[)）]?[、.．]?\s*)')
NOISE_ONLY_RE = re.compile(r'^\s*(?:[（(]?[一二三四五六七八九十0-9]+[)）]?[、.．。]?)\s*$')


def strip_tags(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'(?is)<\s*br\s*/?\s*>', '\n', text)
    text = re.sub(r'(?is)</\s*(div|p|li|tr|table|ul|ol|section|article)\s*>', '\n', text)
    text = re.sub(r'(?is)<[^>]+>', '', text)
    return text


def remove_noise(text: str) -> str:
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
    text = re.sub(r'(?<=[A-Za-z0-9\u4e00-\u9fff])[\?？·•●▪◆★※~_]+(?=[A-Za-z0-9\u4e00-\u9fff])', '', text)
    text = re.sub(r'[?？]{3,}', '', text)
    return text


def sanitize_item(text: str) -> str:
    text = remove_noise(text).replace('?', '').replace('？', '')
    text = re.sub(r'^[?？·•●▪◆★※\s"\']+', '', text)
    text = re.sub(r'[?？·•●▪◆★※\s"\']+$', '', text)
    text = text.strip("[] ")
    text = re.sub(r'\s+', ' ', text).strip(' ;；')
    if text in {'[', ']', '"', "'", '[]', '----', '-----'}:
        return ''
    return text


def normalize_text(text: str) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)
    text = strip_tags(text)
    text = remove_noise(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("：", ":").replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]").replace("．", ".")
    text = re.sub(r"[ \t\u3000\xa0]+", " ", text)
    inline_re = re.compile(
        rf'(?<!^)(?<!\n)(?P<prefix>[；;。.!?？!"\']|\s)\s*(?P<head>(?:\[)?(?:{MIDLINE_PATTERN})(?:\])?{OPTIONAL_NOTE_RE}\s*(?::)?)'
    )
    for _ in range(3):
        text = inline_re.sub(lambda m: f"{m.group('prefix')}\n{m.group('head')}", text)
    text = re.sub(r'(?<!^)(?<!\n)(?=[一二三四五六七八九十]+[、.])', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def standardize_title(raw_title: str) -> str:
    cleaned = sanitize_item(raw_title)
    cleaned = PREFIX_ENUM_RE.sub("", cleaned)
    cleaned = re.sub(OPTIONAL_NOTE_RE + r"$", "", cleaned).strip()
    compact = re.sub(r'[\s:：\[\]()"“”]+', '', cleaned)
    compact_lower = compact.lower()
    if compact in {"职责", "工作职责"}:
        return "岗位职责"
    if compact in {"要求", "任职要求", "任职资格"}:
        return "任职要求"
    if compact in {"作职责", "工作总责", "总责"}:
        return "岗位职责"
    if compact in {"具体要求", "专业要求", "工作技能", "技能"}:
        return "任职要求"
    if compact_lower in {
        "responsibilities", "responsibility", "mainresponsibilities", "jobresponsibilities",
        "keyresponsibilities", "rolesandresponsibilities", "roleandresponsibility",
        "primaryresponsibilities", "primaryandsecondaryresponsibilities", "jobdescription",
        "positionobjective", "rolepurpose", "whatyouwilldo", "jobprofile"
    }:
        return "岗位职责"
    if compact_lower in {
        "requirements", "jobrequirements", "qualification", "qualifications", "candidateprofile"
    }:
        return "任职要求"
    if compact_lower in {"benefits", "companyprofile", "aboutthecompany"}:
        return "其他信息" if "company" in compact_lower else "福利待遇"
    if compact_lower in {"whoweare", "brandintroduction", "positiontitle", "jobtitle"}:
        return "其他信息"
    for patt, alias, std in ALIAS_PATTERNS:
        if patt.fullmatch(cleaned) or patt.fullmatch(compact):
            return std
    if ("任职" in compact or "资格" in compact or "岗位" in compact or "职位" in compact) and ("要求" in compact or "条件" in compact):
        return "任职要求"
    if compact.endswith(("要求", "条件", "资格")) and len(compact) <= 10:
        return "任职要求"
    if ("职责" in compact or "内容" in compact or "描述" in compact or "范围" in compact or "总责" in compact) and (
        "岗位" in compact or "工作" in compact or "职位" in compact or compact == "职责" or len(compact) <= 8
    ):
        return "岗位职责"
    if "福利" in compact or "待遇" in compact or "薪酬" in compact:
        return "福利待遇"
    if any(k in compact for k in [
        "地点", "地址", "时间", "联系", "方式", "简介", "应聘", "社保", "人数", "日期", "方向", "空间", "路线",
        "信息", "提示", "介绍", "校招岗位", "招聘岗位"
    ]):
        return "其他信息"
    return cleaned


def match_heading(line: str):
    original = line.strip()
    if not original:
        return None
    s = remove_noise(original)
    s = re.sub(r'^[?？·•●▪◆★※"\']+\s*', '', s)
    s2 = PREFIX_ENUM_RE.sub("", s)
    s2 = s2.strip()
    if re.fullmatch(r'职责\s*:?', s2):
        return "职责", ""
    if re.fullmatch(r'要求\s*:?', s2):
        return "任职要求", ""
    for patt, alias, std in ALIAS_PATTERNS:
        if std in {"其他信息", "福利待遇"}:
            regex = rf'^(?:\[)?{patt.pattern}(?:\])?{OPTIONAL_NOTE_RE}(?=\s*(?::|$))\s*(?::)?\s*(?P<rest>.*)$'
        else:
            regex = rf'^(?:\[)?{patt.pattern}(?:\])?{OPTIONAL_NOTE_RE}\s*(?::)?\s*(?P<rest>.*)$'
        m = re.match(regex, s2, re.IGNORECASE)
        if m:
            return alias, m.group("rest").strip()
    m = re.match(r'^[\[\(【]?\s*([A-Za-z\u4e00-\u9fff /&+、\-]{1,32})\s*[\]\)】]?\s*:\s*(.*)$', s2)
    if m:
        label, rest = m.group(1).strip(), m.group(2).strip()
        title_std = standardize_title(label)
        if title_std != label:
            return label, rest
    return None


def split_numbered(text: str) -> List[str]:
    starts = [m.start() for m in ITEM_START_RE.finditer(text)]
    if re.match(rf'^\s*(?:"|“)?{ITEM_TOKEN}', text):
        if not starts or starts[0] != 0:
            starts = [0] + starts
    starts = sorted(set(starts))
    if len(starts) >= 2:
        bounds = starts + [len(text)]
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = text[a:b].strip()
            seg = ITEM_LEAD_RE.sub("", seg).strip()
            seg = re.sub(r"\s*\n\s*", " ", seg)
            seg = sanitize_item(seg)
            if seg:
                parts.append(seg)
        return parts
    if len(starts) == 1 and starts[0] == 0:
        seg = sanitize_item(ITEM_LEAD_RE.sub("", text).strip())
        return [seg] if seg else []
    return []


def split_cn_subheads(text: str):
    pat = re.compile(r'([一二三四五六七八九十]+)\s*[、.]\s*([^\n:：；;。]{1,20})(?::)?')
    ms = list(pat.finditer(text))
    if len(ms) < 2 or ms[0].start() != 0:
        return None
    labels = [m.group(2).strip() for m in ms]
    if sum(len(lb) <= 12 and not re.search(r'[，,；;。]', lb) for lb in labels) < 2:
        return None
    bounds = [m.start() for m in ms] + [len(text)]
    items = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = text[a:b].strip()
        m = pat.match(seg)
        if not m:
            continue
        label = sanitize_item(m.group(2))
        rest = seg[m.end():].strip()
        nums = split_numbered(rest)
        item = sanitize_item(label + (": " + "；".join(nums) if nums else (": " + rest if rest else "")))
        if item:
            items.append(item)
    return items if len(items) >= 2 else None


def split_items(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    cn = split_cn_subheads(text)
    if cn:
        return cn
    nums = split_numbered(text)
    if nums and len(nums) >= 2:
        return nums
    lines = [sanitize_item(x) for x in re.split(r"\n+", text) if sanitize_item(x)]
    if len(lines) >= 2:
        return lines
    parts = [sanitize_item(x) for x in re.split(r"[；;]", text) if sanitize_item(x)]
    if len(parts) >= 2:
        return parts
    one = sanitize_item(text)
    return [one] if one else []


def likely_duty_text(lines: List[str]) -> bool:
    t = " ".join(lines)
    duty_score = len(re.findall(r"负责|参与|开展|执行|制定|完成|跟进|处理|管理|维护|开发|设计|研究|协助|推进|撰写|拓展|走访|下达|调试|改善|培训", t))
    req_score = len(re.findall(r"学历|专业|经验|优先|熟练|能力|责任心|沟通|任职|要求|资格|抗压|本科|大专|硕士|博士|以上学历", t))
    return duty_score >= req_score


def infer_section_title(lines: List[str], fallback: Optional[str] = None) -> Optional[str]:
    text = " ".join(sanitize_item(x) for x in lines if sanitize_item(x))
    if not text:
        return fallback

    scores = {
        "岗位职责": len(re.findall(r"负责|参与|开展|执行|制定|完成|跟进|处理|管理|维护|开发|设计|研究|协助|推进|撰写|拓展|走访|下达|调试|改善|培训|巡察|接待|销售|测试|支持", text)),
        "任职要求": len(re.findall(r"学历|专业|经验|优先|熟练|能力|责任心|沟通|任职|要求|资格|抗压|本科|大专|硕士|博士|以上学历|年龄|CET|GPA|证书|职称|相关专业|工作经验|身体健康|可接受|office|英语", text, re.IGNORECASE)),
        "福利待遇": len(re.findall(r"福利|待遇|薪资|薪酬|补贴|奖金|住宿|餐补|房补|五险一金|社保|公积金|年假|休假|节日|体检|提成|补助", text)),
        "其他信息": len(re.findall(r"公司简介|公司介绍|项目背景|工作地点|上班地点|地址|联系|联系方式|简历投递|校招岗位|招聘岗位|职位信息|基本信息|重要提示|上市|成立于|股票代码|欢迎投递|工作地址|办公地点", text)),
    }

    if "公司主要从事" in text or "以下简称" in text or "欢迎投递" in text:
        scores["其他信息"] += 3
    if "岗位类别" in text or "招聘岗位" in text:
        scores["其他信息"] += 2
    if "工作技能" in text or "专业要求" in text or "具体要求" in text:
        scores["任职要求"] += 2
    if "工作范围" in text or "工作总责" in text:
        scores["岗位职责"] += 2

    best_title, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return fallback
    return best_title


def parse_job_description(text: str) -> Dict[str, Any]:
    raw = "" if text is None else str(text)
    text = normalize_text(raw)
    if not text:
        return {"岗位描述_raw": raw, "sections": [], "unclassified": []}
    lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
    sections = []
    current = None
    preamble = []
    explicit_seen = False

    def start_section(title_raw: str, inferred: bool = False):
        nonlocal current
        current = {
            "title_raw": sanitize_item(PREFIX_ENUM_RE.sub("", title_raw)),
            "title_std": standardize_title(title_raw),
            "title_inferred": inferred,
            "buffer": [],
        }
        sections.append(current)

    for line in lines:
        mh = match_heading(line)
        if mh:
            title_raw, rest = mh
            explicit_seen = True
            start_section(title_raw, inferred=False)
            if rest:
                current["buffer"].append(rest)
            continue
        if explicit_seen:
            if current is None:
                start_section("岗位职责", inferred=True)
            current["buffer"].append(line)
        else:
            preamble.append(line)

    # ── 前置文本（preamble）清洗 ──────────────────────────────────────────────
    # preamble 收集了"第一个明确标题出现之前"的所有文本行。
    # 先过滤掉纯噪声行（仅含标点/空白/零宽字符），避免干扰后续推断。
    preamble = [ln for ln in preamble if not NOISE_ONLY_RE.fullmatch(sanitize_item(ln) or "")]

    # ── unclassified 初始化 ───────────────────────────────────────────────────
    # unclassified 最终存放"无法归入任何 section 的文本片段"，供人工复查。
    # 以下分两条路径处理 preamble：
    unclassified = []

    if explicit_seen:
        # ── 路径A：文本中存在至少一个明确标题 ──────────────────────────────
        # 此时 preamble 是标题出现"之前"的内容，通常是公司介绍、岗位背景等。
        if preamble:
            inferred_title = None

            # 特殊情形：第一个明确 section 已经是"任职要求"，
            # 但 preamble 语义上更像职责描述（动作词多于要求词）
            # → 补推断一个"岗位职责" section，避免职责内容丢失。
            if sections and sections[0]["title_std"] == "任职要求" and likely_duty_text(preamble):
                inferred_title = "岗位职责"
            else:
                # 通用情形：用关键词打分推断 preamble 最可能属于哪个 section
                # （岗位职责 / 任职要求 / 福利待遇 / 其他信息）。
                # 若各类别得分均为 0，infer_section_title 返回 None。
                inferred_title = infer_section_title(preamble)

            if inferred_title:
                # 能推断出类别 → 在 sections 头部插入一个"推断 section"
                # title_inferred=True 标记此 section 无显式标题行
                sections.insert(0, {
                    "title_raw": inferred_title,
                    "title_std": inferred_title,
                    "title_inferred": True,
                    "buffer": preamble[:],
                })
            else:
                # 无法推断（得分全为 0，内容模糊）→ 归入 unclassified
                # 典型情况：纯公司宣传语、格式乱码、仅含岗位名称等无意义前言
                unclassified.extend(preamble)
    else:
        # ── 路径B：全文没有任何明确标题行 ───────────────────────────────────
        # 整段 preamble 就是全部内容，强制推断一个 section 标题。
        # fallback 保证即使关键词全无命中，也能兜底为"岗位职责"或"任职要求"。
        # 此路径下不会产生 unclassified 内容。
        inferred_title = infer_section_title(
            preamble,
            fallback="岗位职责" if likely_duty_text(preamble) else "任职要求"
        )
        sections = [{
            "title_raw": inferred_title,
            "title_std": inferred_title,
            "title_inferred": True,
            "buffer": preamble[:],
        }]

    merged = []
    for sec in sections:
        if merged and sec["title_raw"] == merged[-1]["title_raw"] and sec["title_inferred"] == merged[-1]["title_inferred"]:
            merged[-1]["buffer"].extend(sec["buffer"])
        elif merged and sec["title_std"] == merged[-1]["title_std"] and sec["title_std"] in {"岗位职责", "任职要求", "福利待遇"} and (sec["title_inferred"] or merged[-1]["title_inferred"]):
            merged[-1]["buffer"].extend(sec["buffer"])
        else:
            merged.append(sec)
    sections = merged

    final_sections = []
    for sec in sections:
        body = "\n".join(sec.pop("buffer")).strip()
        if not body:
            continue
        items = split_items(body)
        if items:
            final_sections.append({**sec, "items": items})

    for idx, sec in enumerate(final_sections[:-1]):
        if sec["title_std"] == "任职要求" and likely_duty_text(sec["items"]) and final_sections[idx + 1]["title_std"] == "任职要求":
            sec["title_raw"] = "岗位职责"
            sec["title_std"] = "岗位职责"
            sec["title_inferred"] = True

    cleaned_unclassified = []
    seen = set()
    for x in unclassified:
        sx = sanitize_item(x)
        if NOISE_ONLY_RE.fullmatch(sx or ""):
            continue
        if sx and sx not in seen:
            cleaned_unclassified.append(sx)
            seen.add(sx)

    return {"岗位描述_raw": raw, "sections": final_sections, "unclassified": cleaned_unclassified}


def _parse_job_description_worker(text: str) -> Dict[str, Any]:
    """供多进程批处理复用的顶层 worker。"""
    return parse_job_description(text)


def _join_section_items(parsed_obj: Dict[str, Any], title_std: str) -> str:
    """汇总指定 section 的 items，保留原有顺序。"""
    items: List[str] = []
    for sec in parsed_obj.get("sections", []):
        if sec.get("title_std") != title_std:
            continue
        for item in sec.get("items", []):
            clean_item = sanitize_item(item)
            if clean_item:
                items.append(clean_item)
    return " | ".join(items)


def _build_sections_brief(parsed_obj: Dict[str, Any]) -> str:
    """构建紧凑的 section 摘要，便于人工排查。"""
    brief_parts = []
    for sec in parsed_obj.get("sections", []):
        title_std = sec.get("title_std", "")
        items = [sanitize_item(x) for x in sec.get("items", []) if sanitize_item(x)]
        if not items:
            continue
        brief_parts.append(f"{title_std}:{' / '.join(items[:2])}")
    return " | ".join(brief_parts)


def _select_default_rag_query(
    requirement_text: str,
    duty_text: str,
    cleaned_desc: str,
) -> Tuple[str, str]:
    """为预处理阶段提供确定性的默认匹配文本。"""
    if requirement_text:
        return requirement_text, "任职要求"
    if duty_text:
        return duty_text, "岗位职责"
    if cleaned_desc:
        return cleaned_desc, "岗位描述_清洗"
    return "", "空文本"


def _parse_batch_texts(
    texts: List[str],
    num_workers: int,
    executor: Optional[ProcessPoolExecutor] = None,
) -> List[Dict[str, Any]]:
    """对一个批次文本执行解析，单进程和多进程共用。"""
    if not texts:
        return []

    if num_workers <= 1:
        return [parse_job_description(text) for text in texts]

    chunksize = max(1, len(texts) // max(1, num_workers * 4))
    if executor is None:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as local_executor:
            return list(local_executor.map(_parse_job_description_worker, texts, chunksize=chunksize))
    return list(executor.map(_parse_job_description_worker, texts, chunksize=chunksize))


def parse_desc_df(
    df: pd.DataFrame,
    desc_col: str = "岗位描述",
    batch_size: int = 2000,
    num_workers: int = 1,
) -> pd.DataFrame:
    """批量切分岗位描述，返回附带解析结果列的新 DataFrame。

    新增列：
    - 岗位描述_清洗
    - 岗位描述_切分JSON
    - 任职要求_items_text
    - 岗位职责_items_text
    - unclassified_text
    - sections_brief
    - RAG匹配文本
    - RAG匹配来源
    """
    if desc_col not in df.columns:
        raise KeyError(f"DataFrame 缺少描述列: {desc_col}")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if num_workers <= 0:
        raise ValueError("num_workers 必须大于 0")

    output_df = df.copy()
    desc_texts = output_df[desc_col].fillna("").astype(str).tolist()
    parsed_results: List[Dict[str, Any]] = []

    executor: Optional[ProcessPoolExecutor] = None
    active_workers = num_workers
    main_file = getattr(sys.modules.get("__main__"), "__file__", "")
    if main_file.endswith("<stdin>"):
        active_workers = 1
    try:
        if active_workers > 1:
            ctx = mp.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=active_workers, mp_context=ctx)

        for start in range(0, len(desc_texts), batch_size):
            batch_texts = desc_texts[start: start + batch_size]
            try:
                parsed_results.extend(
                    _parse_batch_texts(batch_texts, num_workers=active_workers, executor=executor)
                )
            except (BrokenProcessPool, OSError, RuntimeError):
                if active_workers <= 1:
                    raise
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None
                active_workers = 1
                parsed_results.extend(_parse_batch_texts(batch_texts, num_workers=1))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    cleaned_descs: List[str] = []
    parsed_jsons: List[str] = []
    requirement_texts: List[str] = []
    duty_texts: List[str] = []
    unclassified_texts: List[str] = []
    sections_briefs: List[str] = []
    rag_query_texts: List[str] = []
    rag_query_sources: List[str] = []

    for raw_text, parsed_obj in zip(desc_texts, parsed_results):
        cleaned_desc = normalize_text(raw_text)
        requirement_text = _join_section_items(parsed_obj, "任职要求")
        duty_text = _join_section_items(parsed_obj, "岗位职责")
        unclassified_text = " | ".join(parsed_obj.get("unclassified", []))
        sections_brief = _build_sections_brief(parsed_obj)
        rag_query_text, rag_query_source = _select_default_rag_query(
            requirement_text=requirement_text,
            duty_text=duty_text,
            cleaned_desc=cleaned_desc,
        )

        cleaned_descs.append(cleaned_desc)
        parsed_jsons.append(json.dumps(parsed_obj, ensure_ascii=False))
        requirement_texts.append(requirement_text)
        duty_texts.append(duty_text)
        unclassified_texts.append(unclassified_text)
        sections_briefs.append(sections_brief)
        rag_query_texts.append(rag_query_text)
        rag_query_sources.append(rag_query_source)

    output_df["岗位描述_清洗"] = cleaned_descs
    output_df["岗位描述_切分JSON"] = parsed_jsons
    output_df["任职要求_items_text"] = requirement_texts
    output_df["岗位职责_items_text"] = duty_texts
    output_df["unclassified_text"] = unclassified_texts
    output_df["sections_brief"] = sections_briefs
    output_df["RAG匹配文本"] = rag_query_texts
    output_df["RAG匹配来源"] = rag_query_sources
    return output_df


def build_issue_dataframe(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """输出 unclassified 不为空的问题行，便于人工检查。"""
    if "unclassified_text" not in parsed_df.columns:
        raise KeyError("请先运行 parse_desc_df，再构建问题行数据。")
    issue_mask = parsed_df["unclassified_text"].fillna("").astype(str).str.strip() != ""
    return parsed_df.loc[issue_mask].copy()


def build_hardcase_dataframe(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """识别仍可能存在切分异常的样本。"""
    if "岗位描述_切分JSON" not in parsed_df.columns:
        raise KeyError("请先运行 parse_desc_df，再构建疑难样本数据。")

    suspicious_idx: List[int] = []
    parsed_json_series = parsed_df["岗位描述_切分JSON"].fillna("").astype(str)
    parsed_objects = parsed_json_series.apply(json.loads)

    for idx, obj in enumerate(parsed_objects):
        joined = " ".join(
            [sec["title_raw"] + " " + sec["title_std"] + " " + " ".join(sec["items"]) for sec in obj.get("sections", [])]
            + obj.get("unclassified", [])
        )
        if re.search(
            r'\[[^\]]*(工作职责|岗位职责|任职要求|任职资格|岗位要求|准入要求|工作地点|办公地|上班时间|福利待遇)[^\]]*\]'
            r'|(?:工作职责|岗位职责|任职要求|任职资格|岗位要求|准入要求|工作地点|办公地|上班时间|福利待遇)\s*:',
            joined,
        ):
            suspicious_idx.append(idx)
            continue
        if any(
            (not item or re.fullmatch(r'[\W_]+', item))
            for sec in obj.get("sections", [])
            for item in sec.get("items", [])
        ):
            suspicious_idx.append(idx)

    return parsed_df.iloc[suspicious_idx].copy() if suspicious_idx else parsed_df.iloc[0:0].copy()


def main():
    # name = "智联招聘"
    name = "前程无忧"
    input_path = f"output\\samples\\{name}_样本_1%.csv"
    output_path = f"./{name}_样本_1%_岗位描述切分结果_v2.csv"
    sample_json_path = f"./{name}_岗位描述切分样例_v2.jsonl"
    issue_csv_path = f"./{name}_样本_1%_unclassified问题行.csv"
    hard_case_path = f"./{name}_样本_1%_疑难样本待复核.csv"

    df = pd.read_csv(input_path, encoding="utf-8")
    default_workers = max(1, min(8, max(1, (os.cpu_count() or 1) - 1)))
    parsed_df = parse_desc_df(
        df,
        desc_col="岗位描述",
        batch_size=2000,
        num_workers=default_workers,
    )
    parsed_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    issue_df = build_issue_dataframe(parsed_df)
    issue_df.to_csv(issue_csv_path, index=False, encoding="utf-8-sig")

    with open(sample_json_path, "w", encoding="utf-8") as f:
        for i, obj_text in enumerate(parsed_df["岗位描述_切分JSON"].head(200)):
            obj = json.loads(obj_text)
            f.write(json.dumps({"row": int(i), **obj}, ensure_ascii=False) + "\n")

    hard_df = build_hardcase_dataframe(parsed_df)
    hard_df.to_csv(hard_case_path, index=False, encoding="utf-8-sig")

    print("saved", output_path)
    print("samples", sample_json_path)
    print("hard_cases", hard_case_path)
    print("hard_case_count", len(hard_df))

    print("rows", len(parsed_df))
    print("issues", issue_csv_path)
    print("hard_cases", hard_case_path)

if __name__ == "__main__":
    main()
