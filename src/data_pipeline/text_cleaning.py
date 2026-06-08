"""招聘文本清洗工具。"""

import html
import math
import re

from src.data_pipeline.description_schema import MIDLINE_PATTERN, OPTIONAL_NOTE_RE


def strip_tags(text: str) -> str:
    """移除 HTML 标签，并把段落类标签转换为换行。"""
    text = html.unescape(text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"(?is)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?is)</\s*(div|p|li|tr|table|ul|ol|section|article)\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    return text


def remove_noise(text: str) -> str:
    """移除零宽字符和明显乱码式标点噪声。"""
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(
        r"(?<=[A-Za-z0-9\u4e00-\u9fff])[\?？·•●▪◆★※~_]+(?=[A-Za-z0-9\u4e00-\u9fff])",
        "",
        text,
    )
    text = re.sub(r"[?？]{3,}", "", text)
    return text


def sanitize_item(text: str) -> str:
    """清理单个切分条目两端的噪声符号和多余空白。"""
    text = remove_noise(text).replace("?", "").replace("？", "")
    text = re.sub(r"^[?？·•●▪◆★※\s\"']+", "", text)
    text = re.sub(r"[?？·•●▪◆★※\s\"']+$", "", text)
    text = text.strip("[] ")
    text = re.sub(r"\s+", " ", text).strip(" ;；")
    if text in {"[", "]", '"', "'", "[]", "----", "-----"}:
        return ""
    return text


def normalize_text(text: str) -> str:
    """清洗岗位描述全文，并为内联标题、编号列表补充换行边界。"""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)
    text = strip_tags(text)
    text = remove_noise(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .replace("【", "[")
        .replace("】", "]")
        .replace("．", ".")
    )
    text = re.sub(r"[ \t\u3000\xa0]+", " ", text)
    inline_re = re.compile(
        rf"(?<!^)(?<!\n)(?P<prefix>[；;。.!?？!\"']|\s)\s*"
        rf"(?P<head>(?:\[)?(?:{MIDLINE_PATTERN})(?:\])?{OPTIONAL_NOTE_RE}\s*(?::)?)"
    )
    for _ in range(3):
        text = inline_re.sub(lambda m: f"{m.group('prefix')}\n{m.group('head')}", text)
    text = re.sub(r"(?<!^)(?<!\n)(?=[一二三四五六七八九十]+[、.])", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
