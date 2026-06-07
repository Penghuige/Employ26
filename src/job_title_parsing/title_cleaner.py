"""岗位名称去噪模块。"""

from __future__ import annotations

from typing import Any, Dict, List
import re

from .match_utils import normalize_text


class JobTitleCleaner:
    """根据规则清洗岗位名称中的营销词、薪资和门店信息。"""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("job_title_cleaning", {})
        self.noise_phrases: List[str] = cfg.get("noise_phrases", [])
        self.pure_noise_patterns: List[str] = cfg.get("pure_noise_patterns", [])
        self.salary_patterns: List[str] = cfg.get("salary_patterns", [])
        self.separator_noise_patterns: List[str] = cfg.get("separator_noise_patterns", [])
        self.location_suffixes: List[str] = cfg.get("location_suffixes", [])
        self.title_tail_code_patterns: List[str] = cfg.get(
            "title_tail_code_patterns",
            [
                r"[A-Za-z]{1,4}\d{2,}[A-Za-z0-9]*$",
                r"\d{3,}[A-Za-z]{1,4}$",
            ],
        )
        self.english_role_keywords: List[str] = [
            "ceo",
            "cto",
            "coo",
            "cfo",
            "hr",
            "hrbp",
            "pm",
            "bd",
            "ui",
            "ux",
            "engineer",
            "developer",
            "manager",
            "analyst",
            "architect",
            "consultant",
            "specialist",
            "designer",
            "scientist",
            "product",
            "qa",
            "qc",
            "devops",
            "sre",
            "seo",
            "sem",
            "java",
            "python",
            "android",
            "ios",
        ]
        self.core_tail_patterns = [r"店$", r"门店$", r"商场$"]

    def clean(self, job_title: str) -> str:
        """清洗岗位名称，去除招聘营销词、薪资、地点、编码等噪声，仅保留职业核心词。

        处理顺序：纯噪声模式过滤 → 噪声短语替换 → 薪资表达移除 → 括号内容移除 →
                  分隔符噪声裁剪 → 地点后缀移除 → 型号尾号移除 → 城市/品牌前缀移除 →
                  剩余福利词移除 → whitespace 收拢 → 英文角色名守卫。

        Args:
            job_title: 原始岗位名称。

        Returns:
            str: 清洗后的岗位名称核心部分，若完全为噪声则返回空字符串。
        """
        text = normalize_text(job_title)
        if not text:
            return ""

        for pattern in self.pure_noise_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return ""

        # 先替换纯噪音词
        for phrase in self.noise_phrases:
            text = text.replace(phrase, " ")

        # 去掉薪资表达
        for pattern in self.salary_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # 去掉括号信息
        text = re.sub(r"[\(（][^\)）]*[\)）]", " ", text)

        # 按分隔符裁剪，若后半段是福利/门店噪音则丢弃
        text = self._trim_separator_noise(text)

        # 去掉尾部地点/门店信息
        for suffix in sorted(self.location_suffixes, key=len, reverse=True):
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip(" -_/|")

        # 去掉尾部型号/编号（仅在中文岗位主体下启用，避免误删纯英文岗位）
        text = self._strip_tail_code(text)

        # 去掉开头常见城市/平台品牌噪音
        text = re.sub(r"^(福田|龙岗|南山|罗湖|天河|白云|越秀|海珠|番禺|宝安|龙华)", "", text)
        text = re.sub(r"^(美团|饿了么|FILA|耐克|阿迪达斯)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[A-Za-z]{2,}(?=[\u4e00-\u9fa5]{2,})", "", text)

        # 去掉剩余福利类字眼
        for noise in self.separator_noise_patterns:
            text = text.replace(noise, " ")

        text = re.sub(r"[,_|/\-]+", " ", text)
        text = re.sub(r"\s+", "", text)

        # 纯数字/无职业含义的英文编码残片直接判空，常见英文岗位缩写保留。
        if re.fullmatch(r"[\dA-Za-z]+", text or ""):
            return text if self._is_pure_english_role(text) else ""

        # 剩余文本过于口号化则判空
        if len(text) <= 1:
            return ""
        if any(re.search(p, text) for p in self.core_tail_patterns) and len(text) <= 3:
            return ""
        return text

    def _is_pure_english_role(self, text: str) -> bool:
        """判断纯英文/字母数字文本是否像岗位名，而不是内部编码。"""
        source = str(text or "").strip()
        if not source or source.isdigit():
            return False
        lower = source.lower()
        if any(keyword in lower for keyword in self.english_role_keywords):
            return True
        return bool(re.fullmatch(r"[A-Z]{2,6}", source))

    def _strip_tail_code(self, text: str) -> str:
        """删除尾部型号编码，避免误删英文岗位名称。"""
        source = str(text).strip()
        if not source:
            return source

        has_chinese = bool(re.search(r"[\u4e00-\u9fa5]", source))
        has_space = " " in source
        pure_english_role = has_space and not has_chinese and any(
            keyword in source.lower() for keyword in self.english_role_keywords
        )
        if pure_english_role:
            return source

        if not has_chinese:
            return source

        for pattern in self.title_tail_code_patterns:
            trimmed = re.sub(pattern, "", source).strip(" -_/|")
            if trimmed != source and len(trimmed) >= 2:
                return trimmed
        return source

    def _trim_separator_noise(self, text: str) -> str:
        """处理 -, _, /, | 之后的纯噪音尾巴。"""
        for sep in ["-", "_", "/", "|", "·"]:
            if sep not in text:
                continue
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if not parts:
                continue
            kept = [parts[0]]
            for part in parts[1:]:
                if any(keyword in part for keyword in self.separator_noise_patterns):
                    continue
                if any(part.endswith(loc) or loc in part for loc in self.location_suffixes):
                    continue
                kept.append(part)
            text = " ".join(kept)
        return text
