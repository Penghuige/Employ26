"""职业别名构建模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .match_utils import read_json, unique_keep_order


class AliasBuilder:
    """根据人工词典与规则，为职业标准名生成别名。"""

    def __init__(self, config: Dict[str, Any], alias_dict_path: str | Path | None = None):
        self.config = config
        self.alias_cfg = config.get("alias", {})
        self.weak_suffixes: List[str] = self.alias_cfg.get("weak_suffixes", [])
        self.manual_mapping: Dict[str, str] = self.alias_cfg.get("manual_mapping", {})
        self.external_alias_dict: Dict[str, List[str]] = {}
        if alias_dict_path and Path(alias_dict_path).exists():
            self.external_alias_dict = read_json(alias_dict_path)

    def build_aliases(self, title: str) -> List[str]:
        """为单个职业 title 生成别名列表。

        来源优先级：外部人工维护词典 → 弱后缀裁剪规则 → manual_mapping 反向映射。

        Args:
            title: 职业标准名称。

        Returns:
            List[str]: 去重后的别名列表。
        """
        title = str(title).strip()
        aliases: List[str] = []

        # 外部人工维护词典优先
        aliases.extend(self.external_alias_dict.get(title, []))

        # 规则生成弱别名：去掉常见职业尾缀
        for suffix in self.weak_suffixes:
            if title.endswith(suffix) and len(title) > len(suffix) + 1:
                aliases.append(title[: -len(suffix)])

        # 规则生成：若 title 恰好命中 manual_mapping 的 value，补充 key 作为口语别名
        for alias, canonical in self.manual_mapping.items():
            if canonical == title:
                aliases.append(alias)

        return unique_keep_order(aliases)

    def resolve_manual_alias(self, text: str) -> str:
        """将口语化/非标准岗位名映射到标准职业名。

        Args:
            text: 待映射的岗位名。

        Returns:
            str: 若命中 manual_mapping 则返回标准名，否则原样返回。
        """
        return self.manual_mapping.get(text, text)
