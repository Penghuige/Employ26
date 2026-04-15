"""
职业细类技能词典读写模块。

词典统一保存在 `dicts/occupation_skill_dictionary.json`。
同时提供 LLM 输出导入、标准化与合并能力。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List

import pandas as pd


SUPPORTED_IMPORT_SUFFIXES = {".json", ".md", ".txt"}


def _safe_text(value: object) -> str:
    """安全转字符串并去除首尾空白。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_skill_key(text: object) -> str:
    """生成技能去重键。

    这里只做保守归一化：
    - 小写
    - 折叠空白
    不去掉 `+` `#` `.` 等符号，避免误把 `C++` 和 `C#` 合并。
    """
    value = _safe_text(text).lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _unique_keep_order(items: Iterable[object]) -> List[str]:
    """按原顺序去重。"""
    seen = set()
    result: List[str] = []
    for item in items:
        text = _safe_text(item)
        if not text:
            continue
        key = _normalize_skill_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _extract_json_from_fenced_blocks(text: str) -> List[str]:
    """从 Markdown fenced code block 中提取 JSON 片段。"""
    pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text) if match.group(1).strip()]


def _extract_balanced_json_fragments(text: str) -> List[str]:
    """提取文本中的平衡 JSON 对象或数组。"""
    fragments: List[str] = []
    stack: List[str] = []
    start_index: int | None = None
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            if not stack:
                start_index = index
            stack.append(char)
            continue

        if char in "}]":
            if not stack:
                continue
            expected = "{" if char == "}" else "["
            if stack[-1] != expected:
                stack.clear()
                start_index = None
                continue
            stack.pop()
            if not stack and start_index is not None:
                fragment = text[start_index : index + 1].strip()
                if fragment:
                    fragments.append(fragment)
                start_index = None

    return fragments


class OccupationSkillDictionaryStore:
    """职业细类技能词典存储器。"""

    def __init__(self, dictionary_path: str | Path):
        self.dictionary_path = Path(dictionary_path)

    def load(self) -> Dict:
        """加载词典；不存在时返回空骨架。"""
        if not self.dictionary_path.exists():
            return self._empty_dictionary()
        with open(self.dictionary_path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if "categories" not in data:
            data["categories"] = {}
        if "metadata" not in data:
            data["metadata"] = {}
        return data

    def save(self, dictionary: Dict) -> None:
        """保存词典。"""
        self.dictionary_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary.setdefault("metadata", {})
        dictionary["metadata"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(self.dictionary_path, "w", encoding="utf-8") as file_obj:
            json.dump(dictionary, file_obj, ensure_ascii=False, indent=2)

    def ensure_categories(self, category_summary_df: pd.DataFrame) -> Dict:
        """根据采样结果初始化或补齐职业细类词典骨架。"""
        dictionary = self.load()
        categories = dictionary.setdefault("categories", {})

        for row in category_summary_df.to_dict(orient="records"):
            detail_path = str(row["detail_path"])
            categories.setdefault(
                detail_path,
                {
                    "detail_name": row["detail_name"],
                    "hierarchy": {
                        "大类": row["大类"],
                        "中类": row["中类"],
                        "小类": row["小类"],
                        "细类": row["细类"],
                    },
                    "available_count": int(row["available_count"]),
                    "train_count": int(row["train_count"]),
                    "validation_pool_count": int(row["validation_pool_count"]),
                    "skills": [],
                },
            )

        self.save(dictionary)
        return dictionary

    def get_skills(self, dictionary: Dict, detail_path: str) -> List[Dict]:
        """读取某个职业细类的技能条目。"""
        return list(dictionary.get("categories", {}).get(detail_path, {}).get("skills", []))

    def get_skill_terms(self, dictionary: Dict, detail_path: str) -> List[str]:
        """读取某个职业细类的技能词及别名。"""
        skill_terms: List[str] = []
        for skill in self.get_skills(dictionary, detail_path):
            name = _safe_text(skill.get("name", ""))
            if name:
                skill_terms.append(name)
            aliases = skill.get("aliases", []) or []
            for alias in aliases:
                alias_text = _safe_text(alias)
                if alias_text:
                    skill_terms.append(alias_text)
        return skill_terms

    def import_from_path(self, source_path: str | Path, recursive: bool = True, dry_run: bool = False) -> Dict:
        """把 LLM 返回的 JSON 从文件或目录导入词典。"""
        dictionary = self.load()
        import_stats = {
            "source_path": str(Path(source_path)),
            "processed_files": 0,
            "loaded_payloads": 0,
            "updated_categories": 0,
            "created_categories": 0,
            "created_skills": 0,
            "merged_skills": 0,
            "skipped_payloads": 0,
            "files": [],
        }

        touched_categories = set()
        created_categories = set()

        for file_path in self._iter_import_files(source_path, recursive=recursive):
            payloads = self._load_payloads_from_file(file_path)
            file_stats = {
                "file": str(file_path),
                "payload_count": len(payloads),
                "imported_payload_count": 0,
                "skipped_payload_count": 0,
            }
            import_stats["processed_files"] += 1
            import_stats["loaded_payloads"] += len(payloads)

            for payload in payloads:
                merge_result = self._merge_payload_into_dictionary(dictionary, payload)
                if merge_result["skipped"]:
                    import_stats["skipped_payloads"] += 1
                    file_stats["skipped_payload_count"] += 1
                    continue

                touched_categories.add(merge_result["detail_path"])
                if merge_result["created_category"]:
                    created_categories.add(merge_result["detail_path"])
                import_stats["created_skills"] += merge_result["created_skills"]
                import_stats["merged_skills"] += merge_result["merged_skills"]
                file_stats["imported_payload_count"] += 1

            import_stats["files"].append(file_stats)

        import_stats["updated_categories"] = len(touched_categories)
        import_stats["created_categories"] = len(created_categories)

        dictionary.setdefault("metadata", {})
        dictionary["metadata"]["last_import_summary"] = {
            "processed_files": import_stats["processed_files"],
            "loaded_payloads": import_stats["loaded_payloads"],
            "updated_categories": import_stats["updated_categories"],
            "created_categories": import_stats["created_categories"],
            "created_skills": import_stats["created_skills"],
            "merged_skills": import_stats["merged_skills"],
            "imported_at": datetime.now().isoformat(timespec="seconds"),
        }

        if not dry_run:
            self.save(dictionary)

        return import_stats

    def _merge_payload_into_dictionary(self, dictionary: Dict, payload: Dict) -> Dict:
        """把单个 LLM payload 合并进词典。"""
        detail_path = _safe_text(payload.get("detail_path", ""))
        if not detail_path:
            return {
                "skipped": True,
                "detail_path": "",
                "created_category": False,
                "created_skills": 0,
                "merged_skills": 0,
            }

        skills_key = "skills" if isinstance(payload.get("skills"), list) else "missing_skills"
        raw_skills = payload.get(skills_key, [])
        normalized_skills = [self._normalize_skill_item(item) for item in raw_skills]
        normalized_skills = [item for item in normalized_skills if item is not None]
        if not normalized_skills:
            return {
                "skipped": True,
                "detail_path": detail_path,
                "created_category": False,
                "created_skills": 0,
                "merged_skills": 0,
            }

        categories = dictionary.setdefault("categories", {})
        created_category = False
        if detail_path not in categories:
            categories[detail_path] = self._build_fallback_category(
                detail_path=detail_path,
                detail_name=_safe_text(payload.get("detail_name", "")),
            )
            created_category = True

        category = categories[detail_path]
        category.setdefault("skills", [])
        existing_skills = category["skills"]

        created_skills = 0
        merged_skills = 0
        for new_skill in normalized_skills:
            merge_target = self._find_existing_skill(existing_skills, new_skill)
            if merge_target is None:
                existing_skills.append(new_skill)
                created_skills += 1
                continue
            self._merge_skill_record(merge_target, new_skill)
            merged_skills += 1

        category["skills"] = sorted(
            existing_skills,
            key=lambda item: (_safe_text(item.get("name", "")).lower(), _safe_text(item.get("skill_type", "")).lower()),
        )
        return {
            "skipped": False,
            "detail_path": detail_path,
            "created_category": created_category,
            "created_skills": created_skills,
            "merged_skills": merged_skills,
        }

    def _load_payloads_from_file(self, file_path: Path) -> List[Dict]:
        """从单个文件中提取一个或多个 JSON payload。"""
        text = file_path.read_text(encoding="utf-8")
        candidates: List[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)
        candidates.extend(_extract_json_from_fenced_blocks(text))
        candidates.extend(_extract_balanced_json_fragments(text))

        payloads: List[Dict] = []
        seen = set()
        for candidate in candidates:
            normalized_candidate = candidate.strip()
            if not normalized_candidate or normalized_candidate in seen:
                continue
            seen.add(normalized_candidate)
            try:
                loaded = json.loads(normalized_candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(loaded, dict):
                payloads.extend(self._collect_payload_objects(loaded))
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict):
                        payloads.extend(self._collect_payload_objects(item))
        return payloads

    @staticmethod
    def _collect_payload_objects(obj: Dict) -> List[Dict]:
        """从对象中收集可导入 payload。"""
        if _safe_text(obj.get("detail_path", "")) and (
            isinstance(obj.get("skills"), list) or isinstance(obj.get("missing_skills"), list)
        ):
            return [obj]
        return []

    @staticmethod
    def _iter_import_files(source_path: str | Path, recursive: bool) -> List[Path]:
        """列出可导入文件。"""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"导入路径不存在: {path}")
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
                raise ValueError(f"不支持的导入文件类型: {path.suffix}")
            return [path]

        pattern = "**/*" if recursive else "*"
        files = [
            file_path
            for file_path in path.glob(pattern)
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMPORT_SUFFIXES
        ]
        return sorted(files)

    @staticmethod
    def _build_fallback_category(detail_path: str, detail_name: str) -> Dict:
        """为未初始化的细类构建兜底骨架。"""
        parts = [part.strip() for part in str(detail_path).split(">")]
        parts = [part for part in parts if part]
        detail_name = detail_name or (parts[-1] if parts else detail_path)
        hierarchy_parts = (parts + ["", "", "", ""])[:4]
        return {
            "detail_name": detail_name,
            "hierarchy": {
                "大类": hierarchy_parts[0],
                "中类": hierarchy_parts[1],
                "小类": hierarchy_parts[2],
                "细类": hierarchy_parts[3],
            },
            "available_count": 0,
            "train_count": 0,
            "validation_pool_count": 0,
            "skills": [],
        }

    @staticmethod
    def _normalize_skill_item(item: object) -> Dict | None:
        """标准化 LLM 返回的技能条目。"""
        if isinstance(item, str):
            name = _safe_text(item)
            if not name:
                return None
            return {
                "name": name,
                "aliases": [],
                "skill_type": "",
                "notes": "",
            }

        if not isinstance(item, dict):
            return None

        name = _safe_text(item.get("name", ""))
        if not name:
            return None

        aliases = item.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [part.strip() for part in re.split(r"[|,，/]", aliases) if part.strip()]
        elif not isinstance(aliases, list):
            aliases = []

        alias_values = [alias for alias in _unique_keep_order(aliases) if _normalize_skill_key(alias) != _normalize_skill_key(name)]
        return {
            "name": name,
            "aliases": alias_values,
            "skill_type": _safe_text(item.get("skill_type", "")),
            "notes": _safe_text(item.get("notes", "")),
        }

    @staticmethod
    def _find_existing_skill(existing_skills: List[Dict], new_skill: Dict) -> Dict | None:
        """按 name/alias 查找可合并技能。"""
        incoming_keys = {
            _normalize_skill_key(new_skill.get("name", "")),
            *[_normalize_skill_key(alias) for alias in new_skill.get("aliases", [])],
        }
        incoming_keys.discard("")

        for existing_skill in existing_skills:
            existing_keys = {
                _normalize_skill_key(existing_skill.get("name", "")),
                *[_normalize_skill_key(alias) for alias in existing_skill.get("aliases", [])],
            }
            existing_keys.discard("")
            if incoming_keys & existing_keys:
                return existing_skill
        return None

    @staticmethod
    def _merge_skill_record(target: Dict, incoming: Dict) -> None:
        """把新技能信息并到已有技能条目。"""
        target.setdefault("aliases", [])
        merged_aliases = _unique_keep_order(
            [
                *target.get("aliases", []),
                incoming.get("name", ""),
                *incoming.get("aliases", []),
            ]
        )
        merged_aliases = [alias for alias in merged_aliases if _normalize_skill_key(alias) != _normalize_skill_key(target.get("name", ""))]
        target["aliases"] = merged_aliases

        if not _safe_text(target.get("skill_type", "")) and _safe_text(incoming.get("skill_type", "")):
            target["skill_type"] = incoming["skill_type"]

        existing_notes = _safe_text(target.get("notes", ""))
        incoming_notes = _safe_text(incoming.get("notes", ""))
        if incoming_notes:
            if not existing_notes:
                target["notes"] = incoming_notes
            elif incoming_notes not in existing_notes:
                target["notes"] = f"{existing_notes} | {incoming_notes}"

    @staticmethod
    def _empty_dictionary() -> Dict:
        """构造空词典。"""
        return {
            "metadata": {
                "schema_version": 1,
                "description": "按职业细类维护的技能词典",
            },
            "categories": {},
        }
