"""
岗位名称解析器
使用最长后缀匹配算法从岗位名称中分离职业核心词和修饰词。

数据源约定：
1. 解析输入来自 DuckDB 中后缀为 `_sample` 的表
2. 职业核心词来源由 OccupationDictManager 固定到
   `recruit.main.chinese_occupational_dictionary_joined`
3. 类别映射来源固定到
   `recruit.main.chinese_occupational_dictionary`
"""

from pathlib import Path
from typing import Dict, List, Optional
import logging
import jieba
import re

from src.job_title_parsing.occupation_dict_manager import OccupationDictManager

logger = logging.getLogger(__name__)


class OccupationParser:
    """岗位名称解析器。"""

    def __init__(self, dict_dir=None, db_path=None):
        """初始化。

        Args:
            dict_dir: 词典目录路径
            db_path: DuckDB 数据库路径（可选）
        """
        self.dict_manager = OccupationDictManager(dict_dir=dict_dir, db_path=db_path)

        # 加载词典
        self.occupation_cores = self.dict_manager.load_occupation_cores()
        self.modifiers = self.dict_manager.load_modifiers()
        self.sorted_cores = self.dict_manager.get_core_by_priority()

        # 从 dicts 加载禁用词/停用词（避免硬编码）
        self.welfare_keywords = self.dict_manager.load_welfare_keywords()
        self.location_keywords = self.dict_manager.load_location_keywords()
        self.invalid_suffix_keywords = self.dict_manager.load_invalid_suffix_keywords()
        self.modifier_stopwords = self.dict_manager.load_modifier_stopwords()

        logger.info("岗位名称解析器初始化完成")
        logger.info(f"  职业核心词: {len(self.occupation_cores)} 个")
        logger.info(f"  修饰词: {len(self.modifiers)} 个")

    def _preprocess_job_title(self, job_title: str) -> str:
        """预处理岗位名称，清洗干扰信息。"""
        cleaned = job_title

        # 1. 移除括号及其内容（包括中英文括号）
        cleaned = re.sub(r"[（(][^）)]*[）)]", "", cleaned)
        cleaned = re.sub(r"[\[【][^\]】]*[\]】]", "", cleaned)

        # 2. 移除年份标识（如 -2025、2025、-2024）
        cleaned = re.sub(r"-?\d{4}", "", cleaned)

        # 3. 移除编号（如 BP-2025、GNZW03）
        cleaned = re.sub(r"[A-Z]{2,}\d+", "", cleaned)
        cleaned = re.sub(r"[A-Z]{2,}-\d+", "", cleaned)
        cleaned = re.sub(r"\b\d{2,}\b", "", cleaned)

        # 4. 移除开头福利词（来自 dicts）
        for keyword in self.welfare_keywords:
            if cleaned.startswith(keyword):
                cleaned = cleaned[len(keyword) :].strip()

        # 5. 移除开头/结尾地点词（来自 dicts）
        for location in self.location_keywords:
            if cleaned.startswith(location):
                cleaned = cleaned[len(location) :].strip()
                break

        for location in self.location_keywords:
            if cleaned.endswith(location) and len(cleaned) > len(location):
                if cleaned[-len(location) - 1 : -len(location)] in [" ", "-", "/", "·"]:
                    cleaned = cleaned[: -len(location)].strip()
                    break

        # 6. 清理特殊分隔符
        cleaned = re.sub(r"[/／\\]+", " ", cleaned)
        cleaned = re.sub(r"[-－—]+", " ", cleaned)
        cleaned = re.sub(r"[·•]+", " ", cleaned)

        # 7. 清理空格
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned if cleaned else job_title

    def parse(self, job_title: str) -> Dict:
        """解析岗位名称。"""
        original_title = job_title.strip()
        if not original_title:
            return self._empty_result(original_title)

        job_title_cleaned = self._preprocess_job_title(original_title)

        result = self._longest_suffix_match(job_title_cleaned)
        if result:
            result["job_title_raw"] = original_title
            return result

        result = self._contains_match(job_title_cleaned)
        if result:
            result["job_title_raw"] = original_title
            return result

        return self._fallback_result(original_title)

    def _longest_suffix_match(self, job_title: str) -> Optional[Dict]:
        """最长后缀匹配。"""
        for core_word, _priority in self.sorted_cores:
            if job_title.endswith(core_word):
                modifiers_part = job_title[: -len(core_word)].strip()
                modifier_list = self._extract_modifiers(modifiers_part)
                category = self.dict_manager.get_core_category(core_word)
                return {
                    "job_title_raw": job_title,
                    "occupation_core": core_word,
                    "modifiers": modifiers_part,
                    "modifier_list": modifier_list,
                    "core_category": category,
                    "confidence": 1.0,
                    "match_method": "longest_suffix",
                }
        return None

    def _contains_match(self, job_title: str) -> Optional[Dict]:
        """包含匹配（用于核心词在中间的情况）。"""
        for core_word, _priority in self.sorted_cores:
            if core_word in job_title:
                idx = job_title.find(core_word)
                before = job_title[:idx].strip()
                after = job_title[idx + len(core_word) :].strip()

                modifiers_part = before
                if after and self._is_valid_suffix(after):
                    modifiers_part = f"{before} {after}".strip()

                modifier_list = self._extract_modifiers(modifiers_part)
                category = self.dict_manager.get_core_category(core_word)

                return {
                    "job_title_raw": job_title,
                    "occupation_core": core_word,
                    "modifiers": modifiers_part,
                    "modifier_list": modifier_list,
                    "core_category": category,
                    "confidence": 0.8,
                    "match_method": "contains",
                }
        return None

    def _is_valid_suffix(self, text: str) -> bool:
        """判断后缀是否是有效修饰词。"""
        text = re.sub(r"[（）\(\)]", "", text).strip()
        if not text:
            return False

        if text in self.modifiers:
            return True

        if len(text) > 10:
            return False

        for keyword in self.invalid_suffix_keywords:
            if keyword in text:
                return False
        return True

    def _extract_modifiers(self, modifiers_part: str) -> List[str]:
        """提取修饰词列表。"""
        if not modifiers_part:
            return []

        modifiers_part = re.sub(r"[（）\(\)\[\]【】]", " ", modifiers_part)
        words = jieba.lcut(modifiers_part)

        valid_modifiers: List[str] = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            if len(word) == 1 and not (word.isupper() and word.isalpha()):
                continue
            if word in self.modifier_stopwords:
                continue
            valid_modifiers.append(word)

        return valid_modifiers

    def _fallback_result(self, job_title: str) -> Dict:
        """兜底结果（无法匹配时）。"""
        words = jieba.lcut(job_title)
        if len(words) >= 2:
            core_word = words[-1]
            modifiers_part = "".join(words[:-1])
            modifier_list = words[:-1]
        else:
            core_word = job_title
            modifiers_part = ""
            modifier_list = []

        return {
            "job_title_raw": job_title,
            "occupation_core": core_word,
            "modifiers": modifiers_part,
            "modifier_list": modifier_list,
            "core_category": "未知",
            "confidence": 0.3,
            "match_method": "fallback",
        }

    def _empty_result(self, job_title: str) -> Dict:
        """空结果。"""
        return {
            "job_title_raw": job_title,
            "occupation_core": "",
            "modifiers": "",
            "modifier_list": [],
            "core_category": "未知",
            "confidence": 0.0,
            "match_method": "empty",
        }

    def parse_batch(self, job_titles: List[str]) -> List[Dict]:
        """批量解析岗位名称。"""
        return [self.parse(str(job_title)) for job_title in job_titles]

    def parse_duckdb_table(
        self,
        db_path: str,
        source_table: str,
        target_table: Optional[str] = None,
        job_title_col: str = "岗位名称",
    ) -> int:
        """从 DuckDB 读取表并解析岗位名称，结果写回新表。

        Args:
            db_path: DuckDB 文件路径
            source_table: 源表全名（如 recruit.main.xxx_sample）
            target_table: 目标表全名，默认源表名后缀替换为 `_parsed`
            job_title_col: 岗位名称列名，默认“岗位名称”

        Returns:
            写入目标表的行数
        """
        import duckdb
        import pandas as pd

        if target_table is None:
            if source_table.endswith("_sample"):
                target_table = source_table[:-7] + "_parsed"
            else:
                target_table = source_table + "_parsed"

        conn = duckdb.connect(db_path)
        try:
            df = conn.execute(f"SELECT * FROM {source_table}").df()
            if job_title_col not in df.columns:
                raise KeyError(f"源表缺少列: {job_title_col}")

            parsed = pd.DataFrame(self.parse_batch(df[job_title_col].fillna("").astype(str).tolist()))
            df["occupation_core"] = parsed["occupation_core"]
            df["occupation_category"] = parsed["core_category"]
            df["occupation_modifiers"] = parsed["modifiers"]
            df["occupation_confidence"] = parsed["confidence"]
            df["occupation_match_method"] = parsed["match_method"]

            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            conn.register("tmp_parsed", df)
            conn.execute(f"CREATE TABLE {target_table} AS SELECT * FROM tmp_parsed")
            conn.unregister("tmp_parsed")

            row_count = conn.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
            logger.info(f"解析完成: {source_table} -> {target_table} ({row_count} 行)")
            return int(row_count)
        finally:
            conn.close()

    def parse_duckdb_sample_tables(
        self,
        db_path: str,
        catalog: str = "recruit",
        schema: str = "main",
        suffix: str = "_sample",
        job_title_col: str = "岗位名称",
    ) -> int:
        """批量解析 DuckDB 中后缀为 `_sample` 的表。

        Args:
            db_path: DuckDB 文件路径
            catalog: catalog 名称
            schema: schema 名称
            suffix: 源表后缀，默认 `_sample`
            job_title_col: 岗位名称列名

        Returns:
            总解析行数
        """
        import duckdb

        conn = duckdb.connect(db_path)
        try:
            tables = conn.execute(
                f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_catalog = '{catalog}'
                  AND table_schema = '{schema}'
                  AND table_name LIKE '%{suffix}'
                """
            ).fetchall()
            table_names = [f"{catalog}.{schema}.{r[0]}" for r in tables]
        finally:
            conn.close()

        total = 0
        for source_table in table_names:
            total += self.parse_duckdb_table(
                db_path=db_path,
                source_table=source_table,
                target_table=source_table.replace(suffix, "_parsed"),
                job_title_col=job_title_col,
            )
        return total

    def get_statistics(self, results: List[Dict]) -> Dict:
        """统计解析结果。"""
        from collections import Counter

        total = len(results)
        category_count = Counter(r["core_category"] for r in results)
        method_count = Counter(r["match_method"] for r in results)
        confidence_dist = {
            "高置信(1.0)": sum(1 for r in results if r["confidence"] == 1.0),
            "中置信(0.8)": sum(1 for r in results if r["confidence"] == 0.8),
            "低置信(0.3-0.5)": sum(1 for r in results if 0.3 <= r["confidence"] < 0.8),
            "无效(0.0)": sum(1 for r in results if r["confidence"] == 0.0),
        }

        core_count = Counter(r["occupation_core"] for r in results if r["occupation_core"])
        modifier_count = Counter()
        for r in results:
            for mod in r["modifier_list"]:
                modifier_count[mod] += 1

        return {
            "total": total,
            "category_distribution": dict(category_count),
            "method_distribution": dict(method_count),
            "confidence_distribution": confidence_dist,
            "top_cores": core_count.most_common(20),
            "top_modifiers": modifier_count.most_common(30),
            "coverage": sum(1 for r in results if r["confidence"] >= 0.8) / total if total > 0 else 0.0,
        }


def main():
    """命令行入口：默认解析 DuckDB 中 `_sample` 表。"""
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="岗位名称解析（DuckDB）")
    parser.add_argument(
        "--db",
        default=OccupationDictManager.DEFAULT_DB_PATH,
        help="DuckDB 文件路径",
    )
    parser.add_argument(
        "--catalog",
        default="recruit",
        help="catalog 名称",
    )
    parser.add_argument(
        "--schema",
        default="main",
        help="schema 名称",
    )
    parser.add_argument(
        "--suffix",
        default="_sample",
        help="待解析源表后缀，默认 _sample",
    )
    parser.add_argument(
        "--job-title-col",
        default="岗位名称",
        help="岗位名称列名",
    )
    args = parser.parse_args()

    parser_obj = OccupationParser(db_path=args.db)
    total = parser_obj.parse_duckdb_sample_tables(
        db_path=args.db,
        catalog=args.catalog,
        schema=args.schema,
        suffix=args.suffix,
        job_title_col=args.job_title_col,
    )
    logger.info(f"解析完成，总行数: {total}")


if __name__ == "__main__":
    main()
