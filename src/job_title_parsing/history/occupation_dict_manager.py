"""
职业词典管理器

职责：
1. 从 DuckDB 固定表加载职业核心词（岗位名称来源固定表）
2. 从《中国职业大典》表加载类别层级（大类/中类/小类/细类）
3. 从 dicts 目录加载禁用词与修饰词词典
"""

from pathlib import Path
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


class OccupationDictManager:
    """职业词典管理器。"""

    DEFAULT_DB_PATH = r"D:\PythonProjects\Employ26\output\recruit.duckdb"
    # 固定岗位来源表（按你的要求）
    DEFAULT_OCCUPATION_JOINED_TABLE = "recruit.main.chinese_occupational_dictionary_joined"
    # 固定类别来源表（中国职业大典）
    DEFAULT_CATEGORY_TABLE = "recruit.main.chinese_occupational_dictionary"

    def __init__(
        self,
        dict_dir=None,
        db_path=None,
        occupation_joined_table=None,
        category_table=None,
    ):
        """初始化词典管理器。

        Args:
            dict_dir: 词典目录路径，默认项目根目录下 `dicts`
            db_path: DuckDB 文件路径
            occupation_joined_table: 岗位名称来源表
            category_table: 类别来源表（中国职业大典）
        """
        if dict_dir is None:
            self.dict_dir = Path(__file__).parent.parent.parent / "dicts"
        else:
            self.dict_dir = Path(dict_dir)

        if not self.dict_dir.exists():
            raise FileNotFoundError(f"词典目录不存在: {self.dict_dir}")

        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.occupation_joined_table = (
            occupation_joined_table or self.DEFAULT_OCCUPATION_JOINED_TABLE
        )
        self.category_table = category_table or self.DEFAULT_CATEGORY_TABLE

        # 缓存
        self._occupation_cores = None
        self._modifiers = None
        self._welfare_keywords = None
        self._location_keywords = None
        self._invalid_suffix_keywords = None
        self._modifier_stopwords = None
        self._category_code_map = None

        logger.info(f"职业词典管理器初始化: dict_dir={self.dict_dir}")
        logger.info(f"岗位来源表: {self.occupation_joined_table}")
        logger.info(f"类别来源表: {self.category_table}")

    def _load_simple_word_list(self, filename: str) -> List[str]:
        """从 dicts 目录加载简单词表（每行一个词，支持 # 注释）。"""
        path = self.dict_dir / filename
        if not path.exists():
            logger.warning(f"词表不存在: {path}")
            return []

        words: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                words.append(line)
        return words

    def _resolve_column(self, columns: List[str], candidates: List[str]) -> str:
        """从候选列名中解析可用列名。"""
        for c in candidates:
            if c in columns:
                return c
        return ""

    def _load_category_map_from_dictionary(self) -> Dict[str, Dict[str, str]]:
        """从《中国职业大典》表加载 code -> 类别层级映射。

        目标：将 1-02-05-01 这类层级编码映射为可读类别路径，
        优先使用表中的 `大类/中类/小类/细类` 列。

        Returns:
            Dict[code, {
                "major": 大类,
                "middle": 中类,
                "minor": 小类,
                "detail": 细类,
                "category": 组合类别字符串
            }]
        """
        import duckdb

        conn = duckdb.connect(self.db_path)
        try:
            desc = conn.execute(f"SELECT * FROM {self.category_table} LIMIT 0").description
            columns = [c[0] for c in desc]

            code_col = self._resolve_column(
                columns,
                ["code", "职业编码", "分类代码", "occupation_code", "category_code"],
            )
            major_col = self._resolve_column(columns, ["大类", "major", "major_class"])
            middle_col = self._resolve_column(columns, ["中类", "middle", "middle_class"])
            minor_col = self._resolve_column(columns, ["小类", "minor", "minor_class"])
            detail_col = self._resolve_column(columns, ["细类", "detail", "detail_class", "title"])

            if not code_col:
                raise ValueError(
                    f"类别表缺少编码列，无法建立映射。可用列: {columns}"
                )

            select_cols = [code_col]
            for col in [major_col, middle_col, minor_col, detail_col]:
                if col:
                    select_cols.append(col)

            rows = conn.execute(
                f"SELECT DISTINCT {', '.join(select_cols)} FROM {self.category_table}"
            ).fetchall()

            mapping: Dict[str, Dict[str, str]] = {}
            for row in rows:
                code = "" if row[0] is None else str(row[0]).strip()
                if not code:
                    continue

                idx = 1
                major = ""
                middle = ""
                minor = ""
                detail = ""

                if major_col:
                    major = "" if row[idx] is None else str(row[idx]).strip()
                    idx += 1
                if middle_col:
                    middle = "" if row[idx] is None else str(row[idx]).strip()
                    idx += 1
                if minor_col:
                    minor = "" if row[idx] is None else str(row[idx]).strip()
                    idx += 1
                if detail_col:
                    detail = "" if row[idx] is None else str(row[idx]).strip()

                # 组合为层级类别字符串
                category_parts = [p for p in [major, middle, minor, detail] if p]
                category = " > ".join(category_parts) if category_parts else code

                mapping[code] = {
                    "major": major,
                    "middle": middle,
                    "minor": minor,
                    "detail": detail,
                    "category": category,
                }

            logger.info(f"加载类别映射成功: {len(mapping)} 条 (table={self.category_table})")
            return mapping
        finally:
            conn.close()

    def _match_category_by_code(self, code: str) -> str:
        """根据职业编码匹配类别。

        优先精确匹配；失败时按层级编码前缀回退匹配。
        """
        if not code:
            return "未知"

        category_map = self._category_code_map or {}

        # 1) 精确匹配
        if code in category_map:
            return category_map[code]["category"]

        # 2) 层级前缀匹配（如 1-02-05-01 -> 1-02-05 -> 1-02 -> 1）
        parts = code.split("-")
        for i in range(len(parts) - 1, 0, -1):
            prefix = "-".join(parts[:i])
            if prefix in category_map:
                return category_map[prefix]["category"]

        return "未知"

    def _load_cores_from_duckdb(self) -> Dict[str, Dict]:
        """从固定 joined 表加载职业核心词，并映射到职业大典类别。"""
        import duckdb

        # 先加载类别映射
        self._category_code_map = self._load_category_map_from_dictionary()

        conn = duckdb.connect(self.db_path)
        try:
            desc = conn.execute(
                f"SELECT * FROM {self.occupation_joined_table} LIMIT 0"
            ).description
            columns = [c[0] for c in desc]

            core_col = self._resolve_column(
                columns,
                [
                    "岗位名称",
                    "职业名称",
                    "occupation_name",
                    "title",
                    "细类",
                    "名称",
                ],
            )
            code_col = self._resolve_column(
                columns,
                ["职业编码", "code", "occupation_code", "分类代码", "category_code"],
            )

            if not core_col:
                raise ValueError(
                    f"岗位来源表缺少岗位名称列。可用列: {columns}"
                )

            select_fields = [core_col]
            if code_col:
                select_fields.append(code_col)

            rows = conn.execute(
                f"SELECT DISTINCT {', '.join(select_fields)} FROM {self.occupation_joined_table}"
            ).fetchall()

            cores: Dict[str, Dict] = {}
            for row in rows:
                core_word = "" if row[0] is None else str(row[0]).strip()
                if not core_word:
                    continue

                code = ""
                if code_col:
                    code = "" if row[1] is None else str(row[1]).strip()

                category = self._match_category_by_code(code)

                # 优先级策略：词长越长越优先
                priority = 100 + min(len(core_word), 20)

                cores[core_word] = {
                    "priority": priority,
                    "category": category,
                    "code": code,
                }

            logger.info(
                f"从 joined 表加载职业核心词成功: {len(cores)} 个 (table={self.occupation_joined_table})"
            )
            return cores
        finally:
            conn.close()

    def _load_cores_from_file(self) -> Dict[str, Dict]:
        """从本地文本词典加载核心词（兜底方案）。"""
        core_file = self.dict_dir / "occupation_cores.txt"
        if not core_file.exists():
            raise FileNotFoundError(f"职业核心词词典不存在: {core_file}")

        cores: Dict[str, Dict] = {}
        with open(core_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    logger.warning(f"行{line_num}: 格式错误，跳过: {line}")
                    continue

                word = parts[0]
                try:
                    priority = int(parts[1])
                except ValueError:
                    logger.warning(f"行{line_num}: 优先级格式错误，跳过: {line}")
                    continue

                category = parts[2]
                cores[word] = {"priority": priority, "category": category, "code": ""}

        logger.info(f"从本地词典加载职业核心词: {len(cores)} 个")
        return cores

    def load_occupation_cores(self) -> Dict[str, Dict]:
        """加载职业核心词词典（优先 DuckDB，失败回退本地）。"""
        if self._occupation_cores is not None:
            return self._occupation_cores

        try:
            cores = self._load_cores_from_duckdb()
        except Exception as e:
            logger.warning(f"DuckDB 加载失败，回退本地 occupation_cores.txt。原因: {e}")
            cores = self._load_cores_from_file()

        self._occupation_cores = cores
        return cores

    def load_modifiers(self) -> Dict[str, str]:
        """加载修饰词词典。"""
        if self._modifiers is not None:
            return self._modifiers

        modifier_file = self.dict_dir / "occupation_modifiers.txt"
        if not modifier_file.exists():
            logger.warning(f"修饰词词典不存在: {modifier_file}")
            self._modifiers = {}
            return self._modifiers

        modifiers: Dict[str, str] = {}
        with open(modifier_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    modifiers[parts[0]] = parts[1]
                else:
                    logger.warning(f"行{line_num}: 格式错误，跳过: {line}")

        self._modifiers = modifiers
        logger.info(f"加载修饰词词典: {len(modifiers)} 个")
        return modifiers

    def load_welfare_keywords(self) -> List[str]:
        """加载岗位标题福利相关禁用词。"""
        if self._welfare_keywords is None:
            self._welfare_keywords = self._load_simple_word_list(
                "job_title_welfare_keywords.txt"
            )
        return self._welfare_keywords

    def load_location_keywords(self) -> List[str]:
        """加载岗位标题地点相关禁用词。"""
        if self._location_keywords is None:
            self._location_keywords = self._load_simple_word_list(
                "job_title_location_keywords.txt"
            )
        return self._location_keywords

    def load_invalid_suffix_keywords(self) -> List[str]:
        """加载无效后缀关键词。"""
        if self._invalid_suffix_keywords is None:
            self._invalid_suffix_keywords = self._load_simple_word_list(
                "job_title_invalid_suffix_keywords.txt"
            )
        return self._invalid_suffix_keywords

    def load_modifier_stopwords(self) -> Set[str]:
        """加载修饰词分词停用词。"""
        if self._modifier_stopwords is None:
            self._modifier_stopwords = set(
                self._load_simple_word_list("job_title_modifier_stopwords.txt")
            )
        return self._modifier_stopwords

    def get_core_by_priority(self) -> List[Tuple[str, int]]:
        """获取按优先级排序的核心词列表。"""
        cores = self.load_occupation_cores()
        sorted_cores = sorted(
            cores.items(),
            key=lambda x: (-x[1]["priority"], -len(x[0])),
        )
        return [(word, info["priority"]) for word, info in sorted_cores]

    def get_core_category(self, core_word: str) -> str:
        """获取职业核心词类别。"""
        cores = self.load_occupation_cores()
        if core_word in cores:
            return cores[core_word]["category"]
        return "未知"

    def get_modifier_type(self, modifier: str) -> str:
        """获取修饰词类型。"""
        return self.load_modifiers().get(modifier, "未知")

    def get_all_cores_set(self) -> Set[str]:
        """获取所有核心词集合。"""
        return set(self.load_occupation_cores().keys())

    def get_cores_by_category(self) -> Dict[str, List[str]]:
        """按类别分组获取核心词。"""
        cores = self.load_occupation_cores()
        grouped: Dict[str, List[str]] = {}
        for word, info in cores.items():
            category = info["category"]
            grouped.setdefault(category, []).append(word)
        return grouped

    def export_statistics(self) -> Dict:
        """导出词典统计信息。"""
        cores = self.load_occupation_cores()
        modifiers = self.load_modifiers()

        category_count: Dict[str, int] = {}
        for info in cores.values():
            cat = info["category"]
            category_count[cat] = category_count.get(cat, 0) + 1

        modifier_type_count: Dict[str, int] = {}
        for mod_type in modifiers.values():
            modifier_type_count[mod_type] = modifier_type_count.get(mod_type, 0) + 1

        return {
            "total_cores": len(cores),
            "total_modifiers": len(modifiers),
            "core_categories": category_count,
            "modifier_types": modifier_type_count,
            "occupation_joined_table": self.occupation_joined_table,
            "category_table": self.category_table,
        }


def main():
    """测试函数。"""
    import logging

    logging.basicConfig(level=logging.INFO)
    manager = OccupationDictManager()

    cores = manager.load_occupation_cores()
    print(f"\n职业核心词数量: {len(cores)}")
    print(f"示例: {list(cores.items())[:5]}")

    modifiers = manager.load_modifiers()
    print(f"\n修饰词数量: {len(modifiers)}")
    print(f"示例: {list(modifiers.items())[:10]}")


if __name__ == "__main__":
    main()
