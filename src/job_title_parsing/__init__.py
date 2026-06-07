"""岗位名称解析与职业匹配模块。

当前活跃组件：
- CatalogPreprocessor: 职业大典预处理
- BM25Index: 双路 BM25 检索
- HierarchyFilter: 层级过滤
- ScoreFusion: 多信号融合打分
- MatchPipeline: 主匹配流程
- evaluate_matches / evaluate_matches_parallel: 匹配效果评估

已归档至 history/ 的早期模块：
- OccupationDictManager: 职业词典管理（被 MatchPipeline 方案替代）
- OccupationParser: 岗位名称解析器（被 title_cleaner + 检索方案替代）
- evaluate_parser.py: 解析器原型评估脚本
- evaluate_matching.py: 旧版 CSV 评估入口（被 cli.py evaluate 替代）
"""

__all__ = [
    "CatalogPreprocessor",
    "BM25Index",
    "HierarchyFilter",
    "ScoreFusion",
    "MatchPipeline",
    "evaluate_matches",
    "evaluate_matches_parallel",
]


def __getattr__(name: str):
    if name == "CatalogPreprocessor":
        from .catalog_preprocessor import CatalogPreprocessor

        return CatalogPreprocessor
    if name == "BM25Index":
        from .bm25_index import BM25Index

        return BM25Index
    if name == "HierarchyFilter":
        from .hierarchy_filter import HierarchyFilter

        return HierarchyFilter
    if name == "ScoreFusion":
        from .scoring import ScoreFusion

        return ScoreFusion
    if name == "MatchPipeline":
        from .matching_pipeline import MatchPipeline

        return MatchPipeline
    if name == "evaluate_matches":
        from .matching_evaluator import evaluate_matches

        return evaluate_matches
    if name == "evaluate_matches_parallel":
        from .matching_evaluator import evaluate_matches_parallel

        return evaluate_matches_parallel
    raise AttributeError(name)
