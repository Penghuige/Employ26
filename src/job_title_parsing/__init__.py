"""岗位名称解析与职业匹配模块。"""

__all__ = [
    "OccupationDictManager",
    "OccupationParser",
    "CatalogPreprocessor",
    "BM25Index",
    "HierarchyFilter",
    "ScoreFusion",
    "MatchPipeline",
    "evaluate_matches",
]


def __getattr__(name: str):
    if name == "OccupationDictManager":
        from .occupation_dict_manager import OccupationDictManager

        return OccupationDictManager
    if name == "OccupationParser":
        from .occupation_parser import OccupationParser

        return OccupationParser
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
    raise AttributeError(name)
