"""
技能提取模块初始化文件
"""

from .word2vec_extractor import Word2VecSkillExtractor
from .bert_extractor import BERTSkillExtractor
from .run_extraction_pipeline import SkillExtractionPipeline

__all__ = [
    'Word2VecSkillExtractor',
    'BERTSkillExtractor',
    'SkillExtractionPipeline'
]

