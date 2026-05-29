"""技能抽取模块。"""

__all__ = [
    "OccupationBGEMatcher",
    "OccupationSkillDictionaryStore",
    "OccupationSkillPipeline",
    "FlatSkillPipeline",
]


def __getattr__(name: str):
    if name == "OccupationBGEMatcher":
        from .bge_matcher import OccupationBGEMatcher

        return OccupationBGEMatcher
    if name == "OccupationSkillDictionaryStore":
        from .history.dictionary_store import OccupationSkillDictionaryStore

        return OccupationSkillDictionaryStore
    if name == "OccupationSkillPipeline":
        from .history.occupation_skill_pipeline import OccupationSkillPipeline

        return OccupationSkillPipeline
    if name == "FlatSkillPipeline":
        from .occupation_skill_pipeline import FlatSkillPipeline

        return FlatSkillPipeline
    raise AttributeError(name)
