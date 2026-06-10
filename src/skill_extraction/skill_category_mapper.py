"""技能类别映射模块。

本模块提供从 `skill_type` 到标准 8 类的映射功能，包括：
1. 基于规则的确定性映射（覆盖约 100 种已知 skill_type 值）
2. 对规则未命中的技能通过 LLM 进行分类
3. 批量为技能词典增加 `category` 字段

8 个标准类别：
- programming_language: 编程语言
- framework: 框架
- database: 数据库
- tool: 工具软件
- office: 办公软件
- equipment: 设备/仪器
- process: 工艺方法
- certification: 证书/资质
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 8 个合法 category 值
VALID_CATEGORIES: frozenset[str] = frozenset({
    "programming_language",
    "framework",
    "database",
    "tool",
    "office",
    "equipment",
    "process",
    "certification",
})

# 默认规则文件路径（相对于项目根目录）
_DEFAULT_RULES_RELATIVE = Path("dicts/skill_category_rules.json")


def _project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parents[2]


def _default_rules_path() -> Path:
    """返回默认映射规则文件路径。"""
    return _project_root() / _DEFAULT_RULES_RELATIVE


def load_category_rules(rules_path: str | Path | None = None) -> dict[str, Any]:
    """加载技能类别映射规则文件。

    参数:
        rules_path: 规则文件路径，为空时使用默认路径 dicts/skill_category_rules.json。

    返回:
        dict: 包含 'mapping_rules'、'categories' 和 'llm_classification_prompt' 的字典。

    异常:
        FileNotFoundError: 规则文件不存在。
        ValueError: JSON 解析失败或缺少必要字段。
    """
    path = Path(rules_path) if rules_path else _default_rules_path()
    if not path.exists():
        raise FileNotFoundError(f"映射规则文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "mapping_rules" not in data:
        raise ValueError(f"规则文件缺少 'mapping_rules' 字段: {path}")
    if "categories" not in data:
        raise ValueError(f"规则文件缺少 'categories' 字段: {path}")

    return data


def map_skill_type(
    skill_type: str,
    rules: dict[str, str] | None = None,
    rules_data: dict[str, Any] | None = None,
) -> str | None:
    """将单个 skill_type 值映射到标准类别。

    参数:
        skill_type: 原始 skill_type 值。
        rules: 映射规则字典（skill_type -> category），与 rules_data 二选一。
        rules_data: 完整规则数据（含 mapping_rules 字段），与 rules 二选一。

    返回:
        str | None: 标准类别名称，规则未命中时返回 None。
        返回 "needs_llm" 表示需要 LLM 进一步分类。
    """
    if rules is None:
        if rules_data is None:
            rules_data = load_category_rules()
        rules = rules_data.get("mapping_rules", {})

    category = rules.get(skill_type)
    if category == "needs_llm":
        return None
    return category


def _build_llm_classification_prompt(
    skill_names: list[str],
    categories: dict[str, Any],
    prompt_config: dict[str, Any],
) -> tuple[str, str]:
    """构建 LLM 分类的 system_prompt 和 user_prompt。

    参数:
        skill_names: 待分类的技能名称列表。
        categories: 类别定义字典。
        prompt_config: prompt 模板配置。

    返回:
        tuple[str, str]: (system_prompt, user_prompt)
    """
    system_prompt = prompt_config.get("system_prompt", "")

    # 构建技能列表文本
    skills_text = "\n".join(f"- {name}" for name in skill_names)

    user_prompt = (
        f"请对以下 {len(skill_names)} 个技能进行分类。\n\n"
        f"技能列表：\n{skills_text}\n\n"
        f"请返回一个 JSON 数组，每个元素格式为 {{\"skill\": \"技能名\", \"category\": \"类别英文标识\"}}。\n"
        f"类别必须是以下之一：{', '.join(sorted(VALID_CATEGORIES))}\n"
        f"只返回 JSON 数组，不要返回其他内容。"
    )

    return system_prompt, user_prompt


def classify_batch_by_llm(
    skill_names: list[str],
    rules_data: dict[str, Any] | None = None,
    llm_client: Any = None,
    batch_size: int = 50,
) -> dict[str, str]:
    """通过 LLM 对未映射的技能进行批量分类。

    参数:
        skill_names: 待分类的技能名称列表。
        rules_data: 完整规则数据，用于获取类别定义和 prompt 模板。
        llm_client: LLM client 实例，为空时通过 create_llm_client() 创建。
        batch_size: 每批处理的技能数量，默认 50。

    返回:
        dict[str, str]: 技能名称 -> 类别英文标识的映射。
        LLM 返回无效类别时该技能会被跳过。
    """
    if not skill_names:
        return {}

    if rules_data is None:
        rules_data = load_category_rules()

    categories = rules_data.get("categories", {})
    prompt_config = rules_data.get("llm_classification_prompt", {})

    if llm_client is None:
        from src.model_platform.llm import create_llm_client

        llm_client = create_llm_client()

    results: dict[str, str] = {}

    # 分批处理
    for i in range(0, len(skill_names), batch_size):
        batch = skill_names[i : i + batch_size]
        system_prompt, user_prompt = _build_llm_classification_prompt(
            batch, categories, prompt_config
        )

        try:
            parsed = llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "skill" in item and "category" in item:
                        skill = item["skill"]
                        category = item["category"]
                        if category in VALID_CATEGORIES:
                            results[skill] = category
                        else:
                            logger.warning(
                                "LLM 返回无效类别 '%s'（技能: %s），已跳过",
                                category,
                                skill,
                            )
            elif isinstance(parsed, dict):
                # 兼容单个技能时返回对象的情况
                if "skill" in parsed and "category" in parsed:
                    skill = parsed["skill"]
                    category = parsed["category"]
                    if category in VALID_CATEGORIES:
                        results[skill] = category

        except Exception as exc:
            logger.error("LLM 分类批次失败 (batch %d): %s", i // batch_size, exc)
            # 尝试逐个重试
            for skill_name in batch:
                try:
                    single_prompt = prompt_config.get("user_prompt_template", "").format(
                        skill_name=skill_name, context=""
                    )
                    single_result = llm_client.complete_json(
                        system_prompt=system_prompt,
                        user_prompt=single_prompt,
                    )
                    if isinstance(single_result, dict) and "category" in single_result:
                        category = single_result["category"]
                        if category in VALID_CATEGORIES:
                            results[skill_name] = category
                except Exception as single_exc:
                    logger.error("单个技能分类失败 (%s): %s", skill_name, single_exc)

    return results


def apply_categories_to_dictionary(
    dict_path: str | Path | None = None,
    output_path: str | Path | None = None,
    rules_data: dict[str, Any] | None = None,
    llm_client: Any = None,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """为技能词典中每个技能增加 category 字段。

    处理流程：
    1. 加载映射规则
    2. 对每个技能的 skill_type 尝试规则映射
    3. 对规则未命中的技能（或 skill_type 为 "专业知识"），通过 LLM 分类
    4. 在每个技能对象中增加 "category" 字段
    5. 返回更新后的词典数据

    参数:
        dict_path: 词典文件路径，为空时使用默认路径。
        output_path: 输出文件路径，为空时不写文件。
        rules_data: 完整规则数据，为空时自动加载。
        llm_client: LLM client 实例，为空时自动创建。
        skip_llm: 是否跳过 LLM 分类（未映射技能的 category 设为 None）。

    返回:
        dict: 更新后的词典数据，包含 category 字段。
    """
    from config.paths import get_project_paths

    if dict_path is None:
        paths = get_project_paths()
        dict_path = paths.dict_dir / "flat_skill_dictionary.json"
    dict_path = Path(dict_path)

    if rules_data is None:
        rules_data = load_category_rules()

    mapping_rules = rules_data.get("mapping_rules", {})

    with open(dict_path, "r", encoding="utf-8") as f:
        dict_data = json.load(f)

    skills = dict_data.get("skills", [])
    unmapped_skills: list[str] = []

    # 第一轮：规则映射
    for skill in skills:
        skill_type = skill.get("skill_type", "")
        category = map_skill_type(skill_type, rules=mapping_rules)
        if category is not None:
            skill["category"] = category
        else:
            # 标记为待 LLM 分类
            if not skip_llm:
                unmapped_skills.append(skill["name"])
            else:
                skill["category"] = None

    # 第二轮：LLM 分类
    if unmapped_skills and not skip_llm:
        llm_results = classify_batch_by_llm(
            unmapped_skills,
            rules_data=rules_data,
            llm_client=llm_client,
        )
        for skill in skills:
            if skill.get("category") is None and skill["name"] in llm_results:
                skill["category"] = llm_results[skill["name"]]
            elif skill.get("category") is None:
                skill["category"] = None

    # 统计结果
    category_counts: dict[str | None, int] = {}
    for skill in skills:
        cat = skill.get("category")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("类别分布: %s", category_counts)

    # 写入输出文件
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, ensure_ascii=False, indent=2)
        logger.info("已写入分类结果到: %s", output_path)

    return dict_data


def get_category_definitions(
    rules_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """获取 8 个标准类别的定义信息。

    参数:
        rules_data: 完整规则数据，为空时自动加载。

    返回:
        dict: 类别英文标识 -> {name_zh, description, examples} 的映射。
    """
    if rules_data is None:
        rules_data = load_category_rules()
    return rules_data.get("categories", {})
