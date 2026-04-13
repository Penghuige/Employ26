"""
职业技能词典构建主流程。

本模块包含两套流水线：

v2（推荐） — FlatSkillPipeline:
    按「职业中类」采样，使用本地 vLLM (Qwen3-8B) 直接批量推理，
    输出平面化技能列表，用于正则匹配岗位需求文本中的硬技能。
    1. 每个职业中类抽取 100 条训练样本
    2. vLLM 批量推理提取硬技能 → 合并为平面化词典
    3. 每个中类再抽 10 条验证样本，使用 LLM 评估覆盖率
    4. 覆盖率不足时自动补充缺失技能
    5. 最终去重，确保 name 和 aliases 全局唯一

v1（Legacy） — OccupationSkillPipeline:
    按「职业细类」采样，生成 prompt 文件交由外部 LLM 处理，
    结果手动导入。保留供参考和向后兼容。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import SkillExtractionConfig, load_skill_extraction_config
from .coverage import RequirementCoverageEvaluator
from .data_source import OccupationSampleBuilder
from .dictionary_store import OccupationSkillDictionaryStore


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _now_text() -> str:
    """返回当前时间文本。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _build_training_prompt(detail_row: Dict, sample_texts: List[str]) -> str:
    """构造训练阶段给 LLM 的提示词。"""
    examples = [f"{index}. {text}" for index, text in enumerate(sample_texts, start=1)]
    samples_block = "\n".join(examples) if examples else "无可用样本"

    return f"""你是一名招聘技能词典整理助手。

任务：
请根据下面"职业细类"的任职要求样本，整理该职业细类的技能词典。

职业细类信息：
- 大类：{detail_row["大类"]}
- 中类：{detail_row["中类"]}
- 小类：{detail_row["小类"]}
- 细类：{detail_row["细类"]}
- 层级路径：{detail_row["detail_path"]}

抽取要求：
1. 只保留可标准化、可复用、可统计的技能词。
2. 排除软素质、人格特质、福利待遇、年龄学历年限、岗位名称、空泛职责动词。
3. 同义词、缩写、大小写变体请收敛到同一个标准技能词下。
4. 尽量输出高精度词典，不要为了召回率加入模糊词。
5. 若某技能明显属于"编程语言/框架/数据库/工具/办公软件/证书/行业工具"等，请写明 skill_type。

输出格式：
请只输出 JSON，不要输出解释文字。

{{
  "detail_path": "{detail_row["detail_path"]}",
  "detail_name": "{detail_row["detail_name"]}",
  "skills": [
    {{
      "name": "标准技能词",
      "aliases": ["别名1", "别名2"],
      "skill_type": "技能类别",
      "notes": "可选，必要时说明边界"
    }}
  ]
}}

训练样本（已尽量裁剪到任职要求，减少 token）：
{samples_block}
"""


def _build_supplement_prompt(detail_row: Dict, current_skills: List[Dict], uncovered_items: List[str]) -> str:
    """构造补词阶段给 LLM 的提示词。"""
    current_skill_lines = []
    for skill in current_skills:
        aliases = skill.get("aliases", []) or []
        alias_text = f"（aliases: {', '.join(map(str, aliases))}）" if aliases else ""
        current_skill_lines.append(f'- {skill.get("name", "")} {alias_text}'.strip())

    uncovered_lines = [f"{index}. {item}" for index, item in enumerate(uncovered_items, start=1)]
    current_skill_block = "\n".join(current_skill_lines) if current_skill_lines else "当前词典为空"
    uncovered_block = "\n".join(uncovered_lines) if uncovered_lines else "本轮没有未覆盖条目"

    return f"""你正在补充一个职业细类的技能词典。

职业细类：
- 层级路径：{detail_row["detail_path"]}
- 细类：{detail_row["detail_name"]}

当前词典：
{current_skill_block}

下面这些任职要求条目在验证集中尚未被词典覆盖。
请只补充真正缺失的技能词，不要重复已有词，不要加入软素质、学历年限、福利待遇或岗位名称。

未覆盖条目：
{uncovered_block}

请只输出 JSON：
{{
  "detail_path": "{detail_row["detail_path"]}",
  "missing_skills": [
    {{
      "name": "标准技能词",
      "aliases": ["别名1", "别名2"],
      "skill_type": "技能类别",
      "notes": "可选说明"
    }}
  ]
}}
"""


class OccupationSkillPipeline:
    """(Legacy / v1) 职业细类技能词典流程控制器。

    本类按「职业细类」粒度采样，生成 prompt 文件交由外部 LLM 处理，
    结果手动导入后合并回词典。保留供参考和向后兼容。

    新项目请使用 ``FlatSkillPipeline``（v2），它按「职业中类」采样，
    使用本地 vLLM 直接批量推理，输出平面化技能列表。
    """

    def __init__(self, config: SkillExtractionConfig):
        self.config = config
        self._builder: OccupationSampleBuilder | None = None
        self.dictionary_store = OccupationSkillDictionaryStore(config.dictionary_path)
        self.coverage_evaluator = RequirementCoverageEvaluator(self.dictionary_store)

    @property
    def builder(self) -> OccupationSampleBuilder:
        """按需初始化数据构建器，避免无关命令加载 BGE 模型。"""
        if self._builder is None:
            self._builder = OccupationSampleBuilder(self.config)
        return self._builder

    def prepare(
        self,
        train_size: int,
        validation_batch_size: int,
        seed: int,
        limit_job_rows: int | None,
        limit_categories: int | None,
        match_workers: int,
        match_chunk_size: int,
        parse_workers: int,
        show_progress: bool,
    ) -> None:
        """执行采样、任职要求切割与训练 prompt 生成。"""
        logger.info("开始准备职业细类技能词典训练数据")

        # 这一步会先切出"任职要求"，再使用 BGE 对职业细类做语义匹配，
        # 并把完整结果落到 DuckDB 的 skill_extraction.requirement_match_table。
        from src.preprocessing.prepare_skill_extraction_requirement_matches import (
            prepare_requirement_matches_to_duckdb,
        )

        del match_workers
        del match_chunk_size
        del show_progress

        matched_df = prepare_requirement_matches_to_duckdb(
            config=self.config,
            limit_job_rows=limit_job_rows,
            parse_workers=max(1, parse_workers),
            parse_batch_size=2000,
            top_k=self.config.match_top_k,
        )

        category_summary_df, manifest_df = self.builder.build_sampling_manifests(
            matched_df=matched_df,
            train_size=train_size,
            seed=seed,
            limit_categories=limit_categories,
        )

        train_manifest_df = manifest_df[manifest_df["split"] == "train"].copy()
        validation_pool_df = manifest_df[manifest_df["split"] == "validation_pool"].copy()

        category_summary_df.to_csv(self.config.category_summary_path, index=False, encoding="utf-8-sig")
        train_manifest_df.to_csv(self.config.training_manifest_path, index=False, encoding="utf-8-sig")
        validation_pool_df.to_csv(self.config.validation_pool_path, index=False, encoding="utf-8-sig")

        logger.info("开始整理训练样本中的任职要求与职业匹配结果")
        train_rows_df = matched_df.merge(
            train_manifest_df[
                [
                    "sample_row_id",
                    "prompt_file_key",
                    "sample_order",
                ]
            ],
            on="sample_row_id",
            how="inner",
        ).sort_values(["detail_name", "sample_order"], ascending=[True, True])

        parsed_train_df = train_rows_df.copy()
        parsed_train_df["需求文本"] = parsed_train_df["任职要求_items_text"].where(
            parsed_train_df["任职要求_items_text"].fillna("").astype(str).str.strip() != "",
            parsed_train_df["职业匹配文本"],
        )
        parsed_train_df.to_csv(self.config.training_requirements_path, index=False, encoding="utf-8-sig")

        dictionary = self.dictionary_store.ensure_categories(category_summary_df)
        self._write_training_prompts(parsed_train_df)
        self._initialize_state(category_summary_df, validation_batch_size=validation_batch_size, seed=seed)

        logger.info("训练数据准备完成")
        logger.info("训练清单: %s", self.config.training_manifest_path)
        logger.info("训练需求文本: %s", self.config.training_requirements_path)
        logger.info("验证池: %s", self.config.validation_pool_path)
        logger.info("职业技能词典: %s", self.config.dictionary_path)
        logger.info("已初始化职业细类数量: %s", len(dictionary.get("categories", {})))

    def iterate(
        self,
        validation_batch_size: int,
        coverage_threshold: float,
        limit_categories: int | None,
        parse_workers: int,
    ) -> None:
        """执行一轮覆盖率验证，并为未达标细类生成补词 prompt。"""
        validation_pool_df = pd.read_csv(self.config.validation_pool_path, encoding="utf-8")
        category_summary_df = pd.read_csv(self.config.category_summary_path, encoding="utf-8")
        state = self._load_state(validation_batch_size=validation_batch_size)
        dictionary = self.dictionary_store.load()

        pending_categories = self._pending_categories(validation_pool_df, state)
        if limit_categories is not None:
            pending_categories = pending_categories[: int(limit_categories)]

        if not pending_categories:
            logger.info("没有待验证的职业细类，流程结束")
            return

        round_no = int(state.get("global_round", 0)) + 1
        logger.info("开始第 %s 轮覆盖率验证，共 %s 个职业细类", round_no, len(pending_categories))

        round_manifest_frames: List[pd.DataFrame] = []
        for detail_path in pending_categories:
            used_count = int(state["categories"][detail_path]["used_validation_count"])
            detail_pool_df = validation_pool_df[validation_pool_df["detail_path"] == detail_path].sort_values(
                "sample_order"
            )
            batch_df = detail_pool_df.iloc[used_count: used_count + validation_batch_size].copy()
            if batch_df.empty:
                state["categories"][detail_path]["status"] = "exhausted"
                continue
            round_manifest_frames.append(batch_df)

        if not round_manifest_frames:
            self._save_state(state)
            logger.info("没有可继续抽取的验证样本")
            return

        round_manifest_df = pd.concat(round_manifest_frames, axis=0, ignore_index=True)
        round_rows_df = self.builder.fetch_rows_by_manifest(round_manifest_df)
        from src.preprocessing.parse_desc import parse_desc_df

        parsed_validation_df = parse_desc_df(
            round_rows_df[["岗位名称", "岗位描述"]].copy(),
            desc_col="岗位描述",
            batch_size=1000,
            num_workers=max(1, parse_workers),
        )
        parsed_validation_df = pd.concat(
            [
                round_rows_df[
                    [
                        "sample_row_id",
                        "detail_path",
                        "detail_name",
                        "大类",
                        "中类",
                        "小类",
                        "细类",
                        "prompt_file_key",
                        "sample_order",
                    ]
                ].reset_index(drop=True),
                parsed_validation_df.reset_index(drop=True),
            ],
            axis=1,
        )
        parsed_validation_df["需求文本"] = parsed_validation_df["任职要求_items_text"].where(
            parsed_validation_df["任职要求_items_text"].fillna("").astype(str).str.strip() != "",
            parsed_validation_df["RAG匹配文本"],
        )

        summary_df, item_df, uncovered_df = self.coverage_evaluator.evaluate_batch(parsed_validation_df, dictionary)
        summary_df["round_no"] = round_no
        summary_df["coverage_threshold"] = coverage_threshold
        summary_df["passed"] = summary_df["coverage"] >= float(coverage_threshold)

        round_dir = self.config.report_dir / f"round_{round_no:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        validation_csv_path = round_dir / "validation_samples.csv"
        summary_csv_path = round_dir / "coverage_summary.csv"
        item_csv_path = round_dir / "coverage_items.csv"
        uncovered_csv_path = round_dir / "uncovered_items.csv"

        parsed_validation_df.to_csv(validation_csv_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        item_df.to_csv(item_csv_path, index=False, encoding="utf-8-sig")
        uncovered_df.to_csv(uncovered_csv_path, index=False, encoding="utf-8-sig")

        category_lookup = category_summary_df.set_index("detail_path").to_dict(orient="index")
        self._write_supplement_prompts(
            round_no=round_no,
            summary_df=summary_df,
            uncovered_df=uncovered_df,
            dictionary=dictionary,
            category_lookup=category_lookup,
        )
        self._update_state_after_round(
            state=state,
            round_no=round_no,
            summary_df=summary_df,
            validation_batch_size=validation_batch_size,
        )
        self._save_state(state)

        logger.info("第 %s 轮验证完成", round_no)
        logger.info("覆盖率汇总: %s", summary_csv_path)
        logger.info("未覆盖条目: %s", uncovered_csv_path)

    def status(self) -> None:
        """输出当前迭代状态。"""
        state = self._load_state(validation_batch_size=10)
        summary_rows: List[Dict] = []
        for detail_path, info in state.get("categories", {}).items():
            summary_rows.append(
                {
                    "detail_path": detail_path,
                    "status": info.get("status", "pending"),
                    "used_validation_count": info.get("used_validation_count", 0),
                    "last_round": info.get("last_round", 0),
                    "last_coverage": info.get("last_coverage", ""),
                }
            )
        status_df = pd.DataFrame(summary_rows)
        if status_df.empty:
            logger.info("当前还没有技能词典迭代状态文件")
            return
        logger.info("\n%s", status_df.sort_values(["status", "detail_path"]).to_string(index=False))

    def import_llm_results(self, source_path: str | Path, recursive: bool = True, dry_run: bool = False) -> None:
        """把 LLM 输出的 JSON 合并回职业技能词典。"""
        import_stats = self.dictionary_store.import_from_path(
            source_path=source_path,
            recursive=recursive,
            dry_run=dry_run,
        )

        report_name = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.config.report_dir / report_name
        report_path.write_text(
            json.dumps(import_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("LLM 结果导入完成")
        logger.info("导入源: %s", Path(source_path))
        logger.info("处理文件数: %s", import_stats["processed_files"])
        logger.info("载入 payload 数: %s", import_stats["loaded_payloads"])
        logger.info("更新细类数: %s", import_stats["updated_categories"])
        logger.info("新增技能数: %s", import_stats["created_skills"])
        logger.info("合并技能数: %s", import_stats["merged_skills"])
        logger.info("跳过 payload 数: %s", import_stats["skipped_payloads"])
        logger.info("导入报告: %s", report_path)
        if dry_run:
            logger.info("当前为 dry-run，词典未写回磁盘")

    def _write_training_prompts(self, parsed_train_df: pd.DataFrame) -> None:
        """写出训练阶段 prompt。"""
        for detail_path, group in parsed_train_df.groupby("detail_path", sort=False):
            prompt_file_key = str(group["prompt_file_key"].iloc[0])
            sample_texts = [
                text
                for text in group["需求文本"].fillna("").astype(str).tolist()
                if str(text).strip()
            ]
            prompt_text = _build_training_prompt(
                {
                    "detail_path": detail_path,
                    "detail_name": group["detail_name"].iloc[0],
                    "大类": group["大类"].iloc[0],
                    "中类": group["中类"].iloc[0],
                    "小类": group["小类"].iloc[0],
                    "细类": group["细类"].iloc[0],
                },
                sample_texts=sample_texts,
            )
            prompt_path = self.config.prompt_train_dir / f"{prompt_file_key}.md"
            prompt_path.write_text(prompt_text, encoding="utf-8")

    def _write_supplement_prompts(
        self,
        round_no: int,
        summary_df: pd.DataFrame,
        uncovered_df: pd.DataFrame,
        dictionary: Dict,
        category_lookup: Dict[str, Dict],
    ) -> None:
        """为未达标职业细类写出补词 prompt。"""
        round_prompt_dir = self.config.prompt_supplement_dir / f"round_{round_no:02d}"
        round_prompt_dir.mkdir(parents=True, exist_ok=True)

        failed_detail_paths = set(
            summary_df.loc[summary_df["passed"] == False, "detail_path"].astype(str).tolist()  # noqa: E712
        )
        for detail_path in failed_detail_paths:
            category_info = category_lookup.get(detail_path, {})
            current_skills = self.dictionary_store.get_skills(dictionary, detail_path)
            uncovered_items = (
                uncovered_df.loc[uncovered_df["detail_path"] == detail_path, "未覆盖要求条目"]
                .dropna()
                .astype(str)
                .tolist()
            )
            prompt_text = _build_supplement_prompt(
                detail_row={
                    "detail_path": detail_path,
                    "detail_name": category_info.get("detail_name", detail_path),
                },
                current_skills=current_skills,
                uncovered_items=uncovered_items,
            )
            prompt_file_key = str(category_info.get("prompt_file_key", "detail"))
            prompt_path = round_prompt_dir / f"{prompt_file_key}.md"
            prompt_path.write_text(prompt_text, encoding="utf-8")

    def _initialize_state(self, category_summary_df: pd.DataFrame, validation_batch_size: int, seed: int) -> None:
        """初始化迭代状态文件。"""
        state = {
            "metadata": {
                "created_at": _now_text(),
                "validation_batch_size": int(validation_batch_size),
                "seed": int(seed),
            },
            "global_round": 0,
            "categories": {},
        }
        for row in category_summary_df.to_dict(orient="records"):
            state["categories"][row["detail_path"]] = {
                "detail_name": row["detail_name"],
                "status": "pending" if int(row.get("validation_pool_count", 0)) > 0 else "exhausted",
                "validation_pool_count": int(row.get("validation_pool_count", 0)),
                "used_validation_count": 0,
                "last_round": 0,
                "last_coverage": None,
                "history": [],
            }
        self._save_state(state)

    def _load_state(self, validation_batch_size: int) -> Dict:
        """读取状态文件；不存在时返回空骨架。"""
        if not self.config.state_path.exists():
            return {
                "metadata": {
                    "created_at": _now_text(),
                    "validation_batch_size": int(validation_batch_size),
                },
                "global_round": 0,
                "categories": {},
            }
        with open(self.config.state_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _save_state(self, state: Dict) -> None:
        """保存状态文件。"""
        with open(self.config.state_path, "w", encoding="utf-8") as file_obj:
            json.dump(state, file_obj, ensure_ascii=False, indent=2)

    @staticmethod
    def _pending_categories(validation_pool_df: pd.DataFrame, state: Dict) -> List[str]:
        """读取仍需验证的职业细类列表。"""
        pending: List[str] = []
        categories_state = state.get("categories", {})
        for detail_path in validation_pool_df["detail_path"].dropna().astype(str).unique().tolist():
            category_state = categories_state.get(detail_path, {})
            if category_state.get("status") == "passed":
                continue
            if category_state.get("status") == "exhausted":
                continue
            pending.append(detail_path)
        return pending

    @staticmethod
    def _update_state_after_round(
        state: Dict,
        round_no: int,
        summary_df: pd.DataFrame,
        validation_batch_size: int,
    ) -> None:
        """更新某一轮验证后的状态。"""
        state["global_round"] = round_no
        for row in summary_df.to_dict(orient="records"):
            detail_path = row["detail_path"]
            category_state = state["categories"].setdefault(detail_path, {})
            category_state["used_validation_count"] = int(category_state.get("used_validation_count", 0)) + int(
                row["validation_sample_count"]
            )
            category_state["last_round"] = round_no
            category_state["last_coverage"] = float(row["coverage"])
            category_state["status"] = "passed" if bool(row["passed"]) else "pending"
            category_state.setdefault("history", []).append(
                {
                    "round_no": round_no,
                    "coverage": float(row["coverage"]),
                    "validation_sample_count": int(row["validation_sample_count"]),
                    "skill_item_count": int(row["skill_item_count"]),
                    "covered_skill_item_count": int(row["covered_skill_item_count"]),
                }
            )
            validation_pool_count = int(category_state.get("validation_pool_count", 0))
            if (
                not bool(row["passed"])
                and category_state["used_validation_count"] >= validation_pool_count
            ):
                category_state["status"] = "exhausted"
            elif row["validation_sample_count"] < int(validation_batch_size) and not bool(row["passed"]):
                category_state["status"] = "exhausted"


# ──────────────────────────────────────────────────────────────────────
#  v2 — FlatSkillPipeline  平面化技能词典流水线 (vLLM + Qwen3-8B)
# ──────────────────────────────────────────────────────────────────────

FLAT_TRAINING_SYSTEM_PROMPT = """\
你是一名招聘硬技能提取专家。你需要从岗位任职要求样本中提取可标准化、可复用的硬技能词。

## 提取规则

1. **只保留硬技能**：工具、软件、编程语言、框架、数据库、证书、行业方法、设备等。
2. **排除以下内容**：
   - 软素质（沟通能力、团队协作、责任心……）
   - 人格特质（积极主动、细心认真……）
   - 学历/年限/年龄要求
   - 福利待遇、薪资条件
   - 岗位名称、空泛职责动词（如"负责""处理"）
3. **同义词收敛**：同一技能的不同写法请收敛为一个标准名称，其余写法放入 aliases。
   - 例如 "PS" / "Photoshop" / "Adobe Photoshop" → name: "Photoshop", aliases: ["PS", "Adobe Photoshop"]
4. **高精度优先**：宁可漏掉边缘技能，也不要加入模糊、不确定的词。
5. **标注 skill_type**：请标注技能类别（编程语言、框架、数据库、工具软件、办公软件、\
证书/资质、专业知识、工艺/方法、设备/仪器 等）。
6. **不要输出技能容器词**：不要输出“测试”“仿真软件”“数据分析工具”“办公软件”“资格证书”“执业资格证书”“专业知识”\
这类过泛名称；必须尽量输出可直接落地的具体技能名、工具名、框架名、设备名或具体证书名。
7. **证书必须具体**：可以输出“CPA”“PMP”“医师资格证”“教师资格证”等具体证书，\
但不要输出“证书”“资格证”“执业资格证书”这种泛称。

## 输出格式

请 **只输出 JSON**，不要输出任何解释或 markdown 标记。

{"skills":[{"name":"标准技能名","aliases":["别名1","别名2"],"skill_type":"技能类别","notes":""}]}
"""

FLAT_TRAINING_USER_TEMPLATE = """\
职业中类: {category_name}

以下是该职业中类的 {count} 条岗位任职要求样本，请从中提取硬技能：

{samples_text}
"""

FLAT_EVALUATION_SYSTEM_PROMPT = """\
你是一名技能词典覆盖率评估专家。你需要判断现有技能词典是否能覆盖验证样本中提到的硬技能。

## 评估规则

1. 逐条检查验证样本中的硬技能是否已存在于词典中（匹配 name 或 aliases）。
2. 忽略软素质、学历年限、福利待遇等非硬技能项。
3. 只关注"精准匹配"——如果样本中出现的技能名在词典的 name 或 aliases 里，算覆盖。
4. 若发现未覆盖的硬技能，请提取出来作为缺失技能。
5. 不要把“测试”“仿真软件”“办公软件”“资格证书”“执业资格证书”“专业知识”\
这类泛称当作有效硬技能；如果文本里提到的是具体工具、框架、设备或具体证书，请优先给出具体名称。

## 输出格式

请 **只输出 JSON**，不要输出任何解释或 markdown 标记。

{"covered_count":N,"total_hard_skill_items":N,"accuracy":0.85,\
"missing_skills":[{"name":"缺失技能名","aliases":[],"skill_type":"类别","notes":""}]}
"""

FLAT_EVALUATION_USER_TEMPLATE = """\
## 当前技能词典（共 {skill_count} 项）

{skill_summary}

## 验证样本（共 {sample_count} 条任职要求）

{validation_text}

请评估上述词典对验证样本中硬技能的覆盖率，并提取未覆盖的硬技能。
"""

# 每条 prompt 最多包含的样本数（避免超出 vLLM 上下文窗口）
MAX_SAMPLES_PER_PROMPT: int = 15

# 每条样本文本的最大字符数（截断过长的任职要求，避免 prompt 超长）
MAX_SAMPLE_CHARS: int = 300

# 默认覆盖率阈值
DEFAULT_COVERAGE_THRESHOLD: float = 0.80

# 这些模式用于过滤“看起来像技能、实际上只是容器词/泛化概念”的条目。
# 目标是从词典源头减少会在匹配阶段制造系统性误报的技能项。
LOW_VALUE_SKILL_PATTERNS: List[str] = [
    r"(能力|素养|基础|知识|理论)$",
    r"(工具|软件|系统|平台)$",
    r"^(测试|检测|英语|普通话|电源)$",
    r"^(数据分析工具|仿真软件|办公软件|测试仪器|专业知识|理论基础|知识基础)$",
    r"^(资格证|资格证书|执业资格证书|上岗证|证书)$",
]

LOW_VALUE_ALIAS_PATTERNS: List[str] = [
    r"^(资格|资格证|资格证书|证书|执业证|上岗证|许可证|执照)$",
    r"^(工具|软件|系统|平台|知识|理论|能力|测试)$",
    r".*(资格证|资格证书|执业证|执业资格证书)$",
]

# 默认 Qwen3-8B 模型路径
DEFAULT_MODEL_PATH: str = "D:/model/Qwen3-8B"


class FlatSkillPipeline:
    """v2 平面化技能词典构建流水线。

    按「职业中类」采样岗位任职要求文本，使用本地 vLLM (Qwen3-8B)
    批量推理提取硬技能，输出全局平面化的技能列表。

    流水线步骤:
        1. 加载岗位数据并按职业中类分组采样（训练 100 + 验证 10）
        2. 对每个中类的训练样本，使用 vLLM 批量推理提取硬技能
        3. 合并所有中类的技能，按名称去重
        4. 对每个中类的验证样本，使用 vLLM 评估覆盖率
        5. 覆盖率不足时自动提取缺失技能并补充
        6. 最终全局去重（name + aliases 互不冲突）
        7. 保存为平面化 JSON 词典

    参数:
        config (SkillExtractionConfig): 全局配置（数据库路径、输出目录等）。
        model_path (str): Qwen3-8B 模型的本地路径。
        gpu_memory_utilization (float): GPU 显存利用率，范围 (0, 1)。
        max_model_len (int): vLLM 最大序列长度（输入 + 输出 token 总和）。
        max_num_seqs (int): vLLM 最大并发序列数。

    示例::

        config = load_skill_extraction_config()
        pipeline = FlatSkillPipeline(config, model_path="D:/model/Qwen3-8B")
        pipeline.run(
            train_size=100,
            validation_size=10,
            coverage_threshold=0.80,
        )
    """

    def __init__(
        self,
        config: SkillExtractionConfig,
        model_path: str = DEFAULT_MODEL_PATH,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 8192,
        max_num_seqs: int = 48,
    ) -> None:
        """初始化流水线。

        参数:
            config: 全局配置对象，包含数据库路径、输出目录等。
            model_path: Qwen3-8B 模型目录路径（HuggingFace 格式）。
            gpu_memory_utilization: GPU 显存利用率。
                0.80 为推荐值（RTX 4090 24 GB），留约 4.8 GB 给系统。
            max_model_len: vLLM 最大上下文长度。
                8192 适用于本任务（prompt 约 500-2000 token，
                输出约 500-2048 token），留有余量。
            max_num_seqs: vLLM 最大并发序列数。
                48 为 RTX 4090 推荐值，平衡吞吐与显存。
        """
        self.config = config
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self._llm = None  # 延迟初始化
        self._tokenizer = None

    # ── vLLM 引擎管理 ────────────────────────────────────────────

    def _ensure_llm(self) -> None:
        """延迟初始化 vLLM 推理引擎。

        首次调用时从 ``merge_similar_skills`` 模块导入
        ``init_vllm_engine`` 并初始化 ``vllm.LLM`` 实例。
        后续调用直接复用已有实例，避免重复加载模型权重（约 15.3 GB）。

        副作用:
            - 设置 ``self._llm`` 为 ``vllm.LLM`` 实例。
            - 设置 ``self._tokenizer`` 为模型的 tokenizer。

        异常:
            SystemExit: 如果 GPU 显存不足以初始化 KV cache。
        """
        if self._llm is not None:
            return

        from .merge_similar_skills import init_vllm_engine

        logger.info("正在初始化 vLLM 引擎（模型: %s）...", self.model_path)
        self._llm = init_vllm_engine(
            model_path=self.model_path,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("vLLM 引擎初始化完成")

    def _vllm_batch_generate(
        self,
        prompt_pairs: List[Tuple[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> List[str]:
        """使用 vLLM 批量推理生成文本。

        接受 ``(system_prompt, user_prompt)`` 对列表，使用 tokenizer 的
        chat template 格式化为模型输入，然后一次性提交给 vLLM 引擎
        进行离线批量推理。

        参数:
            prompt_pairs: 每个元素为 ``(system_prompt, user_prompt)`` 的元组。
            max_tokens: 每条 prompt 的最大生成 token 数，默认 2048。
            temperature: 采样温度。0.1 为低温度，确保输出稳定和格式一致。

        返回:
            list[str]: 每条 prompt 对应的生成文本，长度与 ``prompt_pairs`` 相同。

        说明:
            vLLM 的离线批量推理模式（ ``LLM.generate()`` ）吞吐量最高，
            内部通过 continuous batching 和 PagedAttention 自动调度。
        """
        self._ensure_llm()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        # 使用 tokenizer chat template 格式化 prompt
        formatted_prompts: List[str] = []
        for system_prompt, user_prompt in prompt_pairs:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                prompt_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                # 回退：简单拼接
                prompt_text = (
                    f"system: {system_prompt}\n"
                    f"user: {user_prompt}\n"
                    f"assistant:"
                )
            formatted_prompts.append(prompt_text)

        logger.info("vLLM 批量推理: 共 %d 条 prompt", len(formatted_prompts))
        outputs = self._llm.generate(formatted_prompts, sampling_params)

        results: List[str] = []
        for output in outputs:
            results.append(output.outputs[0].text)

        logger.info("vLLM 批量推理完成")
        return results

    # ── 数据加载与采样 ───────────────────────────────────────────

    @staticmethod
    def _get_requirement_text(row: pd.Series) -> str:
        """从匹配结果行中提取需求文本。

        优先使用 ``任职要求_items_text``（已切分的任职要求），
        若为空则回退到 ``职业匹配文本``（语义匹配的原始文本）。

        参数:
            row: 匹配结果 DataFrame 的一行。

        返回:
            str: 非空的需求文本。如果两者均为空，返回空字符串。
        """
        text = str(row.get("任职要求_items_text", "") or "").strip()
        if text:
            return text
        return str(row.get("职业匹配文本", "") or "").strip()

    def _load_and_sample(
        self,
        train_size: int = 100,
        validation_size: int = 10,
        seed: int = 42,
        limit_job_rows: int | None = None,
        limit_categories: int | None = None,
        parse_workers: int = 1,
    ) -> Dict[str, Dict[str, List[str]]]:
        """加载岗位数据并按职业中类分组采样。

        执行流程:
            1. 调用 ``prepare_requirement_matches_to_duckdb`` 将岗位与职业分类匹配。
            2. 提取每行的需求文本（任职要求或匹配文本）。
            3. 按「中类」分组。
            4. 对每个中类随机采样 ``train_size`` 条训练文本和
               ``validation_size`` 条验证文本。

        参数:
            train_size: 每个中类的训练样本数，默认 100。
            validation_size: 每个中类的验证样本数，默认 10。
            seed: 随机种子，确保采样可复现。
            limit_job_rows: 调试用，限制读取的岗位行数。
            limit_categories: 调试用，限制处理的中类数量。
            parse_workers: 岗位描述解析的并发数。

        返回:
            dict[str, dict[str, list[str]]]: 以中类名称为键，值为::

                {
                    "train_texts": ["任职要求文本1", ...],   # 最多 train_size 条
                    "validation_texts": ["任职要求文本1", ...],  # 最多 validation_size 条
                }

        异常:
            FileNotFoundError: 数据库文件不存在时抛出。
        """
        from src.preprocessing.prepare_skill_extraction_requirement_matches import (
            prepare_requirement_matches_to_duckdb,
        )

        logger.info("正在加载岗位数据并匹配职业分类...")
        matched_df = prepare_requirement_matches_to_duckdb(
            config=self.config,
            limit_job_rows=limit_job_rows,
            parse_workers=max(1, parse_workers),
            parse_batch_size=2000,
            top_k=self.config.match_top_k,
        )

        # 提取需求文本
        matched_df["需求文本"] = matched_df.apply(self._get_requirement_text, axis=1)

        # 过滤空文本
        matched_df = matched_df[
            matched_df["需求文本"].str.strip().astype(bool)
        ].copy()

        logger.info("有效岗位行数: %d", len(matched_df))

        # 按中类分组采样
        import random

        category_samples: Dict[str, Dict[str, List[str]]] = {}
        grouped = matched_df.groupby("中类", sort=True)
        category_names = list(grouped.groups.keys())

        if limit_categories is not None:
            category_names = category_names[: int(limit_categories)]

        for category_name in category_names:
            group_df = grouped.get_group(category_name)
            texts = group_df["需求文本"].tolist()

            # 使用固定种子随机打乱
            rng = random.Random(seed)
            rng.shuffle(texts)

            total_needed = train_size + validation_size
            train_texts = texts[:train_size]
            validation_texts = texts[train_size:total_needed]

            if not train_texts:
                logger.warning("中类 '%s' 无可用训练样本，跳过", category_name)
                continue

            category_samples[category_name] = {
                "train_texts": train_texts,
                "validation_texts": validation_texts,
            }

        logger.info(
            "采样完成: %d 个中类, 训练样本 %d 条/中类, 验证样本 %d 条/中类",
            len(category_samples),
            train_size,
            validation_size,
        )
        return category_samples

    # ── 技能提取 ─────────────────────────────────────────────────

    def _extract_skills_from_all_categories(
        self,
        category_samples: Dict[str, Dict[str, List[str]]],
    ) -> List[Dict]:
        """对所有职业中类批量提取硬技能。

        遍历每个中类的训练样本，将样本按 ``MAX_SAMPLES_PER_PROMPT``
        分批构建 prompt，然后一次性提交给 vLLM 批量推理，最大化吞吐量。

        参数:
            category_samples: 来自 ``_load_and_sample`` 的采样结果，
                以中类名称为键，值包含 ``train_texts`` 和 ``validation_texts``。

        返回:
            list[dict]: 全局去重后的技能列表，每个元素包含::

                {
                    "name": "标准技能名",
                    "aliases": ["别名1", ...],
                    "skill_type": "技能类别",
                    "notes": ""
                }

        说明:
            为最大化 vLLM 吞吐量，本方法会先收集所有中类的 prompt，
            然后一次性提交批量推理，而非逐个中类推理。这充分利用了
            vLLM 的 continuous batching 和 PagedAttention 机制。
        """
        from .merge_similar_skills import extract_json_from_response

        logger.info("开始批量提取所有中类的硬技能...")

        # 收集所有 prompt（一次性提交，充分利用 vLLM continuous batching）
        all_prompt_pairs: List[Tuple[str, str]] = []
        # 记录每个中类对应的 prompt 索引范围: (category_name, start, end)
        prompt_category_map: List[Tuple[str, int, int]] = []

        for category_name, samples in category_samples.items():
            train_texts = samples["train_texts"]
            if not train_texts:
                continue

            start_idx = len(all_prompt_pairs)
            for chunk_start in range(0, len(train_texts), MAX_SAMPLES_PER_PROMPT):
                batch_texts = train_texts[
                    chunk_start : chunk_start + MAX_SAMPLES_PER_PROMPT
                ]
                samples_block = "\n".join(
                    f"{i + 1}. {text[:MAX_SAMPLE_CHARS]}"
                    for i, text in enumerate(batch_texts)
                )
                user_prompt = FLAT_TRAINING_USER_TEMPLATE.format(
                    category_name=category_name,
                    count=len(batch_texts),
                    samples_text=samples_block,
                )
                all_prompt_pairs.append(
                    (FLAT_TRAINING_SYSTEM_PROMPT, user_prompt)
                )
            end_idx = len(all_prompt_pairs)
            prompt_category_map.append((category_name, start_idx, end_idx))

        if not all_prompt_pairs:
            logger.warning("没有可用的训练 prompt")
            return []

        # 一次性批量推理
        raw_outputs = self._vllm_batch_generate(all_prompt_pairs)

        # 按中类解析结果
        all_skills: List[Dict] = []
        for category_name, start_idx, end_idx in prompt_category_map:
            category_skills: List[Dict] = []
            for idx in range(start_idx, end_idx):
                parsed = extract_json_from_response(raw_outputs[idx])
                if parsed is None:
                    logger.warning(
                        "中类 '%s': prompt #%d JSON 解析失败",
                        category_name,
                        idx - start_idx + 1,
                    )
                    continue
                # 兼容 LLM 直接输出数组 [{...}] 或包装对象 {"skills":[...]}
                if isinstance(parsed, list):
                    skills = parsed
                elif isinstance(parsed, dict):
                    skills = parsed.get("skills", [])
                else:
                    skills = []
                category_skills.extend(skills)

            deduped = self._merge_skills_by_name(category_skills)
            logger.info(
                "中类 '%s': 提取到 %d 个技能", category_name, len(deduped)
            )
            all_skills.extend(deduped)

        # 全局去重
        merged = self._merge_skills_by_name(all_skills)
        logger.info("所有中类合计提取技能: %d 个（去重后）", len(merged))
        return merged

    # ── 覆盖率评估与补词 ─────────────────────────────────────────

    def _evaluate_and_supplement_all(
        self,
        category_samples: Dict[str, Dict[str, List[str]]],
        current_skills: List[Dict],
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    ) -> List[Dict]:
        """对所有中类进行覆盖率评估并收集缺失技能。

        遍历每个中类的验证样本，使用 vLLM 评估当前词典的覆盖率。
        对于覆盖率不足的中类，提取缺失技能并汇总返回。

        参数:
            category_samples: 来自 ``_load_and_sample`` 的采样结果。
            current_skills: 当前词典中的全部技能列表。
            coverage_threshold: 覆盖率阈值（0~1），低于此值时提取缺失技能。

        返回:
            list[dict]: 所有中类汇总的缺失技能列表。
                如果所有中类均达标，返回空列表。

        说明:
            与提取阶段类似，本方法也会先收集所有中类的评估 prompt，
            然后一次性提交 vLLM 批量推理。
        """
        from .merge_similar_skills import extract_json_from_response

        logger.info("开始覆盖率评估（阈值 %.1f%%）...", coverage_threshold * 100)

        eval_prompt_pairs: List[Tuple[str, str]] = []
        eval_categories: List[str] = []

        # 构建技能摘要文本（所有中类共用同一份词典摘要）
        skill_lines = []
        for skill in current_skills[:200]:  # 限制摘要长度，避免超上下文
            aliases = skill.get("aliases", [])
            alias_text = (
                f" (别名: {', '.join(aliases[:5])})" if aliases else ""
            )
            skill_lines.append(f"- {skill['name']}{alias_text}")
        skill_summary = (
            "\n".join(skill_lines) if skill_lines else "(词典为空)"
        )

        for category_name, samples in category_samples.items():
            validation_texts = samples.get("validation_texts", [])
            if not validation_texts:
                continue

            validation_block = "\n".join(
                f"{i + 1}. {text[:MAX_SAMPLE_CHARS]}"
                for i, text in enumerate(validation_texts)
            )
            user_prompt = FLAT_EVALUATION_USER_TEMPLATE.format(
                skill_count=len(current_skills),
                skill_summary=skill_summary,
                sample_count=len(validation_texts),
                validation_text=validation_block,
            )
            eval_prompt_pairs.append(
                (FLAT_EVALUATION_SYSTEM_PROMPT, user_prompt)
            )
            eval_categories.append(category_name)

        if not eval_prompt_pairs:
            logger.info("没有可用的验证样本")
            return []

        # 一次性批量推理
        raw_outputs = self._vllm_batch_generate(eval_prompt_pairs)

        # 解析并收集缺失技能
        all_missing: List[Dict] = []
        passed_count = 0
        failed_count = 0

        for category_name, raw_text in zip(eval_categories, raw_outputs):
            parsed = extract_json_from_response(raw_text)
            if parsed is None or not isinstance(parsed, dict):
                logger.warning(
                    "中类 '%s': 评估结果 JSON 解析失败或格式异常", category_name
                )
                failed_count += 1
                continue

            accuracy = float(parsed.get("accuracy", 1.0))
            missing_skills = parsed.get("missing_skills", [])

            logger.info(
                "中类 '%s': 覆盖率 %.1f%%, 缺失 %d 个技能",
                category_name,
                accuracy * 100,
                len(missing_skills),
            )

            if accuracy >= coverage_threshold:
                passed_count += 1
            else:
                failed_count += 1
                all_missing.extend(missing_skills)

        logger.info(
            "覆盖率评估完成: %d 个中类达标, %d 个未达标, 共发现 %d 个缺失技能",
            passed_count,
            failed_count,
            len(all_missing),
        )
        return all_missing

    # ── 技能去重与标准化 ─────────────────────────────────────────

    @staticmethod
    def _normalize_skill_key(name: str) -> str:
        """将技能名称标准化为可比较的 key。

        标准化规则:
            1. 转小写
            2. 去除首尾空白
            3. 折叠内部连续空白为单个空格
            4. 保留 ``+``, ``#``, ``.`` 等对编程语言重要的字符

        参数:
            name: 原始技能名称。

        返回:
            str: 标准化后的名称（全小写、空白折叠）。

        示例:
            >>> FlatSkillPipeline._normalize_skill_key("  C++  编程  ")
            'c++ 编程'
        """
        import re as _re

        key = str(name).strip().lower()
        key = _re.sub(r"\s+", " ", key)
        # 对纯 ASCII 技能名进一步折叠空格，减少 "AUTO CAD" / "AutoCAD" 这类重复。
        if _re.fullmatch(r"[a-z0-9 .+#/\-]+", key):
            key = key.replace(" ", "")
        return key

    @staticmethod
    def _is_low_value_skill_name(name: str) -> bool:
        """判断技能名称是否过泛，不适合作为词典主名称。

        被过滤的典型条目包括：
            - 泛能力/泛知识：如“机械传动知识”“理论基础”
            - 泛工具容器：如“数据分析工具”“仿真软件”
            - 泛证书容器：如“资格证书”“执业资格证书”
        """
        import re as _re

        text = str(name).strip()
        if not text:
            return True
        return any(_re.search(pattern, text) for pattern in LOW_VALUE_SKILL_PATTERNS)

    @staticmethod
    def _is_low_value_alias(alias: str) -> bool:
        """判断 alias 是否过泛，过泛 alias 不应进入词典。"""
        import re as _re

        text = str(alias).strip()
        if not text:
            return True
        normalized = FlatSkillPipeline._normalize_skill_key(text)
        if len(normalized) <= 2 and not _re.fullmatch(r"[a-z0-9.+#/\-]+", normalized):
            return True
        return any(_re.fullmatch(pattern, text) for pattern in LOW_VALUE_ALIAS_PATTERNS)

    @staticmethod
    def _clean_skill_record(skill: Dict) -> Dict | None:
        """标准化并清洗单条技能记录。

        返回 ``None`` 表示该技能过泛或为空，应直接丢弃。
        """
        name = str(skill.get("name", "")).strip()
        if FlatSkillPipeline._is_low_value_skill_name(name):
            return None

        aliases = []
        for alias in skill.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if not alias_text:
                continue
            if FlatSkillPipeline._is_low_value_alias(alias_text):
                continue
            aliases.append(alias_text)

        return {
            "name": name,
            "aliases": aliases,
            "skill_type": str(skill.get("skill_type", "")).strip(),
            "notes": str(skill.get("notes", "")).strip(),
        }

    @staticmethod
    def _filter_skill_records(skills: List[Dict]) -> List[Dict]:
        """对技能列表执行统一清洗，移除低质量词条。"""
        filtered: List[Dict] = []
        for skill in skills:
            cleaned = FlatSkillPipeline._clean_skill_record(skill)
            if cleaned is not None:
                filtered.append(cleaned)
        return filtered

    @staticmethod
    def _merge_skills_by_name(skills: List[Dict]) -> List[Dict]:
        """按标准化名称去重合并技能列表。

        相同标准化名称的技能会被合并:
            - 保留第一次出现的 name 作为主名称
            - 后续出现的 name 和 aliases 并入主记录的 aliases
            - notes 和 skill_type 合并（非空优先）

        参数:
            skills: 待去重的技能列表。

        返回:
            list[dict]: 去重合并后的技能列表，保持首次出现的顺序。
        """
        merged_map: Dict[str, Dict] = {}  # normalized_key -> skill dict
        order: List[str] = []  # 保持插入顺序

        for raw_skill in skills:
            skill = FlatSkillPipeline._clean_skill_record(raw_skill)
            if skill is None:
                continue
            name = skill["name"]

            key = FlatSkillPipeline._normalize_skill_key(name)
            aliases = skill["aliases"]
            skill_type = skill["skill_type"]
            notes = skill["notes"]

            if key not in merged_map:
                merged_map[key] = {
                    "name": name,
                    "aliases": list(aliases),
                    "skill_type": skill_type,
                    "notes": notes,
                }
                order.append(key)
            else:
                existing = merged_map[key]

                # 将新出现的别名并入
                existing_name_key = FlatSkillPipeline._normalize_skill_key(
                    existing["name"]
                )
                existing_alias_keys = {
                    FlatSkillPipeline._normalize_skill_key(a)
                    for a in existing["aliases"]
                }

                # 新名称若与主名称不同，加入 aliases
                if key != existing_name_key:
                    if key not in existing_alias_keys:
                        existing["aliases"].append(name)
                        existing_alias_keys.add(key)

                for alias in aliases:
                    alias_key = FlatSkillPipeline._normalize_skill_key(alias)
                    if (
                        alias_key != existing_name_key
                        and alias_key not in existing_alias_keys
                    ):
                        existing["aliases"].append(alias)
                        existing_alias_keys.add(alias_key)

                # 补充 skill_type / notes（非空优先）
                if skill_type and not existing["skill_type"]:
                    existing["skill_type"] = skill_type
                if notes and notes not in existing.get("notes", ""):
                    if existing["notes"]:
                        existing["notes"] += "; " + notes
                    else:
                        existing["notes"] = notes

        # 去重 aliases
        result: List[Dict] = []
        for key in order:
            skill = merged_map[key]
            seen = {FlatSkillPipeline._normalize_skill_key(skill["name"])}
            unique_aliases = []
            for alias in skill["aliases"]:
                alias_key = FlatSkillPipeline._normalize_skill_key(alias)
                if alias_key not in seen:
                    seen.add(alias_key)
                    unique_aliases.append(alias)
            skill["aliases"] = sorted(unique_aliases)
            result.append(skill)

        return result

    @staticmethod
    def _deduplicate_final(skills: List[Dict]) -> List[Dict]:
        """全局去重：确保所有 name 和 aliases 互不冲突。

        检测并解决以下冲突:
            1. 技能 A 的 name 出现在技能 B 的 aliases 中
               → 从 B 的 aliases 中移除。
            2. 两个技能共享相同的 alias
               → 仅保留在第一个技能中。
            3. 技能 A 的 alias 等于技能 B 的 name
               → 从 A 的 aliases 中移除。

        参数:
            skills: 待去重的技能列表。

        返回:
            list[dict]: 全局去重后的技能列表，确保 name 和 aliases
            构成的全集无重复（用于正则匹配时不会产生歧义）。
        """
        import copy as _copy

        result = _copy.deepcopy(
            FlatSkillPipeline._filter_skill_records(skills)
        )

        # 建立 name 索引
        name_keys: set = set()
        for skill in result:
            name_keys.add(
                FlatSkillPipeline._normalize_skill_key(skill["name"])
            )

        # 全局 alias 去重: name 占用的 key 不能被 alias 使用
        used_alias_keys: set = set(name_keys)

        for skill in result:
            name_key = FlatSkillPipeline._normalize_skill_key(skill["name"])
            cleaned_aliases = []
            for alias in skill.get("aliases", []):
                alias_key = FlatSkillPipeline._normalize_skill_key(alias)
                if alias_key == name_key:
                    continue  # alias 与自身 name 相同
                if alias_key in used_alias_keys:
                    continue  # alias 已被其他技能的 name 或 alias 占用
                used_alias_keys.add(alias_key)
                cleaned_aliases.append(alias)
            skill["aliases"] = sorted(cleaned_aliases)

        return result

    # ── 输出 ─────────────────────────────────────────────────────

    def _save_dictionary(
        self, skills: List[Dict], output_path: Path | str
    ) -> None:
        """保存平面化技能词典为 JSON 文件。

        输出格式::

            {
                "metadata": {
                    "schema_version": 3,
                    "created_at": "2026-04-12T10:00:00",
                    "pipeline": "FlatSkillPipeline_v2",
                    "model": "Qwen3-8B",
                    "skill_count": 5000,
                    "alias_count": 12000
                },
                "skills": [
                    {
                        "name": "...",
                        "aliases": [...],
                        "skill_type": "...",
                        "notes": ""
                    },
                    ...
                ]
            }

        参数:
            skills: 最终的技能列表。
            output_path: 输出 JSON 文件路径。
        """
        skills = self._deduplicate_final(skills)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_aliases = sum(len(s.get("aliases", [])) for s in skills)

        data = {
            "metadata": {
                "schema_version": 3,
                "created_at": _now_text(),
                "pipeline": "FlatSkillPipeline_v2",
                "model": self.model_path,
                "skill_count": len(skills),
                "alias_count": total_aliases,
            },
            "skills": skills,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "词典已保存: %s (%d 技能, %d 别名)",
            output_path,
            len(skills),
            total_aliases,
        )

    # ── 主流程编排 ───────────────────────────────────────────────

    def run(
        self,
        train_size: int = 100,
        validation_size: int = 10,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
        seed: int = 42,
        limit_job_rows: int | None = None,
        limit_categories: int | None = None,
        parse_workers: int = 1,
        output_path: str | Path | None = None,
    ) -> None:
        """执行完整的平面化技能词典构建流程。

        编排以下步骤:
            1. 数据加载与采样（按职业中类）
            2. vLLM 批量推理提取硬技能
            3. 覆盖率评估与缺失技能补充
            4. 全局去重
            5. 保存词典

        参数:
            train_size: 每个中类的训练样本数。
            validation_size: 每个中类的验证样本数。
            coverage_threshold: 覆盖率阈值（0~1），低于此值时补充缺失技能。
            seed: 随机种子。
            limit_job_rows: 调试用，限制读取的岗位行数。
            limit_categories: 调试用，限制处理的中类数量。
            parse_workers: 岗位描述解析并发数。
            output_path: 输出词典路径。
                默认为 ``config.dict_dir / "flat_skill_dictionary.json"``。

        流程概览::

            数据加载 → 分组采样 → vLLM提取 → 合并去重
                                                  ↓
                                            覆盖率评估
                                                  ↓
                                    ┌─ 达标 → 保存词典
                                    └─ 未达标 → 补充缺失 → 再次去重 → 保存词典
        """
        if output_path is None:
            output_path = self.config.dict_dir / "flat_skill_dictionary.json"
        output_path = Path(output_path)

        logger.info("=" * 60)
        logger.info("  FlatSkillPipeline v2 — 平面化技能词典构建")
        logger.info("  模型: %s", self.model_path)
        logger.info(
            "  训练样本: %d/中类, 验证样本: %d/中类",
            train_size,
            validation_size,
        )
        logger.info("  覆盖率阈值: %.1f%%", coverage_threshold * 100)
        logger.info("=" * 60)

        # ── Step 1: 数据加载与采样 ──
        logger.info("[Step 1/5] 数据加载与采样...")
        category_samples = self._load_and_sample(
            train_size=train_size,
            validation_size=validation_size,
            seed=seed,
            limit_job_rows=limit_job_rows,
            limit_categories=limit_categories,
            parse_workers=parse_workers,
        )

        if not category_samples:
            logger.error("没有可用的采样数据，流程终止")
            return

        # ── Step 2: vLLM 批量提取硬技能 ──
        logger.info("[Step 2/5] vLLM 批量提取硬技能...")
        all_skills = self._extract_skills_from_all_categories(
            category_samples
        )

        if not all_skills:
            logger.error("未提取到任何技能，流程终止")
            return

        logger.info("初步提取技能数: %d", len(all_skills))

        # ── Step 3: 覆盖率评估与缺失技能补充 ──
        logger.info("[Step 3/5] 覆盖率评估与缺失技能补充...")
        missing_skills = self._evaluate_and_supplement_all(
            category_samples=category_samples,
            current_skills=all_skills,
            coverage_threshold=coverage_threshold,
        )

        if missing_skills:
            logger.info("补充缺失技能 %d 个", len(missing_skills))
            all_skills.extend(missing_skills)
            all_skills = self._merge_skills_by_name(all_skills)
            logger.info("补充后技能数: %d", len(all_skills))

        # ── Step 4: 全局去重 ──
        logger.info("[Step 4/5] 全局去重（确保 name + aliases 唯一）...")
        final_skills = self._deduplicate_final(all_skills)
        logger.info("最终技能数: %d", len(final_skills))

        # ── Step 5: 保存词典 ──
        logger.info("[Step 5/5] 保存词典...")
        self._save_dictionary(final_skills, output_path)

        total_aliases = sum(
            len(s.get("aliases", [])) for s in final_skills
        )
        logger.info("=" * 60)
        logger.info("  流程完成！")
        logger.info("  技能数: %d", len(final_skills))
        logger.info("  别名数: %d", total_aliases)
        logger.info("  输出文件: %s", output_path)
        logger.info("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    支持的子命令:
        - ``flat-run`` (v2, 推荐): 使用 vLLM + Qwen3-8B 一键构建平面化技能词典。
        - ``prepare`` (v1, Legacy): 生成训练样本和 LLM prompt 文件。
        - ``iterate`` (v1, Legacy): 执行一轮覆盖率验证和补词 prompt。
        - ``status``  (v1, Legacy): 查看迭代状态。

    返回:
        argparse.ArgumentParser: 配置好的参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="职业技能词典构建流程（v1 Legacy + v2 Flat）"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── v2: flat-run ─────────────────────────────────────────────
    flat_run = subparsers.add_parser(
        "flat-run",
        help="(v2 推荐) 使用 vLLM 一键构建平面化技能词典",
    )
    flat_run.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Qwen3-8B 模型路径 (默认: {DEFAULT_MODEL_PATH})",
    )
    flat_run.add_argument(
        "--train-size", type=int, default=100,
        help="每个职业中类的训练样本数 (默认: 100)",
    )
    flat_run.add_argument(
        "--validation-size", type=int, default=10,
        help="每个职业中类的验证样本数 (默认: 10)",
    )
    flat_run.add_argument(
        "--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD,
        help=f"覆盖率阈值 (默认: {DEFAULT_COVERAGE_THRESHOLD})",
    )
    flat_run.add_argument(
        "--seed", type=int, default=42, help="随机种子 (默认: 42)",
    )
    flat_run.add_argument(
        "--limit-job-rows", type=int, default=None,
        help="调试用，限制每张招聘表读取行数",
    )
    flat_run.add_argument(
        "--limit-categories", type=int, default=None,
        help="调试用，限制处理的职业中类数量",
    )
    flat_run.add_argument(
        "--parse-workers", type=int, default=1,
        help="岗位描述解析并发数 (默认: 1)",
    )
    flat_run.add_argument(
        "--gpu-memory-utilization", type=float, default=0.80,
        help="GPU 显存利用率 (默认: 0.80)",
    )
    flat_run.add_argument(
        "--max-model-len", type=int, default=8192,
        help="vLLM 最大序列长度 (默认: 8192)",
    )
    flat_run.add_argument(
        "--max-num-seqs", type=int, default=48,
        help="vLLM 最大并发序列数 (默认: 48)",
    )
    flat_run.add_argument(
        "--output", type=str, default=None,
        help="输出词典路径 (默认: dicts/flat_skill_dictionary.json)",
    )

    # ── v1 Legacy: prepare / iterate / status ────────────────────
    prepare = subparsers.add_parser(
        "prepare", help="(v1 Legacy) 生成训练样本、验证池和 LLM prompt",
    )
    prepare.add_argument("--train-size", type=int, default=100, help="每个职业细类的训练样本数")
    prepare.add_argument("--validation-batch-size", type=int, default=10, help="每轮验证每个细类抽取数量")
    prepare.add_argument("--seed", type=int, default=42, help="随机种子")
    prepare.add_argument("--limit-job-rows", type=int, default=None, help="仅用于调试，限制每张招聘表读取行数")
    prepare.add_argument("--limit-categories", type=int, default=None, help="仅用于调试，限制处理的职业细类数量")
    prepare.add_argument("--match-workers", type=int, default=4, help="岗位匹配并发数")
    prepare.add_argument("--match-chunk-size", type=int, default=256, help="岗位匹配分块大小")
    prepare.add_argument("--parse-workers", type=int, default=1, help="岗位描述切分并发数")
    prepare.add_argument("--progress", action="store_true", help="显示岗位匹配进度")

    iterate = subparsers.add_parser(
        "iterate", help="(v1 Legacy) 执行一轮覆盖率验证和补词 prompt 生成",
    )
    iterate.add_argument("--validation-batch-size", type=int, default=10, help="每轮验证每个细类抽取数量")
    iterate.add_argument("--coverage-threshold", type=float, default=0.95, help="目标覆盖率阈值")
    iterate.add_argument("--limit-categories", type=int, default=None, help="仅用于调试，限制验证的职业细类数量")
    iterate.add_argument("--parse-workers", type=int, default=1, help="岗位描述切分并发数")

    subparsers.add_parser("status", help="(v1 Legacy) 查看当前技能词典迭代状态")
    return parser


def main() -> None:
    """CLI 入口。

    根据子命令分发到对应的流水线:
        - ``flat-run``: 调用 ``FlatSkillPipeline.run()``（v2 推荐）。
        - ``prepare`` / ``iterate`` / ``status``: 调用
          ``OccupationSkillPipeline`` 的对应方法（v1 Legacy）。
    """
    parser = build_parser()
    args = parser.parse_args()
    config = load_skill_extraction_config()

    # ── v2: flat-run ─────────────────────────────────────────────
    if args.command == "flat-run":
        flat_pipeline = FlatSkillPipeline(
            config=config,
            model_path=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )
        flat_pipeline.run(
            train_size=args.train_size,
            validation_size=args.validation_size,
            coverage_threshold=args.coverage_threshold,
            seed=args.seed,
            limit_job_rows=args.limit_job_rows,
            limit_categories=args.limit_categories,
            parse_workers=args.parse_workers,
            output_path=args.output,
        )
        return

    # ── v1 Legacy ────────────────────────────────────────────────
    pipeline = OccupationSkillPipeline(config)

    if args.command == "prepare":
        pipeline.prepare(
            train_size=args.train_size,
            validation_batch_size=args.validation_batch_size,
            seed=args.seed,
            limit_job_rows=args.limit_job_rows,
            limit_categories=args.limit_categories,
            match_workers=args.match_workers,
            match_chunk_size=args.match_chunk_size,
            parse_workers=args.parse_workers,
            show_progress=args.progress,
        )
        return

    if args.command == "iterate":
        pipeline.iterate(
            validation_batch_size=args.validation_batch_size,
            coverage_threshold=args.coverage_threshold,
            limit_categories=args.limit_categories,
            parse_workers=args.parse_workers,
        )
        return

    if args.command == "status":
        pipeline.status()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
