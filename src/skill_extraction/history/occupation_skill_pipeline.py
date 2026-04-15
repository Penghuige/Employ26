"""v1 职业细类技能词典流水线。"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import SkillExtractionConfig
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


def _build_supplement_prompt(
    detail_row: Dict,
    current_skills: List[Dict],
    uncovered_items: List[str],
) -> str:
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
    """v1 按职业细类维护分层词典的流水线。"""

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

    def import_llm_results(
        self,
        source_path: str | Path,
        recursive: bool = True,
        dry_run: bool = False,
    ) -> None:
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

    def _initialize_state(
        self,
        category_summary_df: pd.DataFrame,
        validation_batch_size: int,
        seed: int,
    ) -> None:
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
