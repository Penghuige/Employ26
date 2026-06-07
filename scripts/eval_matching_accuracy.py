"""job_title_parsing 工作流准确性检验脚本。

对 100 条抽样数据进行基线（纯规则+检索）和 LLM 增强（+Qwen 重排序）的对比评估。

用法:
    .conda/python.exe scripts/eval_matching_accuracy.py [--llm] [--sample-size 100]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_matching")


def sample_jobs(sample_size: int = 100, seed: int = 42) -> pd.DataFrame:
    """从 1% 样本 CSV 中随机抽取指定数量的岗位。

    三平台等比例抽取，确保多样性。

    Args:
        sample_size: 总抽样数量。
        seed: 随机种子。

    Returns:
        pd.DataFrame: 含 岗位名称、岗位描述 等字段的样本 DataFrame。
    """
    sample_dir = PROJECT_ROOT / "output" / "samples"
    csv_files = list(sample_dir.glob("*_样本_1%.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到样本 CSV: {sample_dir}")

    per_file = max(1, sample_size // len(csv_files))
    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="gbk")
        if len(df) > per_file:
            df = df.sample(n=per_file, random_state=seed)
        frames.append(df)
        logger.info("从 %s 抽取 %d 条", csv_path.name, len(frames[-1]))

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > sample_size:
        combined = combined.sample(n=sample_size, random_state=seed)
    logger.info("总样本数: %d", len(combined))
    return combined


def load_catalog_csv() -> pd.DataFrame:
    """加载预处理的职业大典 CSV。

    Returns:
        pd.DataFrame: 含检索字段的职业大典。
    """
    csv_path = PROJECT_ROOT / "output" / "catalog_preprocessed.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8")
        # 确保 aliases 列被解析为 list
        if "aliases" in df.columns:
            import ast
            df["aliases"] = df["aliases"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
            )
        logger.info("从 CSV 加载职业大典: %d 条", len(df))
        return df

    raise FileNotFoundError(
        "未找到 catalog_preprocessed.csv，请先运行:\n"
        "  python -m src.job_title_parsing.cli preprocess-catalog"
    )


def run_baseline_matching(
    catalog_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
) -> pd.DataFrame:
    """运行基线（纯规则+检索）岗位匹配。

    Args:
        catalog_df: 职业大典 DataFrame。
        jobs_df: 招聘岗位 DataFrame。

    Returns:
        pd.DataFrame: 匹配结果 DataFrame。
    """
    from src.job_title_parsing.matching_pipeline import MatchPipeline

    pipeline = MatchPipeline(catalog_df=catalog_df)

    results = pipeline.match_batch(
        jobs_df=jobs_df,
        job_title_col="岗位名称",
        job_desc_col="岗位描述",
        job_id_col=None,
        top_k=5,
        debug=False,
        workers=4,
        show_progress=True,
        chunk_size=32,
    )
    return results


def run_llm_rerank(
    baseline_results: pd.DataFrame,
    jobs_df: pd.DataFrame,
) -> Optional[List[Any]]:
    """对基线匹配结果进行 LLM 重排序。

    Args:
        baseline_results: 基线匹配结果 DataFrame。
        jobs_df: 原始岗位 DataFrame（用于获取完整岗位描述）。

    Returns:
        Optional[List[RerankResult]]: 重排序结果列表，服务不可用时返回 None。
    """
    from src.job_title_parsing.llm_reranker import LLMReranker

    reranker = LLMReranker()
    if not reranker.is_server_available():
        logger.warning("vLLM 服务不可用，跳过 LLM 重排序。请先启动: python -m src.llm.vllm_server serve")
        return None

    # 构建带岗位描述的匹配结果
    records = baseline_results.to_dict(orient="records")
    jobs_lookup: Dict[int, Dict[str, Any]] = {}
    for i, row in jobs_df.iterrows():
        jobs_lookup[i] = {"岗位名称": row.get("岗位名称", ""), "岗位描述": row.get("岗位描述", "")}

    for idx, rec in enumerate(records):
        job_info = jobs_lookup.get(idx, {})
        rec["job_title"] = rec.get("job_title", job_info.get("岗位名称", ""))
        rec["job_description"] = job_info.get("岗位描述", "")
        rec["job_id"] = idx

    logger.info("开始 LLM 重排序 (%d 条)...", len(records))
    results = reranker.rerank_batch(records, sleep_between=0.3)
    return results


def compute_metrics(
    baseline_df: pd.DataFrame,
    llm_results: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """计算评估指标。

    Args:
        baseline_df: 基线匹配结果。
        llm_results: LLM 重排序结果（可选）。

    Returns:
        Dict[str, Any]: 评估指标字典。
    """
    metrics: Dict[str, Any] = {
        "total_samples": len(baseline_df),
        "baseline": {},
        "llm_enhanced": {},
    }

    # 基线指标
    baseline_df_copy = baseline_df.copy()
    baseline_df_copy["has_candidates"] = baseline_df_copy["candidates"].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False
    )
    baseline_df_copy["confidence_high"] = baseline_df_copy["confidence_level"] == "high"
    baseline_df_copy["confidence_medium"] = baseline_df_copy["confidence_level"] == "medium"
    baseline_df_copy["confidence_low"] = baseline_df_copy["confidence_level"] == "low"
    baseline_df_copy["needs_review"] = baseline_df_copy["is_review_needed"] == True

    metrics["baseline"]["has_candidates_rate"] = float(baseline_df_copy["has_candidates"].mean())
    metrics["baseline"]["high_confidence_rate"] = float(baseline_df_copy["confidence_high"].mean())
    metrics["baseline"]["medium_confidence_rate"] = float(baseline_df_copy["confidence_medium"].mean())
    metrics["baseline"]["low_confidence_rate"] = float(baseline_df_copy["confidence_low"].mean())
    metrics["baseline"]["needs_review_rate"] = float(baseline_df_copy["needs_review"].mean())

    # Top1 平均分数
    metrics["baseline"]["avg_top1_score"] = float(baseline_df_copy["top1_score"].mean())

    # 风险标记分布
    all_risks: List[str] = []
    for flags in baseline_df_copy["risk_flags"]:
        if isinstance(flags, list):
            all_risks.extend(flags)
    from collections import Counter
    metrics["baseline"]["risk_flag_distribution"] = dict(Counter(all_risks).most_common(10))
    metrics["baseline"]["no_risk_flags_rate"] = float(
        baseline_df_copy["risk_flags"].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).mean()
    )

    # LLM 增强指标
    if llm_results is not None:
        llm_ok = sum(1 for r in llm_results if r.llm_call_ok)
        llm_fail = len(llm_results) - llm_ok
        llm_all_irrelevant = sum(1 for r in llm_results if r.all_irrelevant)
        llm_changed_top1 = sum(
            1 for r in llm_results
            if r.llm_call_ok and r.reranked_candidates
            and r.reranked_candidates[0].get("code", "") != r.baseline_top1_code
        )

        metrics["llm_enhanced"]["llm_call_success_rate"] = llm_ok / max(len(llm_results), 1)
        metrics["llm_enhanced"]["llm_call_failures"] = llm_fail
        metrics["llm_enhanced"]["all_irrelevant_count"] = llm_all_irrelevant
        metrics["llm_enhanced"]["top1_changed_rate"] = llm_changed_top1 / max(len(llm_results), 1)

        # LLM 重排后 top1 的 LLM 分数分布
        llm_scores = [
            r.reranked_candidates[0].get("llm_score", 0)
            for r in llm_results
            if r.llm_call_ok and r.reranked_candidates
        ]
        if llm_scores:
            metrics["llm_enhanced"]["avg_llm_top1_score"] = sum(llm_scores) / len(llm_scores)

    return metrics


def generate_report(
    baseline_df: pd.DataFrame,
    metrics: Dict[str, Any],
    llm_results: Optional[List[Any]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """生成评估报告。

    Args:
        baseline_df: 基线匹配结果。
        metrics: 评估指标。
        llm_results: LLM 重排序结果（可选）。
        output_path: 报告输出路径。

    Returns:
        str: Markdown 格式的评估报告文本。
    """
    lines: List[str] = []
    lines.append("# job_title_parsing 工作流准确性检验报告")
    lines.append("")
    lines.append(f"检验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"样本数量: {metrics['total_samples']}")
    lines.append("")

    # 基线结果
    lines.append("## 一、基线匹配结果（纯规则 + 检索）")
    lines.append("")
    b = metrics["baseline"]
    lines.append(f"- 有候选返回率: {b['has_candidates_rate']:.1%}")
    lines.append(f"- 高置信度占比: {b['high_confidence_rate']:.1%}")
    lines.append(f"- 中置信度占比: {b['medium_confidence_rate']:.1%}")
    lines.append(f"- 低置信度占比: {b['low_confidence_rate']:.1%}")
    lines.append(f"- 需要人工复核率: {b['needs_review_rate']:.1%}")
    lines.append(f"- Top1 平均基线分数: {b['avg_top1_score']:.4f}")
    lines.append(f"- 无风险标记率: {b['no_risk_flags_rate']:.1%}")
    lines.append("")
    lines.append("### 风险标记分布")
    for flag, count in b.get("risk_flag_distribution", {}).items():
        lines.append(f"- {flag}: {count}")
    lines.append("")

    # LLM 增强结果
    if llm_results is not None:
        lle = metrics.get("llm_enhanced", {})
        lines.append("## 二、LLM 增强结果（Qwen3.6-27B 重排序）")
        lines.append("")
        lines.append(f"- LLM 调用成功率: {lle.get('llm_call_success_rate', 0):.1%}")
        lines.append(f"- LLM 调用失败数: {lle.get('llm_call_failures', 0)}")
        lines.append(f"- 判定为全部不相关: {lle.get('all_irrelevant_count', 0)} 条")
        lines.append(f"- Top1 被改变率: {lle.get('top1_changed_rate', 0):.1%}")
        lines.append(f"- LLM Top1 平均分: {lle.get('avg_llm_top1_score', 0):.4f}")
        lines.append("")

        # Top1 变更案例
        changed = [
            r for r in llm_results
            if r.llm_call_ok and r.reranked_candidates
            and r.reranked_candidates[0].get("code", "") != r.baseline_top1_code
        ]
        if changed:
            lines.append(f"### Top1 被改变的案例 ({len(changed)} 条)")
            lines.append("")
            for r in changed[:10]:
                new_top1 = r.reranked_candidates[0]
                lines.append(f"- **{r.job_title}**")
                lines.append(f"  - 基线 Top1: [{r.baseline_top1_code}] {r.baseline_top1_title} (score={r.baseline_top1_score:.4f})")
                lines.append(f"  - LLM Top1: [{new_top1.get('code')}] {new_top1.get('title')} (score={new_top1.get('llm_score', 0):.4f})")
                lines.append(f"  - 理由: {new_top1.get('reason', 'N/A')[:100]}")
                lines.append(f"  - 总结: {r.summary[:100]}")
                lines.append("")

        # 全部不相关案例
        irrelevant = [r for r in llm_results if r.all_irrelevant and r.llm_call_ok]
        if irrelevant:
            lines.append(f"### 被 LLM 判定为全部不相关的案例 ({len(irrelevant)} 条)")
            lines.append("")
            for r in irrelevant[:5]:
                lines.append(f"- **{r.job_title}** (基线 Top1: [{r.baseline_top1_code}] {r.baseline_top1_title})")
                lines.append(f"  - 总结: {r.summary[:150]}")
                lines.append("")

    # 抽样展示
    lines.append("## 三、基线匹配抽样展示（前 15 条）")
    lines.append("")
    lines.append("| # | 岗位名称 | Top1 职业 | 分数 | 置信度 | 风险标记 |")
    lines.append("|---|----------|-----------|------|--------|----------|")
    for i, (_, row) in enumerate(baseline_df.head(15).iterrows()):
        title = str(row.get("job_title", ""))[:25]
        top1 = str(row.get("top1_title", ""))[:20]
        score = float(row.get("top1_score", 0))
        conf = row.get("confidence_level", "?")
        risks = ", ".join(row.get("risk_flags", [])[:3])
        lines.append(f"| {i+1} | {title} | {top1} | {score:.4f} | {conf} | {risks} |")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report, encoding="utf-8")
        logger.info("报告已保存至: %s", output_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="job_title_parsing 工作流准确性检验")
    parser.add_argument("--sample-size", type=int, default=100, help="样本数量")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 重排序")
    parser.add_argument("--output", type=str, default="", help="报告输出路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 1. 抽样
    logger.info("=" * 60)
    logger.info("步骤 1/4: 抽取样本")
    logger.info("=" * 60)
    jobs_df = sample_jobs(args.sample_size, args.seed)

    # 2. 加载职业大典
    logger.info("=" * 60)
    logger.info("步骤 2/4: 加载职业大典")
    logger.info("=" * 60)
    catalog_df = load_catalog_csv()

    # 3. 基线匹配
    logger.info("=" * 60)
    logger.info("步骤 3/4: 运行基线匹配")
    logger.info("=" * 60)
    baseline_results = run_baseline_matching(catalog_df, jobs_df)

    # 4. LLM 重排序（可选）
    llm_results = None
    if args.llm:
        logger.info("=" * 60)
        logger.info("步骤 4/4: LLM 重排序")
        logger.info("=" * 60)
        llm_results = run_llm_rerank(baseline_results, jobs_df)

    # 5. 计算指标
    metrics = compute_metrics(baseline_results, llm_results)

    # 6. 生成报告
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "output" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "matching_accuracy_report.md"

    report = generate_report(baseline_results, metrics, llm_results, output_path)
    print(report)

    # 保存详细数据
    detail_path = output_path.with_suffix(".json")
    detail_data = {
        "config": {"sample_size": args.sample_size, "seed": args.seed, "llm_enabled": args.llm},
        "metrics": metrics,
        "baseline_results": json.loads(baseline_results.to_json(orient="records", force_ascii=False)),
    }
    if llm_results:
        detail_data["llm_results"] = [
            {
                "job_id": r.job_id,
                "job_title": r.job_title,
                "baseline_top1_code": r.baseline_top1_code,
                "baseline_top1_title": r.baseline_top1_title,
                "baseline_top1_score": r.baseline_top1_score,
                "reranked_top1_code": r.reranked_candidates[0].get("code", "") if r.reranked_candidates else "",
                "reranked_top1_title": r.reranked_candidates[0].get("title", "") if r.reranked_candidates else "",
                "reranked_top1_score": r.reranked_candidates[0].get("llm_score", 0) if r.reranked_candidates else 0,
                "all_irrelevant": r.all_irrelevant,
                "summary": r.summary,
                "llm_call_ok": r.llm_call_ok,
            }
            for r in llm_results
        ]
    detail_path.write_text(json.dumps(detail_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("详细数据已保存至: %s", detail_path)


if __name__ == "__main__":
    main()
