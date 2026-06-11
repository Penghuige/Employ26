"""结构化统计的 PostgreSQL 主数据源。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from src.db.postgres import create_pg_engine
from src.db.recruitment_jobs_normalized import (
    DEFAULT_NORMALIZED_TABLE,
    quote_table_name,
)
from src.skill_extraction.config import load_skill_extraction_config


DEFAULT_MATCH_TABLE = "public.skill_extraction_requirement_matches"

STRUCTURED_SOURCE_COLUMNS = [
    "recruitment_record_id",
    "source_platform",
    "source_table",
    "source_row_number",
    "job_title",
    "work_city",
    "company_name",
    "publish_date",
    "publish_month",
    "salary_raw",
    "education_requirement_raw",
    "experience_requirement_raw",
    "company_size_raw",
    "company_industry_raw",
    "city_normalized",
    "industry_normalized",
    "company_size_normalized",
    "occupation_code",
    "occupation_title",
    "occupation_core",
    "occupation_category",
    "occupation_major_category",
    "occupation_middle_category",
    "occupation_minor_category",
    "occupation_detail_category",
    "occupation_confidence",
    "occupation_is_matched",
]


@dataclass(frozen=True)
class StructuredSourceConfig:
    """结构化统计 PostgreSQL 源配置。"""

    normalized_table: str = DEFAULT_NORMALIZED_TABLE
    occupation_match_table: str = DEFAULT_MATCH_TABLE


def load_default_structured_source_config() -> StructuredSourceConfig:
    """从项目配置读取结构化统计主输入表。"""
    skill_config = load_skill_extraction_config()
    return StructuredSourceConfig(
        normalized_table=skill_config.recruitment_normalized_table,
        occupation_match_table=skill_config.requirement_match_table,
    )


def build_structured_source_query(config: StructuredSourceConfig) -> str:
    """构建结构化统计主输入查询。"""
    normalized_table = quote_table_name(config.normalized_table)
    match_table = quote_table_name(config.occupation_match_table)
    return f"""
        WITH latest_occupation_match AS (
            SELECT *
            FROM (
                SELECT
                    m.recruitment_record_id,
                    m.occupation_code,
                    m.occupation_title,
                    m."大类" AS occupation_major_category,
                    m."中类" AS occupation_middle_category,
                    m."小类" AS occupation_minor_category,
                    m."细类" AS occupation_detail_category,
                    m.is_matched AS occupation_is_matched,
                    COALESCE(m.top1_score, 0) AS occupation_confidence,
                    row_number() OVER (
                        PARTITION BY m.recruitment_record_id
                        ORDER BY
                            CASE WHEN COALESCE(m.is_matched, false) THEN 0 ELSE 1 END,
                            COALESCE(m.top1_score, 0) DESC,
                            COALESCE(m.occupation_code, '') ASC
                    ) AS rn
                FROM {match_table} m
                WHERE COALESCE(m.recruitment_record_id, '') <> ''
            ) ranked
            WHERE rn = 1
        )
        SELECT
            n.recruitment_record_id,
            n.source_platform,
            n.source_table,
            n.source_row_number,
            n.job_title,
            n.work_city,
            n.company_name,
            n.publish_date,
            n.salary_raw,
            n.education_requirement_raw,
            n.experience_requirement_raw,
            n.company_size_raw,
            n.company_industry_raw,
            m.occupation_code,
            m.occupation_title,
            m.occupation_major_category,
            m.occupation_middle_category,
            m.occupation_minor_category,
            m.occupation_detail_category,
            m.occupation_confidence,
            m.occupation_is_matched
        FROM {normalized_table} n
        LEFT JOIN latest_occupation_match m
          ON m.recruitment_record_id = n.recruitment_record_id
    """


def load_structured_analysis_dataframe(
    config: StructuredSourceConfig | None = None,
) -> pd.DataFrame:
    """读取结构化统计主输入，并补齐分析兼容字段。"""
    resolved_config = config or load_default_structured_source_config()
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            source_df = pd.read_sql_query(
                text(build_structured_source_query(resolved_config)),
                connection,
            )
    finally:
        engine.dispose()

    return normalize_structured_source_dataframe(source_df)


def normalize_structured_source_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    """补齐结构化统计需要的规范列与历史兼容列。"""
    df = source_df.copy()
    for column in STRUCTURED_SOURCE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df["publish_month"] = df["publish_date"].map(parse_publish_month)
    df["city_normalized"] = df["work_city"].map(normalize_city)
    df["industry_normalized"] = df["company_industry_raw"].map(normalize_industry)
    df["company_size_normalized"] = df["company_size_raw"].map(normalize_company_size)

    occupation_title = df["occupation_title"].fillna("").astype(str).str.strip()
    detail_category = df["occupation_detail_category"].fillna("").astype(str).str.strip()
    middle_category = df["occupation_middle_category"].fillna("").astype(str).str.strip()
    major_category = df["occupation_major_category"].fillna("").astype(str).str.strip()

    df["occupation_core"] = occupation_title.where(occupation_title != "", detail_category)
    df["occupation_category"] = middle_category.where(middle_category != "", major_category)
    df["occupation_core"] = df["occupation_core"].replace("", pd.NA)
    df["occupation_category"] = df["occupation_category"].replace("", pd.NA)

    # 兼容现有统计脚本的旧字段名；字段来源仍是 PostgreSQL 规范层。
    df["岗位名称"] = df["job_title"]
    df["工作城市"] = df["work_city"]
    df["公司名称"] = df["company_name"]
    df["发布时间"] = df["publish_date"]
    df["薪资水平"] = df["salary_raw"]
    df["学历要求"] = df["education_requirement_raw"]
    df["经验要求"] = df["experience_requirement_raw"]
    df["公司规模"] = df["company_size_raw"]
    df["公司行业"] = df["company_industry_raw"]
    df["city_clean"] = df["city_normalized"]
    df["industry_clean"] = df["industry_normalized"]
    df["occupation_confidence"] = df["occupation_confidence"]
    return df


def safe_string(value: object) -> str:
    """把数据库空值安全转成去空白字符串。"""
    if value is None or pd.isna(value):
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def parse_publish_month(value: object) -> str:
    """把发布时间解析到 YYYY-MM。"""
    text_value = safe_string(value)
    if not text_value:
        return ""
    parsed = pd.to_datetime(text_value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m")


def normalize_city(value: object) -> str:
    """轻量标准化广东城市。"""
    text_value = safe_string(value)
    if not text_value:
        return "未知"
    cities = [
        "深圳",
        "广州",
        "佛山",
        "东莞",
        "惠州",
        "珠海",
        "中山",
        "江门",
        "肇庆",
        "汕头",
        "湛江",
        "茂名",
        "韶关",
        "梅州",
        "清远",
        "阳江",
        "河源",
        "云浮",
        "潮州",
        "揭阳",
        "汕尾",
    ]
    for city in cities:
        if city in text_value:
            return city
    return "其他"


def normalize_industry(value: object) -> str:
    """轻量标准化行业。"""
    text_value = safe_string(value)
    if not text_value:
        return "未知"
    text_value = re.sub(r"[,，/、]+", ",", text_value)
    return text_value.split(",")[0].strip() or "未知"


def normalize_company_size(value: object) -> str:
    """轻量标准化公司规模。"""
    text_value = safe_string(value)
    if not text_value:
        return "未知"
    return text_value
