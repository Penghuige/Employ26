"""`public.recruitment_jobs_normalized` 的建表、指纹与写入逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
import uuid

from sqlalchemy import text

from src.db.postgres import create_pg_engine, ensure_schema


DEFAULT_NORMALIZED_TABLE = "public.recruitment_jobs_normalized"


def split_table_name(table_name: str) -> tuple[str, str]:
    """拆分 PostgreSQL 表名，支持 schema.table 与带双引号 schema。"""
    normalized = str(table_name).strip()
    if "." not in normalized:
        return "public", normalized.strip('"')
    schema, table = normalized.split(".", 1)
    return schema.strip().strip('"'), table.strip().strip('"')


def quote_table_name(table_name: str) -> str:
    """返回安全双引号包裹的 PostgreSQL 表名。"""
    schema_name, raw_table_name = split_table_name(table_name)
    return f'"{schema_name}"."{raw_table_name}"'


def safe_text(value: object) -> str:
    """安全转文本，统一去除空白。"""
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.lower() == "nan" else text_value


def build_dedupe_fingerprint(
    *,
    source_platform: str,
    company_name: str,
    job_title: str,
    job_description_raw: str,
    publish_date: str,
    work_city: str,
) -> str:
    """基于稳定字段构建内部去重指纹。"""
    parts = [
        safe_text(source_platform).casefold(),
        safe_text(company_name).casefold(),
        safe_text(job_title).casefold(),
        safe_text(job_description_raw).casefold(),
        safe_text(publish_date).casefold(),
        safe_text(work_city).casefold(),
    ]
    canonical = "||".join(parts)
    return sha1(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecruitmentNormalizedRow:
    """统一规范层单条招聘记录。"""

    source_platform: str
    source_table: str
    source_row_number: int
    source_native_job_id: str
    dedupe_fingerprint: str
    job_title: str
    job_description_raw: str
    work_city: str
    company_name: str
    publish_date: str
    salary_raw: str
    education_requirement_raw: str
    experience_requirement_raw: str
    company_size_raw: str
    company_industry_raw: str


def ensure_recruitment_jobs_normalized_table(
    connection,
    table_name: str = DEFAULT_NORMALIZED_TABLE,
) -> None:
    """确保统一规范层表存在，并补齐推荐索引。"""
    schema_name, raw_table_name = split_table_name(table_name)
    ensure_schema(connection, schema_name)
    qualified_table = quote_table_name(table_name)
    object_prefix = raw_table_name.replace('"', "").replace(".", "_").strip("_") or "recruitment_jobs_normalized"

    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                recruitment_record_id text PRIMARY KEY,
                source_platform text NOT NULL,
                source_table text NOT NULL,
                source_row_number bigint NOT NULL,
                source_native_job_id text NOT NULL DEFAULT '',
                dedupe_fingerprint text NOT NULL,
                job_title text,
                job_description_raw text,
                work_city text,
                company_name text,
                publish_date text,
                salary_raw text,
                education_requirement_raw text,
                experience_requirement_raw text,
                company_size_raw text,
                company_industry_raw text,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )
    for column_name in (
        "salary_raw",
        "education_requirement_raw",
        "experience_requirement_raw",
        "company_size_raw",
        "company_industry_raw",
    ):
        connection.execute(
            text(
                f"""
                ALTER TABLE {qualified_table}
                ADD COLUMN IF NOT EXISTS {column_name} text
                """
            )
        )
    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{object_prefix}_source_locator
            ON {qualified_table} (source_table, source_row_number)
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{object_prefix}_native_job
            ON {qualified_table} (source_platform, source_native_job_id)
            WHERE source_native_job_id <> ''
            """
        )
    )
    connection.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{object_prefix}_dedupe_fingerprint
            ON {qualified_table} (dedupe_fingerprint)
            """
        )
    )


def infer_source_platform(source_table: str) -> str:
    """根据来源表名推断招聘平台。"""
    lowered = str(source_table).lower()
    if "51job" in lowered or "qcwy" in lowered:
        return "51job"
    if "liepin" in lowered:
        return "Liepin"
    if "zhilian" in lowered:
        return "Zhilian"
    return ""


def build_normalized_rows_from_dataframe(
    dataframe,
    *,
    source_table: str,
    source_platform: str | None = None,
    title_col: str = "岗位名称",
    desc_col: str = "岗位描述",
    city_col: str = "工作城市",
    company_col: str = "公司名称",
    publish_col: str = "发布时间",
    native_job_id_col: str = "job_id",
    salary_col: str = "薪资水平",
    education_col: str = "学历要求",
    experience_col: str = "经验要求",
    company_size_col: str = "公司规模",
    company_industry_col: str = "公司行业",
) -> list[RecruitmentNormalizedRow]:
    """将原始招聘记录 DataFrame 转成统一规范层写入行。"""
    platform = source_platform or infer_source_platform(source_table)
    rows: list[RecruitmentNormalizedRow] = []
    for fallback_index, (_, row) in enumerate(dataframe.iterrows(), start=1):
        source_row_number = int(row.get("__source_row_number", fallback_index))
        source_native_job_id = safe_text(row.get(native_job_id_col, ""))
        job_title = safe_text(row.get(title_col, ""))
        job_description_raw = safe_text(row.get(desc_col, ""))
        work_city = safe_text(row.get(city_col, ""))
        company_name = safe_text(row.get(company_col, ""))
        publish_date = safe_text(row.get(publish_col, ""))
        salary_raw = safe_text(row.get(salary_col, ""))
        education_requirement_raw = safe_text(row.get(education_col, ""))
        experience_requirement_raw = safe_text(row.get(experience_col, ""))
        company_size_raw = safe_text(row.get(company_size_col, ""))
        company_industry_raw = safe_text(row.get(company_industry_col, ""))
        rows.append(
            RecruitmentNormalizedRow(
                source_platform=platform,
                source_table=source_table,
                source_row_number=source_row_number,
                source_native_job_id=source_native_job_id,
                dedupe_fingerprint=build_dedupe_fingerprint(
                    source_platform=platform,
                    company_name=company_name,
                    job_title=job_title,
                    job_description_raw=job_description_raw,
                    publish_date=publish_date,
                    work_city=work_city,
                ),
                job_title=job_title,
                job_description_raw=job_description_raw,
                work_city=work_city,
                company_name=company_name,
                publish_date=publish_date,
                salary_raw=salary_raw,
                education_requirement_raw=education_requirement_raw,
                experience_requirement_raw=experience_requirement_raw,
                company_size_raw=company_size_raw,
                company_industry_raw=company_industry_raw,
            )
        )
    return rows


def _find_existing_record_id_by_source_locator(connection, table_name: str, row: RecruitmentNormalizedRow) -> str | None:
    """按来源定位键查找已存在的招聘记录身份。"""
    qualified_table = quote_table_name(table_name)
    existing = connection.execute(
        text(
            f"""
            SELECT recruitment_record_id
            FROM {qualified_table}
            WHERE source_table = :source_table
              AND source_row_number = :source_row_number
            LIMIT 1
            """
        ),
        {
            "source_table": row.source_table,
            "source_row_number": row.source_row_number,
        },
    ).scalar_one_or_none()
    return str(existing) if existing else None


def _find_existing_record_id_by_entity_match(connection, table_name: str, row: RecruitmentNormalizedRow) -> str | None:
    """按原生岗位 ID 或去重指纹查找已存在的招聘记录身份。"""
    qualified_table = quote_table_name(table_name)
    if row.source_native_job_id:
        existing = connection.execute(
            text(
                f"""
                SELECT recruitment_record_id
                FROM {qualified_table}
                WHERE source_platform = :source_platform
                  AND source_native_job_id = :source_native_job_id
                LIMIT 1
                """
            ),
            {
                "source_platform": row.source_platform,
                "source_native_job_id": row.source_native_job_id,
            },
        ).scalar_one_or_none()
        if existing:
            return str(existing)

    existing = connection.execute(
        text(
            f"""
            SELECT recruitment_record_id
            FROM {qualified_table}
            WHERE dedupe_fingerprint = :dedupe_fingerprint
            LIMIT 1
            """
        ),
        {"dedupe_fingerprint": row.dedupe_fingerprint},
    ).scalar_one_or_none()
    return str(existing) if existing else None


def upsert_recruitment_jobs_normalized(
    rows: list[RecruitmentNormalizedRow],
    table_name: str = DEFAULT_NORMALIZED_TABLE,
    reuse_existing_identity: bool = False,
) -> int:
    """增量 upsert 统一规范层记录，并冻结 recruitment_record_id。"""
    if not rows:
        return 0

    qualified_table = quote_table_name(table_name)
    engine = create_pg_engine()
    with engine.begin() as connection:
        ensure_recruitment_jobs_normalized_table(connection, table_name=table_name)
        for row in rows:
            recruitment_record_id = _find_existing_record_id_by_source_locator(connection, table_name, row)
            if recruitment_record_id is None and reuse_existing_identity:
                recruitment_record_id = _find_existing_record_id_by_entity_match(connection, table_name, row)
            recruitment_record_id = recruitment_record_id or str(uuid.uuid4())
            connection.execute(
                text(
                    f"""
                    INSERT INTO {qualified_table} (
                        recruitment_record_id,
                        source_platform,
                        source_table,
                        source_row_number,
                        source_native_job_id,
                        dedupe_fingerprint,
                        job_title,
                        job_description_raw,
                        work_city,
                        company_name,
                        publish_date,
                        salary_raw,
                        education_requirement_raw,
                        experience_requirement_raw,
                        company_size_raw,
                        company_industry_raw
                    )
                    VALUES (
                        :recruitment_record_id,
                        :source_platform,
                        :source_table,
                        :source_row_number,
                        :source_native_job_id,
                        :dedupe_fingerprint,
                        :job_title,
                        :job_description_raw,
                        :work_city,
                        :company_name,
                        :publish_date,
                        :salary_raw,
                        :education_requirement_raw,
                        :experience_requirement_raw,
                        :company_size_raw,
                        :company_industry_raw
                    )
                    ON CONFLICT (recruitment_record_id)
                    DO UPDATE SET
                        source_platform = EXCLUDED.source_platform,
                        source_table = EXCLUDED.source_table,
                        source_row_number = EXCLUDED.source_row_number,
                        source_native_job_id = EXCLUDED.source_native_job_id,
                        dedupe_fingerprint = EXCLUDED.dedupe_fingerprint,
                        job_title = EXCLUDED.job_title,
                        job_description_raw = EXCLUDED.job_description_raw,
                        work_city = EXCLUDED.work_city,
                        company_name = EXCLUDED.company_name,
                        publish_date = EXCLUDED.publish_date,
                        salary_raw = EXCLUDED.salary_raw,
                        education_requirement_raw = EXCLUDED.education_requirement_raw,
                        experience_requirement_raw = EXCLUDED.experience_requirement_raw,
                        company_size_raw = EXCLUDED.company_size_raw,
                        company_industry_raw = EXCLUDED.company_industry_raw,
                        updated_at = now()
                    """
                ),
                {
                    "recruitment_record_id": recruitment_record_id,
                    "source_platform": row.source_platform,
                    "source_table": row.source_table,
                    "source_row_number": row.source_row_number,
                    "source_native_job_id": row.source_native_job_id,
                    "dedupe_fingerprint": row.dedupe_fingerprint,
                    "job_title": row.job_title,
                    "job_description_raw": row.job_description_raw,
                    "work_city": row.work_city,
                    "company_name": row.company_name,
                    "publish_date": row.publish_date,
                    "salary_raw": row.salary_raw,
                    "education_requirement_raw": row.education_requirement_raw,
                    "experience_requirement_raw": row.experience_requirement_raw,
                    "company_size_raw": row.company_size_raw,
                    "company_industry_raw": row.company_industry_raw,
                },
            )
    return len(rows)


def load_normalized_jobs_dataframe(table_name: str = DEFAULT_NORMALIZED_TABLE):
    """读取统一规范层表为 DataFrame。"""
    qualified_table = quote_table_name(table_name)
    engine = create_pg_engine()
    try:
        with engine.connect() as connection:
            import pandas as pd

            return pd.read_sql_query(text(f"SELECT * FROM {qualified_table}"), connection)
    finally:
        engine.dispose()
