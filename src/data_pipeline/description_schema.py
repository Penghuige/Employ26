"""岗位描述解析的列名、表名和标题别名配置。"""

import re

PARSER_VERSION = "description_parsing_v1"
DEFAULT_PARSED_TABLE = "public.job_description_parsed"

JOB_DESCRIPTION_CLEAN_COL = "岗位描述_清洗"
DESCRIPTION_SECTIONS_JSON_COL = "岗位描述_切分JSON"
REQUIREMENTS_TEXT_COL = "任职要求_items_text"
DUTIES_TEXT_COL = "岗位职责_items_text"
UNCLASSIFIED_TEXT_COL = "unclassified_text"
SECTIONS_BRIEF_COL = "sections_brief"
RAG_QUERY_TEXT_COL = "RAG匹配文本"
RAG_QUERY_SOURCE_COL = "RAG匹配来源"

LEGACY_OUTPUT_COLUMNS = [
    JOB_DESCRIPTION_CLEAN_COL,
    DESCRIPTION_SECTIONS_JSON_COL,
    REQUIREMENTS_TEXT_COL,
    DUTIES_TEXT_COL,
    UNCLASSIFIED_TEXT_COL,
    SECTIONS_BRIEF_COL,
    RAG_QUERY_TEXT_COL,
    RAG_QUERY_SOURCE_COL,
]

LEGACY_TO_PG_COLUMN_MAP = {
    JOB_DESCRIPTION_CLEAN_COL: "job_description_clean",
    DESCRIPTION_SECTIONS_JSON_COL: "description_sections",
    REQUIREMENTS_TEXT_COL: "requirements_text",
    DUTIES_TEXT_COL: "duties_text",
    UNCLASSIFIED_TEXT_COL: "unclassified_text",
    SECTIONS_BRIEF_COL: "sections_brief",
    RAG_QUERY_TEXT_COL: "rag_query_text",
    RAG_QUERY_SOURCE_COL: "rag_query_source",
}

PG_COLUMN_ORDER = [
    "source_platform",
    "source_table",
    "source_row_number",
    "source_record_id",
    "job_title",
    "job_description_raw",
    "job_description_clean",
    "description_sections",
    "requirements_text",
    "duties_text",
    "unclassified_text",
    "sections_brief",
    "rag_query_text",
    "rag_query_source",
    "parser_version",
]

TITLE_ALIASES = {
    "岗位职责": [
        "岗位职责", "职责描述", "工作职责", "工作内容", "职位描述", "主要职责", "职责",
        "岗位描述", "岗位内容", "主要工作内容", "主要工作职责", "职位职责", "工作描述", "主要工作",
        "职责要求", "工作范围", "职责范围", "工作职能", "工作总责", "总责",
        "Responsibilities", "Responsibility", "Main Responsibilities", "Job Responsibilities",
        "Key Responsibilities", "Roles and Responsibilities", "Role and Responsibility",
        "Primary Responsibilities", "Primary and Secondary Responsibilities", "Job Description",
        "Position Objective", "Role Purpose", "What you will do", "Job Profile",
    ],
    "任职要求": [
        "任职要求", "岗位要求", "任职资格", "职位要求", "任职条件", "资格要求", "招聘要求",
        "技能要求", "工作要求", "能力要求", "基本要求", "人员要求", "任职资格要求", "职责要求",
        "要求", "应聘条件", "资质要求", "申请条件", "岗位条件", "候选人要求", "职位资质", "其他要求",
        "学历要求", "岗位资格", "职位资格", "具体要求", "专业要求", "工作技能", "技能",
        "Requirements", "Job Requirements", "Qualification", "Qualifications", "Candidate Profile",
    ],
    "福利待遇": [
        "福利待遇", "公司福利", "薪酬福利", "薪资待遇", "福利", "薪资福利", "员工福利", "待遇",
        "薪酬待遇", "员工福利", "职位福利", "待遇福利", "岗位福利", "福利保障", "薪资范围",
        "薪资福利待遇", "薪酬福利待遇", "薪资标准", "薪酬标准",
    ],
    "其他信息": [
        "工作地点", "上班地点", "工作时间", "上班时间", "联系方式", "联系地址", "工作地址", "公司地址",
        "地址", "时间", "备注", "加分项", "我们提供", "你将获得", "公司简介", "应聘方式",
        "住宿环境", "食堂", "宿舍", "附近地铁站", "附近公交站", "节日活动", "交通地址", "社保",
        "培训", "假期福利", "社会保障", "人文关怀", "职业规划", "薪资结构", "职能类别", "关键字",
        "关键词", "交通指引", "年龄要求", "联系人", "简历投递", "职位信息", "基本信息", "重要提示",
        "公司介绍", "项目背景介绍", "职业介绍", "校招岗位", "招聘岗位", "招聘岗位和专业", "公司特色",
        "About the Company", "Who we are", "Brand Introduction", "Position Title", "Job Title",
        "友情提醒", "提醒",
    ],
}

MIDLINE_EXCLUDE = {"福利", "待遇", "地址", "时间", "路线"}
OPTIONAL_NOTE_RE = r"(?:[（(][^\n:：)]{{1,12}}[)）])?"

ALIAS_TO_STD = sorted(
    [(alias, std) for std, aliases in TITLE_ALIASES.items() for alias in aliases],
    key=lambda x: len(x[0]),
    reverse=True,
)
MIDLINE_ALIASES = [
    alias for alias, _ in ALIAS_TO_STD
    if len(alias) >= 2 and alias not in MIDLINE_EXCLUDE
]
MIDLINE_PATTERN = "|".join(
    re.escape(x) for x in sorted(set(MIDLINE_ALIASES), key=len, reverse=True)
)
