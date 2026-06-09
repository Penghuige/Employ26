import json

import pandas as pd

from src.data_pipeline.description_parsing import (
    PARSER_VERSION,
    build_parsed_pg_rows,
    parse_desc_df,
)


def test_parse_desc_df_extracts_explicit_requirements_and_duties():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "Java开发工程师",
                "岗位描述": "<p>岗位职责：1. 负责后端开发；2. 维护系统</p><p>任职要求：熟悉 Java；掌握 MySQL</p>",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "负责后端开发" in parsed.loc[0, "岗位职责_items_text"]
    assert "熟悉 Java" in parsed.loc[0, "任职要求_items_text"]
    assert parsed.loc[0, "RAG匹配来源"] == "任职要求"


def test_parse_desc_df_infers_section_when_no_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "数据分析师",
                "岗位描述": "本科以上学历，熟练使用 SQL 和 Python，有数据分析经验优先。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "SQL" in parsed.loc[0, "任职要求_items_text"]
    assert parsed.loc[0, "RAG匹配文本"]


def test_build_parsed_pg_rows_uses_english_columns_and_json_payload():
    df = pd.DataFrame(
        [
            {
                "recruitment_record_id": "rrid-0007",
                "__source_row_number": 7,
                "岗位名称": "测试工程师",
                "岗位描述": "任职要求：熟悉自动化测试。",
            }
        ]
    )
    parsed = parse_desc_df(df)

    rows = build_parsed_pg_rows(parsed, source_table='"51job".sample')

    assert rows[0]["recruitment_record_id"] == "rrid-0007"
    assert rows[0]["source_platform"] == "51job"
    assert rows[0]["source_row_number"] == 7
    assert rows[0]["job_title"] == "测试工程师"
    assert rows[0]["requirements_text"]
    assert rows[0]["parser_version"] == PARSER_VERSION
    assert json.loads(str(rows[0]["description_sections"]))["sections"]


def test_parse_desc_df_splits_platform_advertisement_headings():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "门店兼职",
                "岗位描述": "【工作薪资】23元/小时【工作要求】每周至少出勤3天【工作岗位】点单，迎宾，前厅服务【上班时间】8:30-23:30",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "每周至少出勤3天" in parsed.loc[0, "任职要求_items_text"]
    assert "点单" in parsed.loc[0, "岗位职责_items_text"]
    assert "23元/小时" not in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_keeps_benefits_out_of_duties():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "房产销售",
                "岗位描述": "岗位职责：1、网络直播给客户看房子；2、商业谈判促进成交。更多员工福利：五险、节日关怀礼、年假。任职资格：沟通能力强。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "网络直播给客户看房子" in parsed.loc[0, "岗位职责_items_text"]
    assert "更多员工福利" not in parsed.loc[0, "岗位职责_items_text"]
    assert "沟通能力强" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_splits_embedded_requirement_heading_inside_duty_item():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "珠宝导购",
                "岗位描述": "岗位职责：主动为顾客介绍珠宝饰品，完成销售任务任职要求：年龄22-36岁，有珠宝导购经验优先。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "完成销售任务" in parsed.loc[0, "岗位职责_items_text"]
    assert "年龄22-36岁" in parsed.loc[0, "任职要求_items_text"]
    assert "任职要求" not in parsed.loc[0, "岗位职责_items_text"]


def test_parse_desc_df_splits_embedded_job_requirement_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "茶饮师",
                "岗位描述": "岗位职责：1.茶饮品的调配与制作2.接待顾客岗位要求：1.年龄18-30岁2.身体健康3.经验不限",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "茶饮品的调配与制作" in parsed.loc[0, "岗位职责_items_text"]
    assert "年龄18-30岁" in parsed.loc[0, "任职要求_items_text"]
    assert "岗位要求" not in parsed.loc[0, "岗位职责_items_text"]


def test_parse_desc_df_splits_salary_requirement_and_duty_ad_text():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "配送员",
                "岗位描述": "薪资待遇：月薪8000，包吃住。职位要求：年龄18-45岁，经验不限。工作内容：骑两轮电动车负责仓库配送。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "年龄18-45岁" in parsed.loc[0, "任职要求_items_text"]
    assert "负责仓库配送" in parsed.loc[0, "岗位职责_items_text"]
    assert "月薪8000" not in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_splits_hidden_benefits_from_requirements():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "茶饮师",
                "岗位描述": "招聘要求:年龄18-44周岁，经验不限，其他福利:宿舍2-4人间，有空调，节假日礼品。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "年龄18-44周岁" in parsed.loc[0, "任职要求_items_text"]
    assert "宿舍2-4人间" not in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_handles_implicit_factory_worker_text():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "分拣打包理货员",
                "岗位描述": "男女不限，年龄18-45岁，认识26个英文字母会使用电脑认真细心，吃苦耐劳，服从管理安排。工作内容：小物件打包分拣，扫描，贴标签。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "年龄18-45岁" in parsed.loc[0, "任职要求_items_text"]
    assert "打包分拣" in parsed.loc[0, "岗位职责_items_text"]


def test_parse_desc_df_recognizes_application_requirement_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "封装基板设计工程师",
                "岗位描述": "职位职责：1、负责基板版图设计。应聘要求：1、本科及以上学历，具备3年以上基板设计经验。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "负责基板版图设计" in parsed.loc[0, "岗位职责_items_text"]
    assert "本科及以上学历" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_recognizes_question_style_headings():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "财务会计专员",
                "岗位描述": "我们需要你做什么?1、独立处理日常财务账务；2、负责正确计算收入。我们希望你是什么样的人?1、1年以上相关工作经验；2、有较好的沟通协调能力。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "独立处理日常财务账务" in parsed.loc[0, "岗位职责_items_text"]
    assert "1年以上相关工作经验" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_recognizes_prefixed_job_description_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "物流技师",
                "岗位描述": "物流员岗位描述：1.负责百检仓各项数据记录、维护。岗位要求：1.需要有叉车证。2.有一定沟通能力。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "负责百检仓各项数据记录" in parsed.loc[0, "岗位职责_items_text"]
    assert "需要有叉车证" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_keeps_company_service_out_of_requirements():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "网约车司机",
                "岗位描述": "工作内容：专职网约车司机。职位要求：1、年龄21-55岁，身体健康。公司服务：1、公司免费统一办理网约车证。2、提供专业团队支持。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "专职网约车司机" in parsed.loc[0, "岗位职责_items_text"]
    assert "年龄21-55岁" in parsed.loc[0, "任职要求_items_text"]
    assert "网约车证" not in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_classifies_ability_sentence_as_requirement():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "博士后",
                "岗位描述": "岗位职责：开展科研项目。任职要求：具有较强的研究能力，具备复合专业背景者优先。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "开展科研项目" in parsed.loc[0, "岗位职责_items_text"]
    assert "具有较强的研究能力" in parsed.loc[0, "任职要求_items_text"]
    assert "具有较强的研究能力" not in parsed.loc[0, "岗位职责_items_text"]


def test_parse_desc_df_recognizes_numbered_heading_without_colon():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "人事助理",
                "岗位描述": "一.工作内容1.在招聘平台上发布招聘信息，筛选简历。二.薪资待遇3000元/月。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "发布招聘信息" in parsed.loc[0, "岗位职责_items_text"]
    assert "3000元/月" not in parsed.loc[0, "岗位职责_items_text"]


def test_parse_desc_df_recognizes_recruitment_need_as_requirement():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "跟单",
                "岗位描述": "高薪招聘需求：有从事过不干胶行业的专业人员，熟练不干胶报价、工艺、材料等。福利：8-10K，包吃包住。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "不干胶行业" in parsed.loc[0, "任职要求_items_text"]
    assert "8-10K" not in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_recognizes_required_condition_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "电子销售",
                "岗位描述": "工作内容：拜访客户，维护客户关系。必须条件：日语商务水平，英语沟通，熟练操作电脑。优先条件：贸易公司销售经验优先。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "拜访客户" in parsed.loc[0, "岗位职责_items_text"]
    assert "日语商务水平" in parsed.loc[0, "任职要求_items_text"]
    assert "贸易公司销售经验" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_separates_salary_range_after_requirements():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "渠道经理",
                "岗位描述": "岗位职责：负责项目资源开发和维护。任职要求：具备出色的沟通能力，抗压能力强。薪酬区间：固定薪资绩效工资业绩提成。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "负责项目资源开发" in parsed.loc[0, "岗位职责_items_text"]
    assert "具备出色的沟通能力" in parsed.loc[0, "任职要求_items_text"]
    assert "固定薪资" not in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_infers_delivery_duty_without_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "配送员",
                "岗位描述": "岗位直招，全市可安排家附近上班，负责小件物品配送，日薪350-480，接受短期过渡，接受兼职。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "小件物品配送" in parsed.loc[0, "岗位职责_items_text"]
    assert "接受短期" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_keeps_duties_after_location_prefix():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "机械设计工程师",
                "岗位描述": "上班地点：东莞虎门1、主要负责非标设备的设计工作；2、根据客户要求制作设计图纸；岗位要求：专科以上学历。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "非标设备的设计工作" in parsed.loc[0, "岗位职责_items_text"]
    assert "专科以上学历" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_infers_driver_duty_without_heading():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "小货车司机",
                "岗位描述": "主要跑东莞-深圳周边，短途配送，拉百货快消品，不用装货，要求:熟练驾驶小货车，身体健康。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "短途配送" in parsed.loc[0, "岗位职责_items_text"]
    assert "熟练驾驶小货车" in parsed.loc[0, "任职要求_items_text"]


def test_parse_desc_df_recognizes_japanese_adoption_requirement():
    df = pd.DataFrame(
        [
            {
                "岗位名称": "日语外贸业务员",
                "岗位描述": "工作内容：负责日本客户订单跟进。採用要求：日语熟练，有外贸经验优先。",
            }
        ]
    )

    parsed = parse_desc_df(df)

    assert "日本客户订单跟进" in parsed.loc[0, "岗位职责_items_text"]
    assert "日语熟练" in parsed.loc[0, "任职要求_items_text"]
