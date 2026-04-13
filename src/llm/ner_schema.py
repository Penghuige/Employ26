# -*- coding: utf-8 -*-
"""
NER 实体标注 Schema 定义

定义从岗位描述（jd_snippet）中需要提取的所有实体类型及其说明。

注意：以下字段已在原始数据集中存在，本模块不重复提取：
    发布时间、岗位名称、工作城市、薪资水平、经验要求、学历要求、
    岗位描述、公司名称、公司规模、公司行业
"""

# ============================================================
# NER 实体类型定义
# ============================================================

NER_ENTITY_SCHEMA = {
    "SKILL": {
        "label": "专业技能",
        "description": "岗位所需的专业能力，包括技术技能和业务技能",
        "examples": [
            "数据分析", "财务核算", "项目管理", "机器学习",
            "税务申报", "市场推广", "供应链管理"
        ],
        "bio_prefix": "SKILL",
        "research_value": "构建职业技能图谱，分析技能-薪资溢价关系，识别高需求技能"
    },
    "TOOL": {
        "label": "工具/软件/框架",
        "description": "具体的软件产品、编程语言、开发框架、平台名称",
        "examples": [
            "Python", "SQL", "Excel", "Tableau", "PyTorch",
            "SAP", "AutoCAD", "Git", "Photoshop", "Hadoop"
        ],
        "bio_prefix": "TOOL",
        "research_value": "分析行业工具偏好，追踪技术栈演变趋势，研究工具与薪资关系"
    },
    "CERT": {
        "label": "证书/资质/执照",
        "description": "岗位要求或优先考虑的职业资格证书、执照、认证",
        "examples": [
            "CPA", "CFA", "PMP", "驾照", "教师资格证",
            "注册会计师", "建造师证", "消防工程师证"
        ],
        "bio_prefix": "CERT",
        "research_value": "研究职业准入门槛，分析证书对薪资的影响，支持职业规划指导"
    },
    "BENEFIT": {
        "label": "福利待遇",
        "description": "岗位提供的薪资以外的福利、补贴、激励项目",
        "examples": [
            "五险一金", "带薪年假", "股票期权", "年终奖",
            "餐补", "弹性工时", "员工宿舍", "商业保险"
        ],
        "bio_prefix": "BNF",
        "research_value": "评估就业质量，分析不同行业/城市福利差异，建立非薪资福利指数"
    },
    "DUTY": {
        "label": "核心工作职责",
        "description": "岗位的主要工作内容和职责描述（动词短语为主）",
        "examples": [
            "负责产品需求分析", "协助市场调研", "主导项目推进",
            "编写技术文档", "统筹团队协作"
        ],
        "bio_prefix": "DUTY",
        "research_value": "构建岗位职能画像，分析岗位职责差异与薪资关系，支持岗位评价"
    },
    "HEADCOUNT": {
        "label": "招聘人数",
        "description": "本次招聘的人员数量",
        "examples": ["1人", "若干", "1-3名", "多名"],
        "bio_prefix": "HC",
        "research_value": "量化劳动力需求规模，分析岗位类型与招聘规模的关系"
    },
    "JOB_TYPE": {
        "label": "工作性质",
        "description": "岗位的雇佣形式",
        "examples": ["全职", "兼职", "实习", "合同制", "劳务派遣"],
        "bio_prefix": "JT",
        "research_value": "研究就业形式结构变化，分析非标准就业增长趋势"
    }
}

# ============================================================
# BIO 标注格式说明
# ============================================================
# 采用 BIO (Beginning-Inside-Outside) 标注体系：
#   B-SKILL  : 技能实体的起始词
#   I-SKILL  : 技能实体的内部词
#   B-TOOL   : 工具实体的起始词
#   I-TOOL   : 工具实体的内部词
#   B-CERT   : 证书实体的起始词
#   I-CERT   : 证书实体的内部词
#   B-BNF    : 福利实体的起始词
#   I-BNF    : 福利实体的内部词
#   B-DUTY   : 职责实体的起始词
#   I-DUTY   : 职责实体的内部词
#   B-HC     : 人数实体的起始词
#   I-HC     : 人数实体的内部词
#   B-JT     : 工作性质实体的起始词
#   I-JT     : 工作性质实体的内部词
#   O        : 非实体词

BIO_LABELS = ["O"]
for entity_type, info in NER_ENTITY_SCHEMA.items():
    prefix = info["bio_prefix"]
    BIO_LABELS.append(f"B-{prefix}")
    BIO_LABELS.append(f"I-{prefix}")

LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(BIO_LABELS)}


if __name__ == "__main__":
    print("NER 实体类型列表：")
    for k, v in NER_ENTITY_SCHEMA.items():
        print(f"  [{k}] {v['label']}: {v['description']}")
    print(f"\nBIO 标签集合（共 {len(BIO_LABELS)} 个）:")
    print(BIO_LABELS)
