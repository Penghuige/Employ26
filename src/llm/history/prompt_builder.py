# -*- coding: utf-8 -*-
"""
Prompt 构建模块

为 Qwen3 本地大模型构建结构化信息抽取 Prompt。
支持零样本（zero-shot）和少样本（few-shot）两种模式。
"""

import json
from typing import Optional


# ============================================================
# 系统提示词
# ============================================================
SYSTEM_PROMPT = """你是一个专业的招聘信息结构化抽取专家。
你的任务是从岗位描述文本中抽取以下7类实体信息，以 JSON 格式输出。

【实体类型说明】
- skills   : 专业技能（如 数据分析、财务核算、项目管理）
- tools    : 工具/软件/框架/平台（如 Tableau、PyTorch、SAP、Git）
- certs    : 证书/资质/执照（如 CPA、PMP、驾照、教师资格证）
- benefits : 福利待遇（如 五险一金、带薪年假、股票期权、餐补）
- duties   : 核心工作职责（动词短语，不超过5条，每条不超过20字）
- headcount: 招聘人数（如 1人、若干、1-3名）
- job_type : 工作性质（如 全职、兼职、实习、合同制）

【注意事项】
1. 只提取岗位描述中明确出现的内容，不要推测或补充。
2. skills 和 tools 要严格区分：skills 是能力描述（如"数据分析能力"），
   tools 是具体产品/语言名称（如"Tableau"、"Python"）。
3. 若某类实体不存在，对应字段返回空列表 []。
4. headcount 和 job_type 若无明确表述，返回 null。
5. 严格输出合法 JSON，不要有任何额外解释或 markdown 标记。
"""

# ============================================================
# 少样本示例
# ============================================================
FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "负责公司财务报表编制及分析，熟练使用Excel、SAP系统，"
            "具备CPA证书优先，熟悉税务申报流程，有上市公司经验者优先。"
            "待遇：五险一金+年终奖+带薪年假，招聘1名，全职。"
        ),
        "output": {
            "skills": ["财务报表编制", "财务分析", "税务申报"],
            "tools": ["Excel", "SAP"],
            "certs": ["CPA"],
            "benefits": ["五险一金", "年终奖", "带薪年假"],
            "duties": ["负责公司财务报表编制及分析"],
            "headcount": "1名",
            "job_type": "全职"
        }
    },
    {
        "input": (
            "1.负责制定风险管理政策及制度；2.协同产品部门评审贷款项目；"
            "3.运用SQL、Python制定风险策略并持续优化；"
            "要求本科及以上，具有2年以上金融风控经验，"
            "掌握SQL、SAS、Python任意一种。提供六险一金、股票期权。"
        ),
        "output": {
            "skills": ["风险管理", "数据分析", "风险策略制定"],
            "tools": ["SQL", "Python", "SAS"],
            "certs": [],
            "benefits": ["六险一金", "股票期权"],
            "duties": ["负责制定风险管理政策及制度", "协同产品部门评审贷款项目", "制定并优化风险策略"],
            "headcount": None,
            "job_type": None
        }
    }
]


def build_prompt(
    jd_text: str,
    mode: str = "few_shot",
    thinking: bool = True
) -> list[dict]:
    """
    构建发送给 Qwen3 的消息列表。

    Args:
        jd_text  : 岗位描述文本
        mode     : 'zero_shot' 或 'few_shot'
        thinking : 是否启用 Qwen3 的思维链（/think 模式）

    Returns:
        messages : OpenAI 格式的消息列表
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if mode == "few_shot":
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)})

    # 添加当前待处理文本
    # Qwen3 支持 /think 和 /no_think 指令控制推理模式
    user_content = jd_text
    if thinking:
        user_content = "/think\n" + jd_text
    else:
        user_content = "/no_think\n" + jd_text

    messages.append({"role": "user", "content": user_content})
    return messages


if __name__ == "__main__":
    sample_jd = (
        "负责公司流水线产品的销售，完成销售任务；区域内开拓渠道、维护终端；"
        "负责产品推广、信息收集、工作汇报；base广州，接受省外出差。"
        "任职资格：本科，医学检验、生物工程、临床医学等相关专业；"
        "二年相关工作经验，有诊断试剂工作经验优先；善沟通表达、思路清楚。"
    )
    msgs = build_prompt(sample_jd, mode="few_shot", thinking=False)
    for m in msgs:
        print(f"[{m['role']}]\n{m['content']}\n")
