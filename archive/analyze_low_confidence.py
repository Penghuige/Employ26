"""
分析低置信度样本，识别缺失的职业核心词
"""

import sys
import io
from pathlib import Path
import pandas as pd
from collections import Counter

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.job_title_parsing import OccupationParser

print("=" * 80)
print("低置信度样本分析")
print("=" * 80)

# 读取解析结果
results_file = Path('output/job_title_parsing/sample_parsing_results.csv')
df = pd.read_csv(results_file)

# 筛选低置信度样本
low_conf = df[df['置信度'] < 0.8]

print(f"\n低置信度样本数: {len(low_conf)} / {len(df)} ({len(low_conf)/len(df):.1%})")

# 统计核心词
core_counter = Counter(low_conf['职业核心词'])

print(f"\n低置信度样本的核心词分布（Top 30）:")
for i, (core, count) in enumerate(core_counter.most_common(30), 1):
    print(f"  {i:2d}. {core:20s}: {count:4d}")

# 识别需要添加的核心词
print("\n" + "=" * 80)
print("建议添加的职业核心词")
print("=" * 80)

# 分析模式
suggestions = {
    '技术员': ('技术员', 75, '生产类'),
    '代表': ('代表', 75, '销售类'),
    '司机': ('司机', 75, '其他类'),
    '电工': ('电工', 75, '生产类'),
    '负责人': ('负责人', 80, '管理类'),
    '仓管员': ('仓管员', 75, '采购类'),
    '检验员': ('检验员', 75, '生产类'),
    '质检员': ('质检员', 75, '生产类'),
    '操作员': ('操作员', 70, '生产类'),
    '调度员': ('调度员', 75, '其他类'),
    '跟单员': ('跟单员', 75, '其他类'),
    '采购员': ('采购员', 75, '采购类'),
    '统计员': ('统计员', 75, '其他类'),
    '资料员': ('资料员', 75, '其他类'),
    '绘图员': ('绘图员', 75, '设计类'),
    '营销代表': ('营销代表', 80, '销售类'),
    '销售代表': ('销售代表', 75, '销售类'),  # 已存在，但可能需要调整
    '主任': ('主任', 80, '管理类'),
    '班长': ('班长', 75, '管理类'),
    '组长': ('组长', 75, '管理类'),  # 已存在
    '线长': ('线长', 75, '生产类'),
    '车间主任': ('车间主任', 85, '生产类'),  # 已存在
}

print("\n建议添加到 occupation_cores.txt:")
print("\n# 生产/操作类（优先级 70-80）")
for word, (core, priority, category) in suggestions.items():
    if category in ['生产类', '其他类'] and word in ['技术员', '电工', '仓管员', '检验员', '质检员', '操作员', '线长']:
        print(f"{core} {priority} {category}")

print("\n# 销售类（优先级 75-80）")
for word, (core, priority, category) in suggestions.items():
    if category == '销售类' and word not in ['销售代表']:  # 销售代表已存在
        print(f"{core} {priority} {category}")

print("\n# 管理类（优先级 75-85）")
for word, (core, priority, category) in suggestions.items():
    if category == '管理类' and word not in ['组长', '车间主任']:  # 已存在的跳过
        print(f"{core} {priority} {category}")

print("\n# 其他职业类（优先级 70-80）")
for word, (core, priority, category) in suggestions.items():
    if word in ['司机', '调度员', '跟单员', '统计员', '资料员']:
        print(f"{core} {priority} {category}")

# 分析括号问题
print("\n" + "=" * 80)
print("括号处理问题分析")
print("=" * 80)

bracket_issues = low_conf[low_conf['职业核心词'].str.contains(r'[)）]', na=False)]
print(f"\n括号导致的错误: {len(bracket_issues)} 个")

if len(bracket_issues) > 0:
    print("\n示例:")
    for i, row in bracket_issues.head(10).iterrows():
        print(f"  {row['原始岗位名称']}")
        print(f"    -> 核心词: {row['职业核心词']}")

print("\n建议: 在解析前预处理，去除括号内容")

# 分析特殊岗位
print("\n" + "=" * 80)
print("特殊岗位分析")
print("=" * 80)

special_cases = low_conf[~low_conf['职业核心词'].str.contains(r'[)）员工师理监]', na=False)]
print(f"\n特殊岗位（不含常见后缀）: {len(special_cases)} 个")

if len(special_cases) > 0:
    print("\n示例:")
    for i, row in special_cases.head(20).iterrows():
        print(f"  {row['原始岗位名称']} -> {row['职业核心词']}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
