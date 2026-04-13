"""
岗位名称解析 - 快速演示
"""

import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.job_title_parsing import OccupationParser

print("=" * 80)
print("岗位名称解析 - 快速演示")
print("=" * 80)

# 创建解析器
parser = OccupationParser()

# 演示样例
demo_cases = [
    "Java开发工程师",
    "高级Python后端开发工程师",
    "产品经理",
    "高级产品经理",
    "UI设计师",
    "数据分析师",
    "销售总监",
    "薄膜工艺工程师",
    "CEO",
    "市场营销经理"
]

print("\n解析结果:\n")

for i, job_title in enumerate(demo_cases, 1):
    result = parser.parse(job_title)
    
    print(f"{i:2d}. 原始: {result['job_title_raw']}")
    print(f"    核心词: {result['occupation_core']}")
    print(f"    修饰词: {result['modifiers']}")
    print(f"    类别: {result['core_category']}")
    print(f"    置信度: {result['confidence']}")
    print()

print("=" * 80)
print("演示完成！")
print("=" * 80)
