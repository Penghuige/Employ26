"""
在真实样本数据上测试岗位名称解析器
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
print("真实样本数据测试")
print("=" * 80)

# 创建解析器
parser = OccupationParser()

# 读取样本数据
sample_dir = Path('output/samples')
sample_files = list(sample_dir.glob('*样本_1%.csv'))

print(f"\n找到 {len(sample_files)} 个样本文件:")
for f in sample_files:
    print(f"  - {f.name}")

all_results = []
all_job_titles = []

for csv_file in sample_files:
    print(f"\n处理: {csv_file.name}")
    
    try:
        # 读取数据
        df = pd.read_csv(csv_file, nrows=1000)  # 每个文件读取1000条
        
        if '岗位名称' in df.columns:
            job_titles = df['岗位名称'].dropna().tolist()
            print(f"  读取岗位名称: {len(job_titles)} 条")
            
            # 解析
            results = parser.parse_batch(job_titles)
            all_results.extend(results)
            all_job_titles.extend(job_titles)
            
        else:
            print(f"  [警告] 未找到'岗位名称'字段")
            
    except Exception as e:
        print(f"  [错误] {e}")

print("\n" + "=" * 80)
print(f"解析统计（共 {len(all_results)} 条）")
print("=" * 80)

# 统计分析
stats = parser.get_statistics(all_results)

print(f"\n覆盖率（置信度≥0.8）: {stats['coverage']:.2%}")

print(f"\n置信度分布:")
for conf_level, count in stats['confidence_distribution'].items():
    pct = count / stats['total'] * 100
    bar = '█' * int(pct / 2)
    print(f"  {conf_level:20s}: {count:5d} ({pct:5.1f}%) {bar}")

print(f"\n类别分布（Top 10）:")
sorted_categories = sorted(stats['category_distribution'].items(), key=lambda x: -x[1])
for i, (category, count) in enumerate(sorted_categories[:10], 1):
    pct = count / stats['total'] * 100
    bar = '█' * int(pct / 2)
    print(f"  {i:2d}. {category:15s}: {count:5d} ({pct:5.1f}%) {bar}")

print(f"\n匹配方法分布:")
for method, count in sorted(stats['method_distribution'].items(), key=lambda x: -x[1]):
    pct = count / stats['total'] * 100
    print(f"  {method:20s}: {count:5d} ({pct:5.1f}%)")

print(f"\nTop 20 职业核心词:")
for i, (core, count) in enumerate(stats['top_cores'][:20], 1):
    pct = count / stats['total'] * 100
    print(f"  {i:2d}. {core:20s}: {count:5d} ({pct:4.1f}%)")

print(f"\nTop 30 修饰词:")
for i, (mod, count) in enumerate(stats['top_modifiers'][:30], 1):
    print(f"  {i:2d}. {mod:20s}: {count:5d}")

# 显示一些示例
print("\n" + "=" * 80)
print("解析示例（随机抽取20个）")
print("=" * 80)

import random
sample_indices = random.sample(range(len(all_results)), min(20, len(all_results)))

for i, idx in enumerate(sample_indices, 1):
    result = all_results[idx]
    
    # 根据置信度显示标记
    if result['confidence'] == 1.0:
        status = "[✓]"
    elif result['confidence'] >= 0.8:
        status = "[~]"
    else:
        status = "[?]"
    
    print(f"\n{i:2d}. {status} {result['job_title_raw']}")
    print(f"     核心词: {result['occupation_core']:20s} 类别: {result['core_category']}")
    if result['modifiers']:
        print(f"     修饰词: {result['modifiers']}")

# 找出低置信度样本
print("\n" + "=" * 80)
print("低置信度样本（置信度<0.8，最多显示20个）")
print("=" * 80)

low_conf_results = [r for r in all_results if r['confidence'] < 0.8]
print(f"\n共 {len(low_conf_results)} 个低置信度样本 ({len(low_conf_results)/len(all_results):.1%})")

if low_conf_results:
    print("\n示例:")
    for i, result in enumerate(low_conf_results[:20], 1):
        print(f"\n{i:2d}. {result['job_title_raw']}")
        print(f"     核心词: {result['occupation_core']}")
        print(f"     置信度: {result['confidence']:.1f}")
        print(f"     方法: {result['match_method']}")

# 保存结果
output_file = Path('output/job_title_parsing/sample_parsing_results.csv')
output_file.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame([
    {
        '原始岗位名称': r['job_title_raw'],
        '职业核心词': r['occupation_core'],
        '修饰词': r['modifiers'],
        '类别': r['core_category'],
        '置信度': r['confidence'],
        '匹配方法': r['match_method']
    }
    for r in all_results
])

results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n结果已保存: {output_file}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

# 评估总结
print(f"\n评估总结:")
print(f"  总样本数: {len(all_results)}")
print(f"  覆盖率: {stats['coverage']:.2%}")
print(f"  高置信度（1.0）: {stats['confidence_distribution']['高置信(1.0)']} ({stats['confidence_distribution']['高置信(1.0)']/len(all_results):.1%})")
print(f"  中置信度（0.8）: {stats['confidence_distribution']['中置信(0.8)']} ({stats['confidence_distribution']['中置信(0.8)']/len(all_results):.1%})")
print(f"  低置信度（<0.8）: {len(low_conf_results)} ({len(low_conf_results)/len(all_results):.1%})")

if stats['coverage'] >= 0.95:
    print(f"\n✅ 优秀！覆盖率达到 {stats['coverage']:.1%}")
elif stats['coverage'] >= 0.90:
    print(f"\n✓ 良好！覆盖率达到 {stats['coverage']:.1%}")
else:
    print(f"\n⚠ 需要改进，覆盖率仅 {stats['coverage']:.1%}")
