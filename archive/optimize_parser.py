"""
优化解析器：
1. 添加括号预处理
2. 补充缺失的职业核心词
"""

import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("优化解析器")
print("=" * 80)

# 1. 更新词典
dict_file = Path('dicts/occupation_cores.txt')

with open(dict_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到生产类的位置，添加新词
new_lines = []
added_production = False
added_sales = False
added_management = False
added_other = False

for line in lines:
    new_lines.append(line)
    
    # 在生产类的技师后添加
    if line.strip() == '技师 80 生产类' and not added_production:
        new_lines.append('技术员 75 生产类\n')
        new_lines.append('电工 75 生产类\n')
        new_lines.append('检验员 75 生产类\n')
        new_lines.append('操作员 70 生产类\n')
        new_lines.append('线长 75 生产类\n')
        new_lines.append('仓管员 75 生产类\n')
        new_lines.append('维修工 75 生产类\n')
        new_lines.append('装配工 75 生产类\n')
        new_lines.append('普工 70 生产类\n')
        new_lines.append('施工员 75 生产类\n')
        added_production = True
        print("[OK] 添加生产类职业核心词")
    
    # 在销售类的销售后添加
    if line.strip() == '销售 70 销售类' and not added_sales:
        new_lines.append('代表 75 销售类\n')
        new_lines.append('营销代表 80 销售类\n')
        new_lines.append('医药代表 85 销售类\n')
        added_sales = True
        print("[OK] 添加销售类职业核心词")
    
    # 在管理类的部长后添加
    if line.strip() == '部长 90 管理类' and not added_management:
        new_lines.append('负责人 80 管理类\n')
        new_lines.append('主任 80 管理类\n')
        new_lines.append('班长 75 管理类\n')
        added_management = True
        print("[OK] 添加管理类职业核心词")
    
    # 在其他类的学徒后添加
    if line.strip() == '学徒 65 其他类' and not added_other:
        new_lines.append('司机 75 其他类\n')
        new_lines.append('调度员 75 其他类\n')
        new_lines.append('跟单员 75 其他类\n')
        new_lines.append('统计员 75 其他类\n')
        new_lines.append('资料员 75 其他类\n')
        new_lines.append('店员 70 其他类\n')
        new_lines.append('门卫 70 其他类\n')
        new_lines.append('保安 70 其他类\n')
        new_lines.append('主播 75 其他类\n')
        added_other = True
        print("[OK] 添加其他类职业核心词")

# 写回文件
with open(dict_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"\n[OK] 词典已更新: {dict_file}")

# 2. 优化解析器，添加括号预处理
parser_file = Path('src/job_title_parsing/occupation_parser.py')

with open(parser_file, 'r', encoding='utf-8') as f:
    parser_code = f.read()

# 检查是否已经有括号预处理
if '_preprocess_job_title' not in parser_code:
    print("\n[INFO] 需要手动添加括号预处理功能到 occupation_parser.py")
    print("\n建议在 parse() 方法开头添加:")
    print("""
    def _preprocess_job_title(self, job_title: str) -> str:
        \"\"\"预处理岗位名称，去除括号内容\"\"\"
        import re
        # 去除括号及其内容
        job_title = re.sub(r'[（(].*?[）)]', '', job_title)
        # 去除多余空格
        job_title = ' '.join(job_title.split())
        return job_title.strip()
    
    # 在 parse() 方法中调用:
    job_title = self._preprocess_job_title(job_title)
    """)
else:
    print("\n[OK] 括号预处理功能已存在")

print("\n" + "=" * 80)
print("优化完成")
print("=" * 80)

print("\n下一步:")
print("  1. 重新运行测试: python test_on_samples.py")
print("  2. 查看覆盖率是否提升")
