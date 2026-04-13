"""
更新职业核心词词典，修复评估中发现的问题
"""

import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dict_file = Path(__file__).parent.parent.parent / 'dicts' / 'occupation_cores.txt'

# 读取现有内容
with open(dict_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修改内容
new_lines = []
for line in lines:
    # 在算法研究员后添加工艺工程师、品质工程师、质量工程师
    if line.strip() == '算法研究员 110 技术类':
        new_lines.append(line)
        new_lines.append('工艺工程师 110 技术类\n')
        new_lines.append('品质工程师 110 技术类\n')
        new_lines.append('质量工程师 110 技术类\n')
    # 在市场总监后添加市场营销经理
    elif line.strip() == '市场总监 90 市场类':
        new_lines.append(line)
        new_lines.append('市场营销经理 90 市场类\n')
    # 调整售后客服优先级
    elif line.strip() == '售后客服 75 客服类':
        new_lines.append('售后客服 80 客服类\n')
    elif line.strip() == '在线客服 75 客服类':
        new_lines.append('在线客服 80 客服类\n')
    elif line.strip() == '电话客服 75 客服类':
        new_lines.append('电话客服 80 客服类\n')
    # 移除生产类中的品质工程师和工艺工程师（已移到技术类）
    elif line.strip() in ['品质工程师 90 生产类', '工艺工程师 90 生产类']:
        continue  # 跳过这两行
    else:
        new_lines.append(line)

# 写回文件
with open(dict_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("[OK] 词典已更新")
print("更新内容:")
print("  1. 添加 工艺工程师 (优先级110) 到技术类")
print("  2. 添加 品质工程师 (优先级110) 到技术类")
print("  3. 添加 质量工程师 (优先级110) 到技术类")
print("  4. 添加 市场营销经理 (优先级90) 到市场类")
print("  5. 提升 售后客服/在线客服/电话客服 优先级到80")
print("  6. 从生产类移除 品质工程师/工艺工程师（避免重复）")
