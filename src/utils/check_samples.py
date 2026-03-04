import os
import pandas as pd
from pathlib import Path

# 查看samples目录
samples_dir = Path('../../output/samples')

print("=" * 60)
print("样本文件列表:")
print("=" * 60)

for file in samples_dir.glob('*.csv'):
    print(f"\n文件: {file.name}")
    try:
        # 读取文件统计信息
        df = pd.read_csv(file, nrows=0)
        total_lines = sum(1 for _ in open(file, 'r', encoding='utf-8')) - 1
        print(f"  总行数: {total_lines:,}")
        print(f"  列数: {len(df.columns)}")
        print(f"  文件大小: {file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 读取前3行
        df_sample = pd.read_csv(file, nrows=3)
        print(f"  前3行预览:")
        print(df_sample[['发布时间', '岗位名称', '工作城市', '薪资水平']].to_string(index=False))
    except Exception as e:
        print(f"  错误: {e}")

print("\n" + "=" * 60)

