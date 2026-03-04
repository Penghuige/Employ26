"""
样本数据提取模块
从三个原始数据文件中按比例均匀提取10%的数据作为样本
"""

import pandas as pd
import os
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_lines(filepath):
    """快速统计文件行数"""
    logger.info(f"正在统计文件行数: {filepath}")
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count - 1  # 减去表头


def extract_sample_uniform(input_file, output_file, sample_rate=0.1):
    """
    均匀间隔提取样本数据
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_rate: 采样比例，默认0.1（10%）
    """
    logger.info(f"开始处理: {input_file}")
    
    # 统计总行数
    total_lines = count_lines(input_file)
    logger.info(f"总行数: {total_lines}")
    
    # 计算采样间隔
    interval = int(1 / sample_rate)
    expected_samples = total_lines // interval
    logger.info(f"采样间隔: {interval}, 预计采样: {expected_samples} 行")
    
    # 分块读取并均匀采样
    chunksize = 10000
    sampled_data = []
    row_count = 0
    sampled_count = 0
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize, encoding='utf-8', low_memory=False):
        for idx in range(len(chunk)):
            if row_count % interval == 0:
                sampled_data.append(chunk.iloc[idx])
                sampled_count += 1
                
                # 每采样10000行输出一次进度
                if sampled_count % 10000 == 0:
                    logger.info(f"已采样: {sampled_count} 行")
            
            row_count += 1
    
    # 转换为DataFrame并保存
    logger.info(f"正在保存样本数据...")
    sample_df = pd.DataFrame(sampled_data)
    sample_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"完成! 原始数据: {total_lines} 行, 采样数据: {len(sample_df)} 行, 采样率: {len(sample_df)/total_lines*100:.2f}%")
    logger.info(f"保存到: {output_file}\n")
    
    return len(sample_df)


def main():
    """主函数：提取三个数据源的样本"""
    
    # 定义路径
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output' / 'samples'
    # 采样率
    sample_rate = 1
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义三个数据文件
    data_files = [
        ('智联招聘_广东省_202203_202506.csv', f'智联招聘_样本_{sample_rate}%.csv'),
        ('广东省招聘数据_猎聘网_202201_202506.csv', f'猎聘网_样本_{sample_rate}%.csv'),
        ('广东省招聘数据_前程无忧_202201_202506.csv', f'前程无忧_样本_{sample_rate}%.csv')
    ]
    
    logger.info("=" * 60)
    logger.info(f"开始提取样本数据 - 采样率: {sample_rate}%")
    logger.info("=" * 60)
    
    total_sampled = 0
    
    # 处理每个文件
    for input_name, output_name in data_files:
        input_path = data_dir / input_name
        output_path = output_dir / output_name
        
        if not input_path.exists():
            logger.warning(f"文件不存在，跳过: {input_path}")
            continue
        
        try:
            sampled = extract_sample_uniform(input_path, output_path, sample_rate=sample_rate/100)
            total_sampled += sampled
        except Exception as e:
            logger.error(f"处理文件失败: {input_name}, 错误: {e}")
    
    logger.info("=" * 60)
    logger.info(f"样本提取完成! 总采样数: {total_sampled} 行")
    logger.info(f"样本数据保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

