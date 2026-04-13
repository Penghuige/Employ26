#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量解析岗位名称
对NLP处理后的数据进行职业名称解析
支持灵活的输入路径配置
"""

import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import argparse

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.job_title_parsing.occupation_parser import OccupationParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchOccupationParser:
    """批量岗位名称解析器"""
    
    def __init__(self, base_dir=None, input_dir=None):
        """初始化
        
        Args:
            base_dir: 项目根目录
            input_dir: NLP处理后的数据目录（默认：output/nlp_processed_full/）
        """
        if base_dir is None:
            base_dir = Path(__file__).parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        
        # 设置输入目录
        if input_dir is None:
            self.input_dir = base_dir / 'output' / 'nlp_processed_full'
        else:
            self.input_dir = Path(input_dir)
            if not self.input_dir.is_absolute():
                self.input_dir = base_dir / self.input_dir
        
        self.output_dir = base_dir / 'output' / 'job_title_parsing'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化解析器
        self.parser = OccupationParser()
        
        logger.info("批量岗位名称解析器初始化完成")
        logger.info(f"  输入目录: {self.input_dir}")
        logger.info(f"  输出目录: {self.output_dir}")
    
    def parse_file(self, input_file):
        """解析单个文件"""
        logger.info(f"\n处理文件: {input_file.name}")
        
        # 读取数据
        logger.info("  读取数据...")
        df = pd.read_csv(input_file, encoding='utf-8')
        logger.info(f"  总数据: {len(df):,} 行")
        
        # 检查必要字段
        if '岗位名称' not in df.columns:
            logger.warning(f"  ⚠️  缺少'岗位名称'字段，跳过")
            return None
        
        # 解析岗位名称
        logger.info("  解析岗位名称...")
        results = []
        
        for job_title in tqdm(df['岗位名称'], desc="  解析进度"):
            result = self.parser.parse(str(job_title))
            results.append(result)
        
        # 转换为DataFrame
        df_parsed = pd.DataFrame(results)
        
        # 合并到原数据
        df['occupation_core'] = df_parsed['occupation_core']
        df['occupation_category'] = df_parsed['core_category']
        df['occupation_modifiers'] = df_parsed['modifiers']
        df['occupation_confidence'] = df_parsed['confidence']
        
        # 保存结果
        output_file = self.output_dir / f"{input_file.stem}_解析.csv"
        logger.info(f"  保存到: {output_file.name}")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 统计信息
        logger.info(f"  ✅ 完成: {input_file.name}")
        logger.info(f"  统计信息:")
        logger.info(f"    - 总岗位: {len(df):,}")
        logger.info(f"    - 成功解析: {df['occupation_core'].notna().sum():,}")
        logger.info(f"    - 解析率: {df['occupation_core'].notna().sum()/len(df)*100:.1f}%")
        
        # 职业类别分布
        logger.info(f"  职业类别分布:")
        category_counts = df['occupation_category'].value_counts()
        for category, count in category_counts.head(10).items():
            logger.info(f"    - {category}: {count:,} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def parse_all(self):
        """解析所有文件"""
        logger.info("=" * 80)
        logger.info("批量岗位名称解析")
        logger.info("=" * 80)
        
        # 检查输入目录
        if not self.input_dir.exists():
            logger.error(f"❌ 输入目录不存在: {self.input_dir}")
            logger.info(f"提示：请先运行 process_full_data_nlp.py 进行NLP处理")
            return
        
        # 查找所有CSV文件
        csv_files = list(self.input_dir.glob('*_NLP处理.csv'))
        
        if not csv_files:
            logger.warning(f"⚠️  在 {self.input_dir} 目录下未找到NLP处理后的CSV文件")
            logger.info(f"提示：请先运行 process_full_data_nlp.py")
            return
        
        logger.info(f"\n找到 {len(csv_files)} 个CSV文件:")
        for f in csv_files:
            logger.info(f"  - {f.name}")
        
        # 解析每个文件
        all_results = []
        for csv_file in csv_files:
            try:
                df = self.parse_file(csv_file)
                if df is not None:
                    all_results.append(df)
            except Exception as e:
                logger.error(f"  ❌ 解析失败: {csv_file.name} - {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 批量岗位名称解析完成!")
        logger.info("=" * 80)
        logger.info(f"\n解析后的数据保存在: {self.output_dir}")
        logger.info("\n新增字段:")
        logger.info("  - occupation_core: 职业核心词（如：工程师、经理）")
        logger.info("  - occupation_category: 职业类别（如：技术类、管理类）")
        logger.info("  - occupation_modifiers: 修饰词（如：高级 Python 后端）")
        logger.info("  - occupation_confidence: 解析置信度（0.0-1.0）")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量岗位名称解析')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='NLP处理后的数据目录（默认：output/nlp_processed_full/）')
    
    args = parser.parse_args()
    
    parser_obj = BatchOccupationParser(input_dir=args.input)
    parser_obj.parse_all()


if __name__ == '__main__':
    main()
