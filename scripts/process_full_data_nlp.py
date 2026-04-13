#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量数据NLP处理
对指定目录下的所有原始数据进行NLP处理（分词、关键词提取等）
"""

import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import jieba
import jieba.analyse
import re
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullDataNLPProcessor:
    """全量数据NLP处理器"""
    
    def __init__(self, input_dir=None, output_dir=None, base_dir=None):
        """初始化
        
        Args:
            input_dir: 输入数据目录（默认：data/）
            output_dir: 输出数据目录（默认：output/nlp_processed_full/）
            base_dir: 项目根目录
        """
        if base_dir is None:
            base_dir = Path(__file__).parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        
        # 设置输入目录
        if input_dir is None:
            self.data_dir = base_dir / 'data'
        else:
            self.data_dir = Path(input_dir)
            if not self.data_dir.is_absolute():
                self.data_dir = base_dir / self.data_dir
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = base_dir / 'output' / 'nlp_processed_full'
        else:
            self.output_dir = Path(output_dir)
            if not self.output_dir.is_absolute():
                self.output_dir = base_dir / self.output_dir
        
        # 设置报告目录
        self.report_dir = base_dir / 'output' / 'nlp_reports'
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.dicts_dir = base_dir / 'dicts'
        
        # 加载自定义词典和停用词
        self._load_dicts()
        
        logger.info("全量数据NLP处理器初始化完成")
        logger.info(f"  输入目录: {self.data_dir}")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info(f"  报告目录: {self.report_dir}")
    
    def _load_dicts(self):
        """加载自定义词典和停用词"""
        # 加载自定义词典
        userdict_file = self.dicts_dir / 'userdict_zh_recruitment.txt'
        if userdict_file.exists():
            jieba.load_userdict(str(userdict_file))
            logger.info(f"  已加载自定义词典: {userdict_file.name}")
        else:
            logger.warning(f"  ⚠️  未找到自定义词典: {userdict_file}")
        
        # 加载停用词
        stopwords_file = self.dicts_dir / 'stopwords_recruitment_short.txt'
        self.stopwords = set()
        if stopwords_file.exists():
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            logger.info(f"  已加载停用词: {len(self.stopwords)} 个")
        else:
            logger.warning(f"  ⚠️  未找到停用词文件: {stopwords_file}")
    
    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text):
            return ''
        
        text = str(text)
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除特殊字符（保留中文、英文、数字）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def segment_text(self, text):
        """分词"""
        if not text:
            return ''
        
        words = jieba.cut(text)
        # 过滤停用词和单字
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        return ' '.join(words)
    
    def extract_keywords(self, text, topK=10):
        """提取关键词"""
        if not text:
            return ''
        
        keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=False)
        return ','.join(keywords)
    
    def process_file(self, input_file):
        """处理单个文件"""
        logger.info(f"\n处理文件: {input_file.name}")
        
        # 读取数据（分块处理）
        chunk_size = 10000
        chunks = []
        total_rows = 0
        
        logger.info("  读取数据...")
        try:
            for chunk in tqdm(pd.read_csv(input_file, encoding='utf-8', chunksize=chunk_size), 
                             desc="  读取数据块"):
                total_rows += len(chunk)
                chunks.append(chunk)
        except Exception as e:
            logger.error(f"  ❌ 读取文件失败: {e}")
            return None
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  总数据: {len(df):,} 行")
        
        # 检查必要字段
        if '岗位描述' not in df.columns:
            logger.warning(f"  ⚠️  缺少'岗位描述'字段，跳过")
            return None
        
        # NLP处理
        logger.info("  NLP处理中...")
        
        # 1. 清洗岗位描述
        tqdm.pandas(desc="  清洗文本")
        df['岗位描述_清洗'] = df['岗位描述'].progress_apply(self.clean_text)
        
        # 2. 分词
        tqdm.pandas(desc="  分词")
        df['岗位描述_分词'] = df['岗位描述_清洗'].progress_apply(self.segment_text)
        
        # 3. 提取关键词
        tqdm.pandas(desc="  提取关键词")
        df['关键词'] = df['岗位描述_清洗'].progress_apply(lambda x: self.extract_keywords(x, topK=10))
        
        # 保存结果
        output_file = self.output_dir / f"{input_file.stem}_NLP处理.csv"
        logger.info(f"  保存到: {output_file.name}")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"  ✅ 完成: {input_file.name}")
        
        # 统计信息
        stats = {
            'file_name': input_file.name,
            'total_rows': len(df),
            'cleaned_rows': df['岗位描述_清洗'].notna().sum(),
            'segmented_rows': df['岗位描述_分词'].notna().sum(),
            'keywords_rows': df['关键词'].notna().sum()
        }
        
        logger.info(f"  统计信息:")
        logger.info(f"    - 原始数据: {stats['total_rows']:,} 行")
        logger.info(f"    - 清洗后非空: {stats['cleaned_rows']:,} 行")
        logger.info(f"    - 分词后非空: {stats['segmented_rows']:,} 行")
        logger.info(f"    - 关键词非空: {stats['keywords_rows']:,} 行")
        
        return stats
    
    def process_all(self):
        """处理所有文件"""
        logger.info("=" * 80)
        logger.info("全量数据NLP处理")
        logger.info("=" * 80)
        
        # 检查输入目录
        if not self.data_dir.exists():
            logger.error(f"❌ 输入目录不存在: {self.data_dir}")
            logger.info(f"提示：请确保数据文件在 {self.data_dir} 目录下")
            return
        
        # 查找所有CSV文件
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning(f"⚠️  在 {self.data_dir} 目录下未找到CSV文件")
            logger.info(f"提示：请将原始数据文件放在 {self.data_dir} 目录下")
            return
        
        logger.info(f"\n找到 {len(csv_files)} 个CSV文件:")
        for f in csv_files:
            logger.info(f"  - {f.name}")
        
        # 处理每个文件
        all_stats = []
        for csv_file in csv_files:
            try:
                stats = self.process_file(csv_file)
                if stats:
                    all_stats.append(stats)
            except Exception as e:
                logger.error(f"  ❌ 处理失败: {csv_file.name} - {e}")
                import traceback
                traceback.print_exc()
        
        # 生成处理报告
        self._generate_report(all_stats)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 全量数据NLP处理完成!")
        logger.info("=" * 80)
        logger.info(f"\n处理后的数据保存在: {self.output_dir}")
        logger.info(f"处理报告保存在: {self.report_dir}")
        logger.info("\n新增字段:")
        logger.info("  - 岗位描述_清洗: 清洗后的岗位描述")
        logger.info("  - 岗位描述_分词: 分词后的岗位描述")
        logger.info("  - 关键词: TF-IDF提取的关键词（逗号分隔）")
    
    def _generate_report(self, all_stats):
        """生成处理报告"""
        if not all_stats:
            logger.warning("  ⚠️  没有统计数据，跳过报告生成")
            return
        
        logger.info("\n生成处理报告...")
        
        # 创建统计DataFrame
        df_stats = pd.DataFrame(all_stats)
        
        # 保存CSV报告
        report_csv = self.report_dir / 'NLP处理统计报告.csv'
        df_stats.to_csv(report_csv, index=False, encoding='utf-8-sig')
        logger.info(f"  ✅ CSV报告: {report_csv}")
        
        # 生成文本报告
        report_txt = self.report_dir / 'NLP处理统计报告.txt'
        with open(report_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("全量数据NLP处理统计报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"输入目录: {self.data_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"处理文件数: {len(all_stats)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("各文件处理统计\n")
            f.write("=" * 80 + "\n\n")
            
            for stats in all_stats:
                f.write(f"文件: {stats['file_name']}\n")
                f.write(f"  原始数据: {stats['total_rows']:,} 行\n")
                f.write(f"  清洗后非空: {stats['cleaned_rows']:,} 行 ({stats['cleaned_rows']/stats['total_rows']*100:.1f}%)\n")
                f.write(f"  分词后非空: {stats['segmented_rows']:,} 行 ({stats['segmented_rows']/stats['total_rows']*100:.1f}%)\n")
                f.write(f"  关键词非空: {stats['keywords_rows']:,} 行 ({stats['keywords_rows']/stats['total_rows']*100:.1f}%)\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("总计\n")
            f.write("=" * 80 + "\n\n")
            
            total_rows = sum(s['total_rows'] for s in all_stats)
            total_cleaned = sum(s['cleaned_rows'] for s in all_stats)
            total_segmented = sum(s['segmented_rows'] for s in all_stats)
            total_keywords = sum(s['keywords_rows'] for s in all_stats)
            
            f.write(f"总原始数据: {total_rows:,} 行\n")
            f.write(f"总清洗后非空: {total_cleaned:,} 行 ({total_cleaned/total_rows*100:.1f}%)\n")
            f.write(f"总分词后非空: {total_segmented:,} 行 ({total_segmented/total_rows*100:.1f}%)\n")
            f.write(f"总关键词非空: {total_keywords:,} 行 ({total_keywords/total_rows*100:.1f}%)\n")
        
        logger.info(f"  ✅ 文本报告: {report_txt}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全量数据NLP处理')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='输入数据目录（默认：data/）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出数据目录（默认：output/nlp_processed_full/）')
    
    args = parser.parse_args()
    
    processor = FullDataNLPProcessor(
        input_dir=args.input,
        output_dir=args.output
    )
    processor.process_all()


if __name__ == '__main__':
    main()
