"""
NLP文本预处理模块 - 第一步
对岗位描述进行清洗、分词、停用词过滤
"""

import pandas as pd
import jieba
import jieba.analyse
import re
import os
from pathlib import Path
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        """初始化"""
        self.base_dir = Path(__file__).parent.parent.parent
        self.dict_dir = self.base_dir / 'dicts'
        self.stopwords = self._load_stopwords()
        self._add_custom_dict()
        logger.info("文本预处理器初始化完成")
    
    def _load_stopwords(self):
        """从txt文件加载停用词"""
        stopwords = set()
        
        # 读取停用词文件
        stopwords_file = self.dict_dir / 'stopwords_recruitment_short'
        
        if stopwords_file.exists():
            logger.info(f"从文件加载停用词: {stopwords_file}")
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # 跳过空行和注释
                        stopwords.add(word)
            logger.info(f"加载停用词: {len(stopwords)} 个")
        else:
            logger.warning(f"停用词文件不存在: {stopwords_file}")
            # 使用基础停用词作为后备
            stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个'}
            logger.info(f"使用默认停用词: {len(stopwords)} 个")
        
        return stopwords
    
    def _add_custom_dict(self):
        """从txt文件加载自定义词典"""
        custom_dict_file = self.dict_dir / 'userdict_zh_recruitment.txt'
        
        if custom_dict_file.exists():
            logger.info(f"从文件加载自定义词典: {custom_dict_file}")
            
            # 使用jieba的load_userdict方法直接加载
            jieba.load_userdict(str(custom_dict_file))
            
            # 统计词典数量
            word_count = 0
            with open(custom_dict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        word_count += 1
            
            logger.info(f"加载自定义词典: {word_count} 个词汇")
        else:
            logger.warning(f"自定义词典文件不存在: {custom_dict_file}")
            logger.info("使用jieba默认词典")
    
    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除网址
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除邮箱
        text = re.sub(r'\w+@\w+\.\w+', '', text)
        
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 去除特殊字符（保留中文、英文、数字）
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text):
        """分词"""
        if not text:
            return []
        
        # 使用jieba分词
        words = jieba.cut(text)
        
        # 过滤停用词和短词
        words = [w.strip() for w in words if w.strip() and len(w.strip()) > 1]
        words = [w for w in words if w not in self.stopwords]
        
        return words
    
    def extract_keywords(self, text, topK=20):
        """提取关键词"""
        if not text:
            return []
        
        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=False)
        return keywords
    
    def process_text(self, text):
        """完整的文本处理流程"""
        # 清洗
        cleaned = self.clean_text(text)
        
        # 分词
        tokens = self.tokenize(cleaned)
        
        # 返回分词结果（空格分隔）
        return ' '.join(tokens)


def process_sample_data(input_file, output_file):
    """处理样本数据"""
    logger.info(f"开始处理: {input_file}")
    
    # 读取数据
    df = pd.read_csv(input_file, encoding='utf-8')
    logger.info(f"读取数据: {len(df)} 行")
    
    # 初始化预处理器
    preprocessor = TextPreprocessor()
    
    # 处理岗位描述
    logger.info("正在处理岗位描述...")
    tqdm.pandas(desc="文本预处理")
    df['岗位描述_清洗'] = df['岗位描述'].progress_apply(preprocessor.clean_text)
    df['岗位描述_分词'] = df['岗位描述_清洗'].progress_apply(preprocessor.process_text)
    
    # 提取关键词
    logger.info("正在提取关键词...")
    tqdm.pandas(desc="关键词提取")
    df['关键词'] = df['岗位描述'].progress_apply(lambda x: ','.join(preprocessor.extract_keywords(x, topK=10)))
    
    # 处理岗位名称
    logger.info("正在处理岗位名称...")
    df['岗位名称_分词'] = df['岗位名称'].apply(preprocessor.process_text)
    
    # 保存结果
    logger.info(f"正在保存到: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"处理完成! 保存了 {len(df)} 行数据\n")
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("数据统计:")
    logger.info(f"总行数: {len(df)}")
    logger.info(f"岗位描述非空: {df['岗位描述'].notna().sum()}")
    logger.info(f"平均分词数: {df['岗位描述_分词'].apply(lambda x: len(x.split())).mean():.2f}")
    logger.info("=" * 60)
    
    return df


def main():
    """主函数：处理所有样本数据"""
    
    # 定义路径
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / 'output' / 'samples'
    output_dir = base_dir / 'output' / 'nlp_processed'

    # 样本名
    sample_name = "样本_1%"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义文件
    sample_files = [
        (f'智联招聘_{sample_name}.csv', f'智联招聘_NLP处理_{sample_name}.csv'),
        (f'猎聘网_{sample_name}.csv', f'猎聘网_NLP处理_{sample_name}.csv'),
        (f'前程无忧_{sample_name}.csv', f'前程无忧_NLP处理_{sample_name}.csv')
    ]
    
    logger.info("=" * 60)
    logger.info("开始NLP文本预处理")
    logger.info("=" * 60)
    
    # 处理每个文件
    for input_name, output_name in sample_files:
        input_path = input_dir / input_name
        output_path = output_dir / output_name
        
        if not input_path.exists():
            logger.warning(f"文件不存在，跳过: {input_path}")
            continue
        
        try:
            process_sample_data(input_path, output_path)
        except Exception as e:
            logger.error(f"处理文件失败: {input_name}, 错误: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("NLP预处理完成!")
    logger.info(f"处理结果保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

