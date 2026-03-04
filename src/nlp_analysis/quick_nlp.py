"""
快速NLP处理 - 处理前程无忧的小样本（6万行）
演示完整的NLP流程
"""

import pandas as pd
import jieba
import jieba.analyse
import re
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加自定义词典
skills = [
    'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go', 
    '机器学习', '深度学习', '数据分析', '大数据', '云计算',
    '前端开发', '后端开发', '全栈开发', 'Web开发',
    'MySQL', 'Redis', 'MongoDB', 'Vue', 'React', 'Spring',
    'Photoshop', 'Excel', 'AutoCAD', '产品经理', '项目管理'
]
for skill in skills:
    jieba.add_word(skill, freq=10000)

# 停用词
stopwords = set([
    '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '上', '也', '很',
    '到', '要', '会', '着', '看', '好', '个', '地', '为', '中', '大', '与', '及',
    '岗位', '职责', '要求', '任职', '工作', '负责', '具有', '具备', '能力', '相关',
    '经验', '优先', '熟悉', '了解', '掌握', '良好', '较强', '以上', '公司', '团队'
])

def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)  # 去HTML
    text = re.sub(r'http[s]?://\S+', '', text)  # 去URL
    text = re.sub(r'\s+', ' ', text)  # 去多余空格
    return text.strip()

def tokenize(text):
    """分词并过滤"""
    if not text:
        return []
    words = jieba.cut(text)
    words = [w.strip() for w in words if w.strip() and len(w.strip()) > 1]
    words = [w for w in words if w not in stopwords]
    return words

def process_file(input_file, output_file):
    """处理单个文件"""
    logger.info(f"开始处理: {input_file.name}")
    
    # 读取数据
    df = pd.read_csv(input_file, encoding='utf-8')
    logger.info(f"读取数据: {len(df):,} 行")
    
    # 清洗岗位描述
    logger.info("清洗文本...")
    tqdm.pandas(desc="清洗")
    df['岗位描述_清洗'] = df['岗位描述'].progress_apply(clean_text)
    
    # 分词
    logger.info("分词...")
    tqdm.pandas(desc="分词")
    df['岗位描述_分词'] = df['岗位描述_清洗'].progress_apply(
        lambda x: ' '.join(tokenize(x))
    )
    
    # 提取关键词
    logger.info("提取关键词...")
    tqdm.pandas(desc="关键词")
    df['关键词'] = df['岗位描述'].progress_apply(
        lambda x: ','.join(jieba.analyse.extract_tags(x, topK=10)) if pd.notna(x) else ''
    )
    
    # 保存
    logger.info(f"保存到: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 统计
    logger.info("=" * 60)
    logger.info(f"处理完成: {len(df):,} 行")
    logger.info(f"平均分词数: {df['岗位描述_分词'].apply(lambda x: len(x.split())).mean():.1f}")
    logger.info(f"平均关键词数: {df['关键词'].apply(lambda x: len(x.split(',')) if x else 0).mean():.1f}")
    logger.info("=" * 60)
    
    return df

def main():
    base_dir = Path('/')
    input_dir = base_dir / 'output' / 'samples'
    output_dir = base_dir / 'output' / 'nlp_processed'
    output_dir.mkdir(exist_ok=True)
    
    # 只处理前程无忧（最小的文件）
    input_file = input_dir / '前程无忧_样本_10%.csv'
    output_file = output_dir / '前程无忧_NLP处理.csv'
    
    logger.info("=" * 60)
    logger.info("快速NLP处理 - 前程无忧样本")
    logger.info("=" * 60)
    
    df = process_file(input_file, output_file)
    
    # 显示示例
    logger.info("\n示例数据:")
    sample = df[['岗位名称', '关键词', '岗位描述_分词']].head(3)
    for idx, row in sample.iterrows():
        logger.info(f"\n岗位: {row['岗位名称']}")
        logger.info(f"关键词: {row['关键词']}")
        logger.info(f"分词: {row['岗位描述_分词'][:100]}...")
    
    logger.info("\n✅ NLP处理完成!")

if __name__ == '__main__':
    main()

