"""
NLP处理结果分析
统计关键词、技能需求、生成初步报告
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_nlp_results():
    """分析NLP处理结果"""
    
    base_dir = Path('../../')
    nlp_dir = base_dir / 'output' / 'nlp_processed'
    
    logger.info("=" * 80)
    logger.info("NLP处理结果分析")
    logger.info("=" * 80)
    
    all_keywords = []
    total_jobs = 0
    
    # 分析每个文件
    for csv_file in nlp_dir.glob('*.csv'):
        logger.info(f"\n分析文件: {csv_file.name}")
        
        try:
            # 读取数据（只读取需要的列）
            df = pd.read_csv(csv_file, usecols=['岗位名称', '工作城市', '薪资水平', '关键词', '岗位描述_分词'])
            logger.info(f"  总行数: {len(df):,}")
            
            total_jobs += len(df)
            
            # 统计关键词
            keywords_list = df['关键词'].dropna().str.split(',').tolist()
            for kw_list in keywords_list:
                all_keywords.extend(kw_list)
            
            # 城市分布
            city_dist = df['工作城市'].value_counts().head(10)
            logger.info(f"\n  Top 10 城市:")
            for city, count in city_dist.items():
                logger.info(f"    {city}: {count:,} ({count/len(df)*100:.1f}%)")
            
            # 岗位分布
            job_dist = df['岗位名称'].value_counts().head(10)
            logger.info(f"\n  Top 10 岗位:")
            for job, count in job_dist.items():
                logger.info(f"    {job}: {count:,}")
            
        except Exception as e:
            logger.error(f"  处理失败: {e}")
    
    # 全局关键词统计
    logger.info("\n" + "=" * 80)
    logger.info("全局统计")
    logger.info("=" * 80)
    logger.info(f"总岗位数: {total_jobs:,}")
    logger.info(f"总关键词数: {len(all_keywords):,}")
    
    # Top 50 热门关键词
    keyword_counter = Counter(all_keywords)
    top_keywords = keyword_counter.most_common(50)
    
    logger.info(f"\nTop 50 热门关键词:")
    logger.info("-" * 80)
    for i, (keyword, count) in enumerate(top_keywords, 1):
        logger.info(f"{i:2d}. {keyword:20s} - {count:8,} 次 ({count/total_jobs*100:.2f}%)")
    
    # 保存关键词统计
    output_file = base_dir / 'output' / 'reports' / '关键词统计.txt'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("广东省招聘数据 - 关键词统计报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总岗位数: {total_jobs:,}\n")
        f.write(f"总关键词数: {len(all_keywords):,}\n")
        f.write(f"独特关键词数: {len(keyword_counter):,}\n\n")
        f.write("Top 100 热门关键词:\n")
        f.write("-" * 80 + "\n")
        
        for i, (keyword, count) in enumerate(keyword_counter.most_common(100), 1):
            f.write(f"{i:3d}. {keyword:25s} - {count:10,} 次 ({count/total_jobs*100:.2f}%)\n")
    
    logger.info(f"\n关键词统计已保存到: {output_file}")
    
    # 识别技能关键词
    skill_keywords = [
        'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go', 'SQL',
        '机器学习', '深度学习', '数据分析', '大数据', '云计算', '人工智能',
        '前端', '后端', '全栈', 'Web', 'APP',
        'MySQL', 'Redis', 'MongoDB', 'Oracle',
        'Vue', 'React', 'Angular', 'Spring', 'Django',
        'Linux', 'Docker', 'Kubernetes',
        'Excel', 'PPT', 'Word', 'Photoshop', 'AutoCAD',
        '产品经理', '项目管理', '市场营销', '数据运营'
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("技能需求统计")
    logger.info("=" * 80)
    
    skill_stats = []
    for skill in skill_keywords:
        count = keyword_counter.get(skill, 0)
        if count > 0:
            skill_stats.append((skill, count))
    
    skill_stats.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n发现 {len(skill_stats)} 个技能关键词:")
    for i, (skill, count) in enumerate(skill_stats[:30], 1):
        logger.info(f"{i:2d}. {skill:20s} - {count:8,} 次 ({count/total_jobs*100:.2f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 分析完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    analyze_nlp_results()

