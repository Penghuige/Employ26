"""
薪资分析模块
分析不同技能、学历、经验对薪资的影响
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_salary(salary_str):
    """解析薪资字符串，返回最小值和最大值（月薪，单位：元）"""
    if pd.isna(salary_str) or salary_str == '':
        return None, None
    
    salary_str = str(salary_str).strip()
    
    # 匹配各种格式
    patterns = [
        (r'(\d+\.?\d*)-(\d+\.?\d*)万', 10000),  # 1-2万
        (r'(\d+)-(\d+)万', 10000),              # 1-2万
        (r'(\d+\.?\d*)万-(\d+\.?\d*)万', 10000), # 1万-2万
        (r'(\d+)-(\d+)千', 1000),               # 8-10千
        (r'(\d+)千-(\d+)千', 1000),             # 8千-10千
        (r'(\d+)-(\d+)k', 1000),                # 15-20k
        (r'(\d+)k-(\d+)k', 1000),               # 15k-20k
        (r'(\d+)-(\d+)元', 1),                  # 5000-8000元
        (r'(\d+)元-(\d+)元', 1),                # 5000元-8000元
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, salary_str, re.IGNORECASE)
        if match:
            min_sal = float(match.group(1)) * multiplier
            max_sal = float(match.group(2)) * multiplier
            
            # 处理年薪（如果有"年"字）
            if '年' in salary_str:
                min_sal /= 12
                max_sal /= 12
            
            return min_sal, max_sal
    
    return None, None


def analyze_salary_by_skill(df, skill_keywords):
    """分析不同技能的薪资水平"""
    logger.info("分析技能与薪资关系...")
    
    skill_salary = {}
    
    for skill in skill_keywords:
        # 找到包含该技能的岗位
        mask = df['关键词'].str.contains(skill, na=False, case=False)
        skill_jobs = df[mask]
        
        if len(skill_jobs) > 0:
            avg_salary = skill_jobs['平均薪资'].mean()
            median_salary = skill_jobs['平均薪资'].median()
            count = len(skill_jobs)
            
            skill_salary[skill] = {
                '平均薪资': avg_salary,
                '中位数薪资': median_salary,
                '岗位数量': count
            }
    
    return skill_salary


def analyze_salary_by_education(df):
    """分析学历与薪资关系"""
    logger.info("分析学历与薪资关系...")
    
    education_levels = ['博士', '硕士', '本科', '大专', '高中', '中专', '初中']
    education_salary = {}
    
    for edu in education_levels:
        mask = df['学历要求'].str.contains(edu, na=False)
        edu_jobs = df[mask]
        
        if len(edu_jobs) > 0:
            education_salary[edu] = {
                '平均薪资': edu_jobs['平均薪资'].mean(),
                '中位数薪资': edu_jobs['平均薪资'].median(),
                '岗位数量': len(edu_jobs)
            }
    
    return education_salary


def analyze_salary_by_experience(df):
    """分析经验与薪资关系"""
    logger.info("分析经验与薪资关系...")
    
    experience_levels = ['10年以上', '8-10年', '5-10年', '5-7年', '3-5年', '1-3年', '1年以下', '应届']
    experience_salary = {}
    
    for exp in experience_levels:
        mask = df['经验要求'].str.contains(exp, na=False)
        exp_jobs = df[mask]
        
        if len(exp_jobs) > 0:
            experience_salary[exp] = {
                '平均薪资': exp_jobs['平均薪资'].mean(),
                '中位数薪资': exp_jobs['平均薪资'].median(),
                '岗位数量': len(exp_jobs)
            }
    
    return experience_salary


def analyze_salary_by_city(df):
    """分析城市与薪资关系"""
    logger.info("分析城市与薪资关系...")
    
    # 提取主要城市
    df['主要城市'] = df['工作城市'].str.extract(r'(深圳|广州|佛山|东莞|惠州|珠海|中山|江门|肇庆|汕头|湛江|茂名|韶关|梅州|清远|阳江|河源|云浮|潮州|揭阳|汕尾)')
    
    city_salary = df.groupby('主要城市')['平均薪资'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    
    return city_salary


def main():
    """主函数"""
    base_dir = Path('d:/pythonProject/leisure/Employ26')
    nlp_dir = base_dir / 'output' / 'nlp_processed'
    output_dir = base_dir / 'output' / 'reports'
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("薪资分析")
    logger.info("=" * 80)
    
    # 读取数据
    all_data = []
    for csv_file in nlp_dir.glob('*.csv'):
        logger.info(f"读取: {csv_file.name}")
        df = pd.read_csv(csv_file, usecols=['岗位名称', '工作城市', '薪资水平', '学历要求', '经验要求', '关键词'])
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"总数据: {len(df):,} 行")
    
    # 解析薪资
    logger.info("解析薪资...")
    df[['最低薪资', '最高薪资']] = df['薪资水平'].apply(
        lambda x: pd.Series(parse_salary(x))
    )
    df['平均薪资'] = (df['最低薪资'] + df['最高薪资']) / 2
    
    # 过滤有效薪资数据
    df_valid = df[df['平均薪资'].notna()].copy()
    logger.info(f"有效薪资数据: {len(df_valid):,} 行 ({len(df_valid)/len(df)*100:.1f}%)")
    
    # 基础统计
    logger.info("\n" + "=" * 80)
    logger.info("薪资基础统计")
    logger.info("=" * 80)
    logger.info(f"平均薪资: {df_valid['平均薪资'].mean():,.0f} 元/月")
    logger.info(f"中位数薪资: {df_valid['平均薪资'].median():,.0f} 元/月")
    logger.info(f"最低薪资: {df_valid['平均薪资'].min():,.0f} 元/月")
    logger.info(f"最高薪资: {df_valid['平均薪资'].max():,.0f} 元/月")
    logger.info(f"25分位数: {df_valid['平均薪资'].quantile(0.25):,.0f} 元/月")
    logger.info(f"75分位数: {df_valid['平均薪资'].quantile(0.75):,.0f} 元/月")
    
    # 技能薪资分析
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
    
    skill_salary = analyze_salary_by_skill(df_valid, skill_keywords)
    
    logger.info("\n" + "=" * 80)
    logger.info("技能薪资排行 Top 30")
    logger.info("=" * 80)
    
    # 按平均薪资排序
    sorted_skills = sorted(skill_salary.items(), key=lambda x: x[1]['平均薪资'], reverse=True)
    
    for i, (skill, data) in enumerate(sorted_skills[:30], 1):
        logger.info(f"{i:2d}. {skill:20s} - 平均: {data['平均薪资']:8,.0f}元 | "
                   f"中位数: {data['中位数薪资']:8,.0f}元 | 岗位数: {data['岗位数量']:6,}")
    
    # 学历薪资分析
    education_salary = analyze_salary_by_education(df_valid)
    
    logger.info("\n" + "=" * 80)
    logger.info("学历薪资分析")
    logger.info("=" * 80)
    
    for edu, data in sorted(education_salary.items(), key=lambda x: x[1]['平均薪资'], reverse=True):
        logger.info(f"{edu:10s} - 平均: {data['平均薪资']:8,.0f}元 | "
                   f"中位数: {data['中位数薪资']:8,.0f}元 | 岗位数: {data['岗位数量']:6,}")
    
    # 经验薪资分析
    experience_salary = analyze_salary_by_experience(df_valid)
    
    logger.info("\n" + "=" * 80)
    logger.info("经验薪资分析")
    logger.info("=" * 80)
    
    for exp, data in sorted(experience_salary.items(), key=lambda x: x[1]['平均薪资'], reverse=True):
        logger.info(f"{exp:15s} - 平均: {data['平均薪资']:8,.0f}元 | "
                   f"中位数: {data['中位数薪资']:8,.0f}元 | 岗位数: {data['岗位数量']:6,}")
    
    # 城市薪资分析
    city_salary = analyze_salary_by_city(df_valid)
    
    logger.info("\n" + "=" * 80)
    logger.info("城市薪资排行 Top 15")
    logger.info("=" * 80)
    
    for city, row in city_salary.head(15).iterrows():
        logger.info(f"{city:10s} - 平均: {row['mean']:8,.0f}元 | "
                   f"中位数: {row['median']:8,.0f}元 | 岗位数: {int(row['count']):6,}")
    
    # 保存报告
    report_file = output_dir / '薪资分析报告.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("广东省招聘数据 - 薪资分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("基础统计:\n")
        f.write(f"  平均薪资: {df_valid['平均薪资'].mean():,.0f} 元/月\n")
        f.write(f"  中位数薪资: {df_valid['平均薪资'].median():,.0f} 元/月\n")
        f.write(f"  25分位数: {df_valid['平均薪资'].quantile(0.25):,.0f} 元/月\n")
        f.write(f"  75分位数: {df_valid['平均薪资'].quantile(0.75):,.0f} 元/月\n\n")
        
        f.write("技能薪资排行 Top 50:\n")
        f.write("-" * 80 + "\n")
        for i, (skill, data) in enumerate(sorted_skills[:50], 1):
            f.write(f"{i:3d}. {skill:25s} - 平均: {data['平均薪资']:10,.0f}元 | "
                   f"中位数: {data['中位数薪资']:10,.0f}元 | 岗位数: {data['岗位数量']:8,}\n")
        
        f.write("\n学历薪资分析:\n")
        f.write("-" * 80 + "\n")
        for edu, data in sorted(education_salary.items(), key=lambda x: x[1]['平均薪资'], reverse=True):
            f.write(f"{edu:15s} - 平均: {data['平均薪资']:10,.0f}元 | "
                   f"中位数: {data['中位数薪资']:10,.0f}元 | 岗位数: {data['岗位数量']:8,}\n")
        
        f.write("\n经验薪资分析:\n")
        f.write("-" * 80 + "\n")
        for exp, data in sorted(experience_salary.items(), key=lambda x: x[1]['平均薪资'], reverse=True):
            f.write(f"{exp:20s} - 平均: {data['平均薪资']:10,.0f}元 | "
                   f"中位数: {data['中位数薪资']:10,.0f}元 | 岗位数: {data['岗位数量']:8,}\n")
        
        f.write("\n城市薪资排行:\n")
        f.write("-" * 80 + "\n")
        for city, row in city_salary.iterrows():
            f.write(f"{city:15s} - 平均: {row['mean']:10,.0f}元 | "
                   f"中位数: {row['median']:10,.0f}元 | 岗位数: {int(row['count']):8,}\n")
    
    logger.info(f"\n报告已保存到: {report_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 薪资分析完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

