"""
生成整合的Excel分析报告
将薪资分析、技能组合分析、时间趋势分析整合到一个Excel文件中
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_salary(salary_str):
    """解析薪资字符串，返回最小值和最大值（月薪，单位：元）"""
    if pd.isna(salary_str) or salary_str == '':
        return None, None
    
    salary_str = str(salary_str).strip()
    
    # 匹配各种格式
    patterns = [
        (r'(\d+\.?\d*)-(\d+\.?\d*)万', 10000),
        (r'(\d+)-(\d+)万', 10000),
        (r'(\d+\.?\d*)万-(\d+\.?\d*)万', 10000),
        (r'(\d+)-(\d+)千', 1000),
        (r'(\d+)千-(\d+)千', 1000),
        (r'(\d+)-(\d+)k', 1000),
        (r'(\d+)k-(\d+)k', 1000),
        (r'(\d+)-(\d+)元', 1),
        (r'(\d+)元-(\d+)元', 1),
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, salary_str, re.IGNORECASE)
        if match:
            min_sal = float(match.group(1)) * multiplier
            max_sal = float(match.group(2)) * multiplier
            
            if '年' in salary_str:
                min_sal /= 12
                max_sal /= 12
            
            return min_sal, max_sal
    
    return None, None


class ExcelReportGenerator:
    """Excel报告生成器"""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.nlp_dir = self.base_dir / 'output' / 'nlp_processed'
        self.output_dir = self.base_dir / 'output' / 'reports'
        self.output_dir.mkdir(exist_ok=True)
        
        # 定义关注的技能
        self.target_skills = [
            'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go', 'SQL',
            '机器学习', '深度学习', '数据分析', '大数据', '云计算', '人工智能',
            '前端', '后端', '全栈', 'Web', 'APP',
            'MySQL', 'Redis', 'MongoDB', 'Oracle',
            'Vue', 'React', 'Angular', 'Spring', 'Django',
            'Linux', 'Docker', 'Kubernetes',
            'Excel', 'PPT', 'Word', 'Photoshop', 'AutoCAD',
            '产品经理', '项目管理', '市场营销', '数据运营'
        ]
        
        self.df = None
        self.df_valid = None
    
    def load_data(self):
        """加载数据"""
        logger.info("=" * 80)
        logger.info("加载数据")
        logger.info("=" * 80)
        
        all_data = []
        for csv_file in self.nlp_dir.glob('*.csv'):
            logger.info(f"读取: {csv_file.name}")
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        self.df = pd.concat(all_data, ignore_index=True)
        logger.info(f"总数据: {len(self.df):,} 行")
        
        # 解析薪资
        logger.info("解析薪资...")
        self.df[['最低薪资', '最高薪资']] = self.df['薪资水平'].apply(
            lambda x: pd.Series(parse_salary(x))
        )
        self.df['平均薪资'] = (self.df['最低薪资'] + self.df['最高薪资']) / 2
        
        # 过滤有效薪资数据
        self.df_valid = self.df[self.df['平均薪资'].notna()].copy()
        logger.info(f"有效薪资数据: {len(self.df_valid):,} 行 ({len(self.df_valid)/len(self.df)*100:.1f}%)")
        
        # 解析日期
        self.df['发布时间'] = pd.to_datetime(self.df['发布时间'], errors='coerce')
        self.df['年份'] = self.df['发布时间'].dt.year
        self.df['月份'] = self.df['发布时间'].dt.to_period('M')
        
        # 提取主要城市
        self.df_valid['主要城市'] = self.df_valid['工作城市'].str.extract(
            r'(深圳|广州|佛山|东莞|惠州|珠海|中山|江门|肇庆|汕头|湛江|茂名|韶关|梅州|清远|阳江|河源|云浮|潮州|揭阳|汕尾)'
        )
    
    def generate_summary_sheet(self):
        """生成概览Sheet"""
        logger.info("\n生成概览数据...")
        
        summary_data = {
            '指标': [
                '数据总量',
                '有效薪资数据量',
                '有效薪资数据占比',
                '平均薪资（元/月）',
                '中位数薪资（元/月）',
                '最低薪资（元/月）',
                '最高薪资（元/月）',
                '25分位数薪资（元/月）',
                '75分位数薪资（元/月）',
                '数据时间跨度',
                '涵盖城市数',
                '分析技能数',
                '报告生成时间'
            ],
            '数值': [
                f"{len(self.df):,}",
                f"{len(self.df_valid):,}",
                f"{len(self.df_valid)/len(self.df)*100:.2f}%",
                f"{self.df_valid['平均薪资'].mean():,.0f}",
                f"{self.df_valid['平均薪资'].median():,.0f}",
                f"{self.df_valid['平均薪资'].min():,.0f}",
                f"{self.df_valid['平均薪资'].max():,.0f}",
                f"{self.df_valid['平均薪资'].quantile(0.25):,.0f}",
                f"{self.df_valid['平均薪资'].quantile(0.75):,.0f}",
                f"{self.df['发布时间'].min().strftime('%Y-%m-%d')} 至 {self.df['发布时间'].max().strftime('%Y-%m-%d')}",
                f"{self.df_valid['主要城市'].nunique()}",
                f"{len(self.target_skills)}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def generate_skill_salary_sheet(self):
        """生成技能薪资分析Sheet"""
        logger.info("生成技能薪资分析...")
        
        skill_salary_list = []
        
        for skill in self.target_skills:
            mask = self.df_valid['关键词'].str.contains(skill, na=False, case=False)
            skill_jobs = self.df_valid[mask]
            
            if len(skill_jobs) > 0:
                skill_salary_list.append({
                    '技能': skill,
                    '平均薪资': skill_jobs['平均薪资'].mean(),
                    '中位数薪资': skill_jobs['平均薪资'].median(),
                    '最低薪资': skill_jobs['平均薪资'].min(),
                    '最高薪资': skill_jobs['平均薪资'].max(),
                    '岗位数量': len(skill_jobs),
                    '占比': f"{len(skill_jobs)/len(self.df_valid)*100:.2f}%"
                })
        
        df_skill = pd.DataFrame(skill_salary_list)
        df_skill = df_skill.sort_values('平均薪资', ascending=False).reset_index(drop=True)
        df_skill.index = df_skill.index + 1
        
        return df_skill
    
    def generate_education_salary_sheet(self):
        """生成学历薪资分析Sheet"""
        logger.info("生成学历薪资分析...")
        
        education_levels = ['博士', '硕士', '本科', '大专', '高中', '中专', '初中']
        education_salary_list = []
        
        for edu in education_levels:
            mask = self.df_valid['学历要求'].str.contains(edu, na=False)
            edu_jobs = self.df_valid[mask]
            
            if len(edu_jobs) > 0:
                education_salary_list.append({
                    '学历': edu,
                    '平均薪资': edu_jobs['平均薪资'].mean(),
                    '中位数薪资': edu_jobs['平均薪资'].median(),
                    '最低薪资': edu_jobs['平均薪资'].min(),
                    '最高薪资': edu_jobs['平均薪资'].max(),
                    '岗位数量': len(edu_jobs),
                    '占比': f"{len(edu_jobs)/len(self.df_valid)*100:.2f}%"
                })
        
        df_edu = pd.DataFrame(education_salary_list)
        df_edu = df_edu.sort_values('平均薪资', ascending=False).reset_index(drop=True)
        df_edu.index = df_edu.index + 1
        
        return df_edu
    
    def generate_experience_salary_sheet(self):
        """生成经验薪资分析Sheet"""
        logger.info("生成经验薪资分析...")
        
        experience_levels = ['10年以上', '8-10年', '5-10年', '5-7年', '3-5年', '1-3年', '1年以下', '应届']
        experience_salary_list = []
        
        for exp in experience_levels:
            mask = self.df_valid['经验要求'].str.contains(exp, na=False)
            exp_jobs = self.df_valid[mask]
            
            if len(exp_jobs) > 0:
                experience_salary_list.append({
                    '经验要求': exp,
                    '平均薪资': exp_jobs['平均薪资'].mean(),
                    '中位数薪资': exp_jobs['平均薪资'].median(),
                    '最低薪资': exp_jobs['平均薪资'].min(),
                    '最高薪资': exp_jobs['平均薪资'].max(),
                    '岗位数量': len(exp_jobs),
                    '占比': f"{len(exp_jobs)/len(self.df_valid)*100:.2f}%"
                })
        
        df_exp = pd.DataFrame(experience_salary_list)
        df_exp = df_exp.sort_values('平均薪资', ascending=False).reset_index(drop=True)
        df_exp.index = df_exp.index + 1
        
        return df_exp
    
    def generate_city_salary_sheet(self):
        """生成城市薪资分析Sheet"""
        logger.info("生成城市薪资分析...")
        
        city_stats = self.df_valid.groupby('主要城市')['平均薪资'].agg([
            ('平均薪资', 'mean'),
            ('中位数薪资', 'median'),
            ('最低薪资', 'min'),
            ('最高薪资', 'max'),
            ('岗位数量', 'count')
        ]).reset_index()
        
        city_stats['占比'] = (city_stats['岗位数量'] / len(self.df_valid) * 100).apply(lambda x: f"{x:.2f}%")
        city_stats = city_stats.sort_values('平均薪资', ascending=False).reset_index(drop=True)
        city_stats.index = city_stats.index + 1
        
        return city_stats
    
    def generate_skill_combination_sheet(self):
        """生成技能组合分析Sheet"""
        logger.info("生成技能组合分析...")
        
        # 收集技能集合
        all_skill_sets = []
        for keywords_str in self.df['关键词'].dropna():
            keywords = keywords_str.split(',')
            job_skills = [skill for skill in keywords if skill in self.target_skills]
            if len(job_skills) >= 2:
                all_skill_sets.append(set(job_skills))
        
        logger.info(f"找到 {len(all_skill_sets):,} 个包含多个技能的岗位")
        
        # 统计技能对
        skill_pairs = Counter()
        for skill_set in all_skill_sets:
            for pair in combinations(sorted(skill_set), 2):
                skill_pairs[pair] += 1
        
        # 转换为DataFrame
        pair_list = []
        for (skill1, skill2), count in skill_pairs.most_common(100):
            pair_list.append({
                '技能1': skill1,
                '技能2': skill2,
                '共现次数': count,
                '占比': f"{count/len(all_skill_sets)*100:.2f}%"
            })
        
        df_pairs = pd.DataFrame(pair_list)
        df_pairs.index = df_pairs.index + 1
        
        return df_pairs
    
    def generate_skill_triple_sheet(self):
        """生成技能三元组分析Sheet"""
        logger.info("生成技能三元组分析...")
        
        # 收集技能集合
        all_skill_sets = []
        for keywords_str in self.df['关键词'].dropna():
            keywords = keywords_str.split(',')
            job_skills = [skill for skill in keywords if skill in self.target_skills]
            if len(job_skills) >= 3:
                all_skill_sets.append(set(job_skills))
        
        # 统计技能三元组
        skill_triples = Counter()
        for skill_set in all_skill_sets:
            for triple in combinations(sorted(skill_set), 3):
                skill_triples[triple] += 1
        
        # 转换为DataFrame
        triple_list = []
        for (skill1, skill2, skill3), count in skill_triples.most_common(50):
            triple_list.append({
                '技能1': skill1,
                '技能2': skill2,
                '技能3': skill3,
                '共现次数': count,
                '占比': f"{count/len(all_skill_sets)*100:.2f}%"
            })
        
        df_triples = pd.DataFrame(triple_list)
        df_triples.index = df_triples.index + 1
        
        return df_triples
    
    def generate_time_trend_sheet(self):
        """生成时间趋势分析Sheet"""
        logger.info("生成时间趋势分析...")
        
        # 按年统计
        df_time = self.df[self.df['年份'].notna()].copy()
        yearly_counts = df_time.groupby('年份').size().reset_index(name='岗位数量')
        
        # 计算同比增长
        yearly_counts['同比增长'] = yearly_counts['岗位数量'].pct_change() * 100
        yearly_counts['同比增长'] = yearly_counts['同比增长'].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
        )
        
        yearly_counts.index = yearly_counts.index + 1
        
        return yearly_counts
    
    def generate_skill_trend_sheet(self):
        """生成技能需求趋势Sheet"""
        logger.info("生成技能需求趋势...")
        
        df_time = self.df[self.df['年份'].notna()].copy()
        
        # 统计每年每个技能的需求
        skill_by_year = defaultdict(lambda: defaultdict(int))
        yearly_totals = df_time.groupby('年份').size()
        
        for _, row in df_time.iterrows():
            year = row['年份']
            if pd.notna(row['关键词']):
                keywords = row['关键词'].split(',')
                for skill in self.target_skills:
                    if skill in keywords:
                        skill_by_year[skill][year] += 1
        
        # 转换为DataFrame
        trend_list = []
        for skill in self.target_skills:
            row_data = {'技能': skill}
            for year in sorted(yearly_totals.index):
                count = skill_by_year[skill][year]
                total = yearly_totals[year]
                percentage = count / total * 100 if total > 0 else 0
                row_data[f'{int(year)}年次数'] = count
                row_data[f'{int(year)}年占比'] = f"{percentage:.2f}%"
            trend_list.append(row_data)
        
        df_trend = pd.DataFrame(trend_list)
        
        return df_trend
    
    def generate_report(self):
        """生成完整的Excel报告"""
        logger.info("\n" + "=" * 80)
        logger.info("开始生成Excel报告")
        logger.info("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 生成各个Sheet
        sheets = {
            '概览': self.generate_summary_sheet(),
            '技能薪资分析': self.generate_skill_salary_sheet(),
            '学历薪资分析': self.generate_education_salary_sheet(),
            '经验薪资分析': self.generate_experience_salary_sheet(),
            '城市薪资分析': self.generate_city_salary_sheet(),
            '技能组合Top100': self.generate_skill_combination_sheet(),
            '技能三元组Top50': self.generate_skill_triple_sheet(),
            '年度招聘趋势': self.generate_time_trend_sheet(),
            '技能需求趋势': self.generate_skill_trend_sheet()
        }
        
        # 保存到Excel
        output_file = self.output_dir / f'招聘数据分析报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        logger.info(f"\n保存Excel文件: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                logger.info(f"  写入Sheet: {sheet_name} ({len(df)} 行)")
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Excel报告生成完成!")
        logger.info(f"📁 文件位置: {output_file}")
        logger.info(f"📊 包含 {len(sheets)} 个分析Sheet")
        logger.info("=" * 80)
        
        return output_file


def main():
    """主函数"""
    base_dir = Path('d:/pythonProject/leisure/Employ26')
    
    generator = ExcelReportGenerator(base_dir)
    output_file = generator.generate_report()
    
    logger.info(f"\n可以打开文件查看: {output_file}")


if __name__ == '__main__':
    main()








