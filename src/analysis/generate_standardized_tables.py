#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成规范化汇总表。

用途:
- 将 `output/reports` 中已有的分析产物重新整理为更稳定、列名更统一的交付表。
- 同时补一张 `学历月度趋势.csv`，方便后续 Excel 汇总或外部消费。

前置依赖:
- `职业学历薪资`、`职业月度薪资` 两张表依赖 `occupation_salary_analysis.py` 先生成源 CSV。
- `学历月度趋势` 直接读取 `output/integrated/*_整合_*.csv`，因此也依赖
  `src/preprocessing/integrate_occupation.py` 先完成数据整合。

主要输出:
- `output/reports/structured_analysis_{mm-dd}/standardized_salary_by_education_occupation.csv`
- `output/reports/structured_analysis_{mm-dd}/standardized_salary_by_occupation_month.csv`
- `output/reports/structured_analysis_{mm-dd}/standardized_salary_by_education_month.csv`

运行方式:
- `python -m src.analysis.generate_standardized_tables`

维护说明:
- 这是当前目录中的“二次整理层”，主要解决交付口径统一问题。
- `parse_salary` 逻辑与其他分析脚本存在重复，后续如继续维护可考虑抽到公共工具模块。
"""

import logging
import re
from pathlib import Path

import pandas as pd

from src.analysis.structured_common import write_csv_with_legacy_copy

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


class StandardizedTableGenerator:
    """规范化汇总表生成器"""
    
    def __init__(self, base_dir=None, output_dir=None):
        """初始化"""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.reports_dir = Path(output_dir) if output_dir is not None else base_dir / 'output' / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("规范化汇总表生成器初始化完成")
    
    def generate_occupation_degree_salary_summary(self):
        """生成汇总表：职业学历薪资
        
        目标列：职业、职业类别、学历、平均薪资、岗位数量
        """
        logger.info("\n生成汇总表：职业学历薪资")
        logger.info("=" * 60)
        
        # 读取职业数据
        occupation_file = self._resolve_input_file(
            'salary_by_education_occupation.csv',
            '学历职业薪资数据.csv',
        )
        category_file = self._resolve_input_file(
            'salary_by_education_occupation_category.csv',
            '学历职业类别薪资数据.csv',
        )
        
        if not occupation_file.exists() or not category_file.exists():
            logger.error("  错误：缺少必要的源文件")
            return None
        
        # 读取职业数据（主口径）
        df_occupation = pd.read_csv(occupation_file, encoding='utf-8-sig')
        logger.info(f"  读取职业数据: {len(df_occupation)} 行")
        
        # 读取职业类别数据（用于补充职业类别字段）
        df_category = pd.read_csv(category_file, encoding='utf-8-sig')
        logger.info(f"  读取职业类别数据: {len(df_category)} 行")
        
        # 处理职业数据：添加职业类别字段
        # 从原始数据中获取职业到职业类别的映射
        occupation_to_category = self._load_occupation_category_mapping()
        
        occupation_column = 'occupation_core'
        education_column = 'education_level' if 'education_level' in df_occupation.columns else '学历'
        salary_column = 'avg_salary' if 'avg_salary' in df_occupation.columns else '平均薪资'
        count_column = 'job_count' if 'job_count' in df_occupation.columns else '岗位数量'
        df_occupation['职业类别'] = df_occupation[occupation_column].map(occupation_to_category)
        
        # 重命名和选择列
        df_result = df_occupation[[
            occupation_column,
            '职业类别',
            education_column,
            salary_column,
            count_column,
        ]].copy()
        
        # 重命名列为标准名称
        df_result.columns = ['职业', '职业类别', '学历', '平均薪资', '岗位数量']
        
        # 数据清洗：过滤空值
        original_count = len(df_result)
        df_result = df_result[
            df_result['职业'].notna() &
            df_result['职业类别'].notna() &
            df_result['学历'].notna() &
            df_result['平均薪资'].notna()
        ].copy()
        
        filtered_count = original_count - len(df_result)
        if filtered_count > 0:
            logger.info(f"  过滤空值记录: {filtered_count} 行")
        
        # 排序：按职业类别、职业、学历
        education_order = {'博士': 1, '硕士': 2, '本科': 3, '大专': 4, '高中': 5, '中专': 6}
        df_result['学历排序'] = df_result['学历'].map(education_order).fillna(99)
        df_result = df_result.sort_values(['职业类别', '职业', '学历排序'])
        df_result = df_result.drop('学历排序', axis=1)
        
        # 保存
        output_files = write_csv_with_legacy_copy(
            df_result.rename(
                columns={'职业': 'occupation_core', '职业类别': 'occupation_category', '学历': 'education_level', '平均薪资': 'avg_salary', '岗位数量': 'job_count'}
            ),
            self.reports_dir,
            canonical_filename='standardized_salary_by_education_occupation.csv',
            legacy_filename='职业学历薪资.csv',
        )
        output_file = self.reports_dir / output_files[0]
        
        logger.info(f"  成功：已生成 {output_file.name}")
        logger.info(f"  总记录数: {len(df_result)}")
        logger.info(f"  职业数量: {df_result['职业'].nunique()}")
        logger.info(f"  职业类别数量: {df_result['职业类别'].nunique()}")
        
        return df_result
    
    def generate_occupation_monthly_salary_summary(self):
        """生成汇总表：职业月度薪资
        
        目标列：职业、职业类别、月度、平均薪资、岗位数量
        """
        logger.info("\n生成汇总表：职业月度薪资")
        logger.info("=" * 60)
        
        # 读取职业数据
        occupation_file = self._resolve_input_file(
            'salary_by_occupation_month.csv',
            '职业月度薪资数据.csv',
        )
        
        if not occupation_file.exists():
            logger.error("  错误：缺少必要的源文件")
            return None
        
        # 读取职业数据（主口径）
        df_occupation = pd.read_csv(occupation_file, encoding='utf-8-sig')
        logger.info(f"  读取职业数据: {len(df_occupation)} 行")
        
        # 添加职业类别字段
        occupation_to_category = self._load_occupation_category_mapping()
        occupation_column = 'occupation_core'
        month_column = 'publish_month'
        salary_column = 'avg_salary' if 'avg_salary' in df_occupation.columns else '平均薪资'
        count_column = 'job_count' if 'job_count' in df_occupation.columns else '岗位数量'
        df_occupation['职业类别'] = df_occupation[occupation_column].map(occupation_to_category)
        
        # 重命名和选择列
        df_result = df_occupation[[
            occupation_column,
            '职业类别',
            month_column,
            salary_column,
            count_column,
        ]].copy()
        
        # 重命名列为标准名称
        df_result.columns = ['职业', '职业类别', '月度', '平均薪资', '岗位数量']
        
        # 数据清洗：过滤空值
        original_count = len(df_result)
        df_result = df_result[
            df_result['职业'].notna() &
            df_result['职业类别'].notna() &
            df_result['月度'].notna() &
            df_result['平均薪资'].notna()
        ].copy()
        
        filtered_count = original_count - len(df_result)
        if filtered_count > 0:
            logger.info(f"  过滤空值记录: {filtered_count} 行")
        
        # 排序：按职业类别、职业、月度
        df_result = df_result.sort_values(['职业类别', '职业', '月度'])
        
        # 保存
        output_files = write_csv_with_legacy_copy(
            df_result.rename(
                columns={'职业': 'occupation_core', '职业类别': 'occupation_category', '月度': 'publish_month', '平均薪资': 'avg_salary', '岗位数量': 'job_count'}
            ),
            self.reports_dir,
            canonical_filename='standardized_salary_by_occupation_month.csv',
            legacy_filename='职业月度薪资.csv',
        )
        output_file = self.reports_dir / output_files[0]
        
        logger.info(f"  成功：已生成 {output_file.name}")
        logger.info(f"  总记录数: {len(df_result)}")
        logger.info(f"  职业数量: {df_result['职业'].nunique()}")
        logger.info(f"  职业类别数量: {df_result['职业类别'].nunique()}")
        logger.info(f"  月度数量: {df_result['月度'].nunique()}")
        
        return df_result
    
    def generate_education_monthly_trend(self):
        """生成汇总表：学历月度趋势
        
        目标列：学历、月度、平均薪资、岗位数量
        分析不同学历随时间的薪资变化和岗位需求变化
        """
        logger.info("\n生成汇总表：学历月度趋势")
        logger.info("=" * 60)
        
        # 从整合数据中读取原始数据
        integrated_dir = self.base_dir / 'output' / 'integrated'
        
        all_data = []
        for csv_file in integrated_dir.glob('*_整合_*.csv'):
            # 读取所有数据
            df = pd.read_csv(csv_file, encoding='utf-8')
            # 选择需要的列
            df = df[['学历要求', 'publish_month', '薪资水平']].copy()
            all_data.append(df)
        
        if not all_data:
            logger.error("  错误：未找到整合数据文件")
            return None
        
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"  读取原始数据: {len(df):,} 行")
        
        # 解析薪资
        df[['最低薪资', '最高薪资']] = df['薪资水平'].apply(
            lambda x: pd.Series(parse_salary(x))
        )
        df['平均薪资'] = (df['最低薪资'] + df['最高薪资']) / 2
        
        # 过滤有效数据
        df = df[
            df['学历要求'].notna() &
            df['publish_month'].notna() &
            df['平均薪资'].notna()
        ].copy()
        
        logger.info(f"  有效数据: {len(df):,} 行")
        
        # 提取学历（标准化）
        education_levels = ['博士', '硕士', '本科', '大专', '高中', '中专']
        
        def extract_education(edu_str):
            """从学历要求中提取标准学历"""
            if pd.isna(edu_str):
                return None
            edu_str = str(edu_str)
            for edu in education_levels:
                if edu in edu_str:
                    return edu
            return None
        
        df['学历'] = df['学历要求'].apply(extract_education)
        
        # 过滤有学历的数据
        df = df[df['学历'].notna()].copy()
        logger.info(f"  提取学历后: {len(df):,} 行")
        
        # 按学历和月度聚合
        df_grouped = df.groupby(['学历', 'publish_month'])
        df_result = df_grouped.agg({
            '平均薪资': 'mean',
            '薪资水平': 'count'  # 计数作为岗位数量
        }).reset_index()
        
        # 重命名列
        df_result.columns = ['学历', '月度', '平均薪资', '岗位数量']
        
        # 只保留岗位数量>=10的数据点（确保统计可靠性）
        df_result = df_result[df_result['岗位数量'] >= 10].copy()
        
        logger.info(f"  过滤后数据点: {len(df_result)} 个")
        
        # 排序：按学历、月度
        education_order = {'博士': 1, '硕士': 2, '本科': 3, '大专': 4, '高中': 5, '中专': 6}
        df_result['学历排序'] = df_result['学历'].map(education_order).fillna(99)
        df_result = df_result.sort_values(['学历排序', '月度'])
        df_result = df_result.drop('学历排序', axis=1)
        
        # 保存
        output_files = write_csv_with_legacy_copy(
            df_result.rename(
                columns={'学历': 'education_level', '月度': 'publish_month', '平均薪资': 'avg_salary', '岗位数量': 'job_count'}
            ),
            self.reports_dir,
            canonical_filename='standardized_salary_by_education_month.csv',
            legacy_filename='学历月度趋势.csv',
        )
        output_file = self.reports_dir / output_files[0]
        
        logger.info(f"  成功：已生成 {output_file.name}")
        logger.info(f"  总记录数: {len(df_result)}")
        logger.info(f"  学历类型: {df_result['学历'].unique().tolist()}")
        logger.info(f"  月度数量: {df_result['月度'].nunique()}")
        
        # 显示统计信息
        logger.info("\n  各学历数据点统计:")
        for edu in education_levels:
            count = len(df_result[df_result['学历'] == edu])
            if count > 0:
                logger.info(f"    {edu}: {count} 个月度数据点")
        
        return df_result
    
    def _load_occupation_category_mapping(self):
        """加载职业到职业类别的映射
        
        从整合数据中提取映射关系
        """
        integrated_dir = self.base_dir / 'output' / 'integrated'
        
        mapping = {}
        
        # 读取所有整合数据文件
        for csv_file in integrated_dir.glob('*_整合_*.csv'):
            df = pd.read_csv(csv_file, encoding='utf-8', usecols=['occupation_core', 'occupation_category'])
            df = df[df['occupation_core'].notna() & df['occupation_category'].notna()]
            
            # 构建映射（一个职业对应一个类别）
            for _, row in df.iterrows():
                occupation = row['occupation_core']
                category = row['occupation_category']
                if occupation not in mapping:
                    mapping[occupation] = category
        
        logger.info(f"  加载职业类别映射: {len(mapping)} 个职业")
        
        return mapping

    def _resolve_input_file(self, canonical_filename, legacy_filename):
        """优先读取规范文件名，兼容历史中文文件名。"""
        canonical_path = self.reports_dir / canonical_filename
        if canonical_path.exists():
            return canonical_path
        return self.reports_dir / legacy_filename
    
    def generate_all(self):
        """生成所有规范化汇总表"""
        logger.info("=" * 80)
        logger.info("生成规范化汇总表")
        logger.info("=" * 80)
        
        # 生成职业学历薪资表
        df1 = self.generate_occupation_degree_salary_summary()
        
        # 生成职业月度薪资表
        df2 = self.generate_occupation_monthly_salary_summary()
        
        # 生成学历月度趋势表（新增）
        df3 = self.generate_education_monthly_trend()
        
        logger.info("\n" + "=" * 80)
        logger.info("成功：所有规范化汇总表生成完成!")
        logger.info("=" * 80)
        
        return df1, df2, df3


def main():
    """主函数"""
    generator = StandardizedTableGenerator()
    generator.generate_all()


if __name__ == '__main__':
    main()
