#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Excel 汇总报告。

用途:
- 将 `output/reports` 下已经生成的 CSV 与 Markdown 报告整理到一个总 Excel 中，便于交付或人工浏览。
- 这是“二次汇总脚本”，本身不直接分析原始岗位数据。

前置依赖:
- 建议先运行以下脚本，再执行本脚本:
  `occupation_salary_analysis.py`
  `education_distribution_analysis.py`
  `industry_trend_analysis.py`
  `generate_standardized_tables.py`

输入来源:
- `output/reports/职业类别月度薪资数据.csv`
- `output/reports/职业月度薪资数据.csv`
- `output/reports/学历职业类别薪资数据.csv`
- `output/reports/学历职业薪资数据.csv`
- `output/reports/学历月度趋势.csv`
- 以及目录中的多个 `*.md` 分析报告

输出文件:
- `output/reports/广东省招聘数据分析汇总报告.xlsx`

运行方式:
- `python -m src.analysis.generate_excel_summary`
- 或 `python src/analysis/generate_excel_summary.py`

维护说明:
- 该脚本主要负责汇总，不与 `src` 下其他建模/抽取脚本重复。
"""

import pandas as pd
from pathlib import Path
import logging
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExcelReportGenerator:
    """Excel汇总报告生成器"""
    
    def __init__(self, base_dir=None):
        """初始化"""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.reports_dir = base_dir / 'output' / 'reports'
        self.output_file = self.reports_dir / '广东省招聘数据分析汇总报告.xlsx'
        
        logger.info("Excel汇总报告生成器初始化完成")
    
    def create_summary_report(self):
        """创建汇总报告"""
        logger.info("=" * 80)
        logger.info("生成Excel汇总报告")
        logger.info("=" * 80)
        
        # 创建Excel写入器
        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            
            # ========================================
            # 职业维度统计（主口径）
            # ========================================
            
            # 1. 职业月度薪资（主口径 + 汇总口径合并）
            self._add_occupation_monthly_salary(writer)
            
            # 2. 职业学历薪资（主口径 + 汇总口径合并）
            self._add_occupation_education_salary(writer)
            
            # ========================================
            # 学历维度统计
            # ========================================
            
            # 3. 学历月度趋势（薪资）
            self._add_sheet(writer, '学历月度趋势.csv', '学历月度趋势')
            
            # 4. 职业类别年度学历分布（新增）
            self._add_sheet(writer, '职业类别年度学历分布.csv', '职业类别年度学历')
            
            # 5. 职业年度学历分布（新增）
            self._add_sheet(writer, '职业年度学历分布.csv', '职业年度学历')
            
            # 6. 职业类别月度学历分布（新增）
            self._add_sheet(writer, '职业类别月度学历分布.csv', '职业类别月度学历')
            
            # 7. 职业月度学历分布（新增）
            self._add_sheet(writer, '职业月度学历分布.csv', '职业月度学历')
            
            # ========================================
            # 行业维度统计
            # ========================================
            
            # 8. 城市行业月度数据
            self._add_sheet(writer, '城市行业月度数据.csv', '城市×行业月度')
            
            # 9. 行业月度数据
            self._add_sheet(writer, '行业月度数据.csv', '行业月度')
            
            # ========================================
            # 报告摘要
            # ========================================
            
            # 10. 添加 Markdown 报告摘要
            self._add_text_summary(writer)
        
        logger.info(f"\n✅ Excel汇总报告已生成: {self.output_file}")
        logger.info("=" * 80)
    
    def _add_occupation_monthly_salary(self, writer):
        """添加职业月度薪资（合并职业类别和职业两个维度）"""
        logger.info("  📊 合并职业月度薪资数据...")
        
        # 读取职业类别数据
        category_file = self.reports_dir / '职业类别月度薪资数据.csv'
        occupation_file = self.reports_dir / '职业月度薪资数据.csv'
        
        dfs = []
        
        # 职业类别数据（汇总口径）
        if category_file.exists():
            df_category = pd.read_csv(category_file, encoding='utf-8-sig')
            df_category['统计口径'] = '职业类别（汇总）'
            df_category['职业/类别'] = df_category['occupation_category']
            df_category = df_category[['统计口径', '职业/类别', 'publish_month', '平均薪资', '岗位数量']]
            dfs.append(df_category)
            logger.info(f"    ✅ 职业类别数据: {len(df_category)} 行")
        
        # 职业数据（主口径）
        if occupation_file.exists():
            df_occupation = pd.read_csv(occupation_file, encoding='utf-8-sig')
            df_occupation['统计口径'] = '职业（主口径）'
            df_occupation['职业/类别'] = df_occupation['occupation_core']
            df_occupation = df_occupation[['统计口径', '职业/类别', 'publish_month', '平均薪资', '岗位数量']]
            dfs.append(df_occupation)
            logger.info(f"    ✅ 职业数据: {len(df_occupation)} 行")
        
        if dfs:
            # 合并数据
            df_merged = pd.concat(dfs, ignore_index=True)
            
            # 重命名列
            df_merged.columns = ['统计口径', '职业/类别', '月份', '平均薪资(元)', '岗位数量']
            
            # 排序：先按统计口径，再按职业/类别，最后按月份
            df_merged = df_merged.sort_values(['统计口径', '职业/类别', '月份'])
            
            # 写入Excel
            df_merged.to_excel(writer, sheet_name='职业月度薪资', index=False)
            
            # 格式化
            worksheet = writer.sheets['职业月度薪资']
            self._format_worksheet(worksheet, df_merged)
            
            logger.info(f"    ✅ 已合并: 职业月度薪资 ({len(df_merged)} 行)")
        else:
            logger.warning(f"    ⚠️  未找到职业月度薪资数据文件")
    
    def _add_occupation_education_salary(self, writer):
        """添加职业学历薪资（合并职业类别和职业两个维度）"""
        logger.info("  📊 合并职业学历薪资数据...")
        
        # 读取数据
        category_file = self.reports_dir / '学历职业类别薪资数据.csv'
        occupation_file = self.reports_dir / '学历职业薪资数据.csv'
        
        dfs = []
        
        # 职业类别数据（汇总口径）
        if category_file.exists():
            df_category = pd.read_csv(category_file, encoding='utf-8-sig')
            df_category['统计口径'] = '职业类别（汇总）'
            df_category['职业/类别'] = df_category['occupation_category']
            df_category = df_category[['统计口径', '职业/类别', '学历', '平均薪资', '岗位数量']]
            dfs.append(df_category)
            logger.info(f"    ✅ 职业类别数据: {len(df_category)} 行")
        
        # 职业数据（主口径）
        if occupation_file.exists():
            df_occupation = pd.read_csv(occupation_file, encoding='utf-8-sig')
            df_occupation['统计口径'] = '职业（主口径）'
            df_occupation['职业/类别'] = df_occupation['occupation_core']
            df_occupation = df_occupation[['统计口径', '职业/类别', '学历', '平均薪资', '岗位数量']]
            dfs.append(df_occupation)
            logger.info(f"    ✅ 职业数据: {len(df_occupation)} 行")
        
        if dfs:
            # 合并数据
            df_merged = pd.concat(dfs, ignore_index=True)
            
            # 重命名列
            df_merged.columns = ['统计口径', '职业/类别', '学历', '平均薪资(元)', '岗位数量']
            
            # 排序：先按统计口径，再按职业/类别，最后按学历
            education_order = {'博士': 1, '硕士': 2, '本科': 3, '大专': 4, '高中': 5, '中专': 6}
            df_merged['学历排序'] = df_merged['学历'].map(education_order).fillna(99)
            df_merged = df_merged.sort_values(['统计口径', '职业/类别', '学历排序'])
            df_merged = df_merged.drop('学历排序', axis=1)
            
            # 写入Excel
            df_merged.to_excel(writer, sheet_name='职业学历薪资', index=False)
            
            # 格式化
            worksheet = writer.sheets['职业学历薪资']
            self._format_worksheet(worksheet, df_merged)
            
            logger.info(f"    ✅ 已合并: 职业学历薪资 ({len(df_merged)} 行)")
        else:
            logger.warning(f"    ⚠️  未找到职业学历薪资数据文件")
    
    def _add_sheet(self, writer, csv_filename, sheet_name):
        """添加CSV数据到Excel工作表"""
        csv_file = self.reports_dir / csv_filename
        
        if not csv_file.exists():
            logger.warning(f"  ⚠️  文件不存在，跳过: {csv_filename}")
            return
        
        try:
            # 读取CSV
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 写入Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 格式化工作表
            worksheet = writer.sheets[sheet_name]
            self._format_worksheet(worksheet, df)
            
            logger.info(f"  ✅ 已添加: {sheet_name} ({len(df)} 行)")
            
        except Exception as e:
            logger.error(f"  ❌ 添加失败: {sheet_name} - {e}")
    
    def _format_worksheet(self, worksheet, df):
        """格式化工作表"""
        # 设置标题行样式
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # 自动调整列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    def _resolve_report_file(self, markdown_filename):
        """优先读取 Markdown 报告，兼容历史 TXT 报告。"""
        markdown_path = self.reports_dir / markdown_filename
        if markdown_path.exists():
            return markdown_path
        legacy_path = markdown_path.with_suffix('.txt')
        if legacy_path.exists():
            return legacy_path
        return None
    
    def _add_text_summary(self, writer):
        """添加报告摘要"""
        logger.info("  📝 添加报告摘要...")
        
        # 读取 Markdown 报告，必要时兼容历史 TXT 报告。
        summaries = []
        
        report_files = [
            ('职业类别薪资分析报告.md', '职业薪资分析'),
            ('学历需求分布分析报告.md', '学历需求分布'),
            ('行业景气度分析报告.md', '行业景气度'),
        ]
        
        for filename, title in report_files:
            report_file = self._resolve_report_file(filename)
            if report_file is not None:
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 提取前500字符作为摘要
                        summary = content[:500] + "..." if len(content) > 500 else content
                        summaries.append({
                            '报告名称': title,
                            '摘要': summary
                        })
                except Exception as e:
                    logger.warning(f"  ⚠️  读取报告失败: {filename} - {e}")
        
        if summaries:
            df_summary = pd.DataFrame(summaries)
            df_summary.to_excel(writer, sheet_name='报告摘要', index=False)
            
            worksheet = writer.sheets['报告摘要']
            self._format_worksheet(worksheet, df_summary)
            
            logger.info(f"  ✅ 已添加: 报告摘要 ({len(summaries)} 个报告)")


def main():
    """主函数"""
    generator = ExcelReportGenerator()
    generator.create_summary_report()


if __name__ == '__main__':
    main()
