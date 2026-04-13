"""
完整分析流程 - 运行所有分析模块并生成整合Excel报告
"""

import sys
from pathlib import Path
import logging

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """运行完整分析流程"""
    
    logger.info("=" * 80)
    logger.info("广东省招聘数据 - 完整分析流程")
    logger.info("=" * 80)
    
    base_dir = Path(__file__).parent
    
    # 0. 数据整合（添加职业类别字段）
    logger.info(f"\n{'='*80}")
    logger.info(f"【0/5】数据整合 - 添加职业类别字段")
    logger.info(f"{'='*80}\n")
    try:
        from src.preprocessing.integrate_occupation import DataIntegrator
        integrator = DataIntegrator(base_dir)
        integrator.integrate_all()
        logger.info(f"\n✅ 数据整合完成!")
    except Exception as e:
        logger.error(f"\n❌ 数据整合失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 1. 职业类别薪资分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【1/7】职业类别薪资分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis.occupation_salary_analysis import OccupationSalaryAnalyzer
        analyzer = OccupationSalaryAnalyzer(base_dir)
        analyzer.run()
        logger.info(f"\n✅ 职业类别薪资分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 职业类别薪资分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 1.5. 学历需求分布分析（新增）
    logger.info(f"\n{'='*80}")
    logger.info(f"【1.5/7】学历需求分布分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis.education_distribution_analysis import EducationDistributionAnalyzer
        analyzer = EducationDistributionAnalyzer(base_dir)
        analyzer.run()
        logger.info(f"\n✅ 学历需求分布分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 学历需求分布分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 行业景气度分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【2/7】行业景气度分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis.industry_trend_analysis import IndustryTrendAnalyzer
        analyzer = IndustryTrendAnalyzer(base_dir)
        analyzer.run()
        logger.info(f"\n✅ 行业景气度分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 行业景气度分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 时间趋势分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【3/7】时间趋势分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis import time_trend_analysis
        time_trend_analysis.analyze_time_trends()
        logger.info(f"\n✅ 时间趋势分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 时间趋势分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 词云生成
    logger.info(f"\n{'='*80}")
    logger.info(f"【4/7】词云生成")
    logger.info(f"{'='*80}\n")
    try:
        from src.visualization import wordcloud_generator
        wordcloud_generator.generate_wordcloud_data()
        logger.info(f"\n✅ 词云生成完成!")
    except Exception as e:
        logger.error(f"\n❌ 词云生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 基础薪资分析（保留用于对比）
    logger.info(f"\n{'='*80}")
    logger.info(f"【5/8】基础薪资分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis import salary_analysis
        salary_analysis.main()
        logger.info(f"\n✅ 基础薪资分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 基础薪资分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 生成规范化汇总表（新增）
    logger.info(f"\n{'='*80}")
    logger.info(f"【6/8】生成规范化汇总表")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis.generate_standardized_tables import StandardizedTableGenerator
        generator = StandardizedTableGenerator(base_dir)
        generator.generate_all()
        logger.info(f"\n✅ 规范化汇总表生成完成!")
    except Exception as e:
        logger.error(f"\n❌ 规范化汇总表生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 生成Excel汇总报告
    logger.info(f"\n{'='*80}")
    logger.info(f"【7/8】生成Excel汇总报告")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis.generate_excel_summary import ExcelReportGenerator
        generator = ExcelReportGenerator(base_dir)
        generator.create_summary_report()
        logger.info(f"\n✅ Excel汇总报告生成完成!")
    except Exception as e:
        logger.error(f"\n❌ Excel汇总报告生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("🎉 所有分析完成!")
    logger.info("=" * 80)
    logger.info("\n📄 生成的文本报告:")
    logger.info("  - output/reports/职业类别薪资分析报告.txt")
    logger.info("  - output/reports/学历需求分布分析报告.txt")
    logger.info("  - output/reports/行业景气度分析报告.txt")
    logger.info("  - output/reports/时间趋势分析报告.txt")
    logger.info("  - output/reports/薪资分析报告.txt")
    
    logger.info("\n🎨 可视化图表:")
    logger.info("  - output/reports/职业类别薪资分析图.html")
    logger.info("  - output/reports/行业景气度分析图.html")
    logger.info("  - output/reports/词云图.html")
    logger.info("  - output/reports/时间趋势图.html")
    
    logger.info("\n📊 数据文件:")
    logger.info("  【薪资分析】")
    logger.info("  - output/reports/职业类别月度薪资数据.csv")
    logger.info("  - output/reports/职业月度薪资数据.csv")
    logger.info("  - output/reports/学历职业类别薪资数据.csv")
    logger.info("  - output/reports/学历职业薪资数据.csv")
    logger.info("  【学历分布分析】")
    logger.info("  - output/reports/职业类别年度学历分布.csv")
    logger.info("  - output/reports/职业年度学历分布.csv")
    logger.info("  - output/reports/职业类别月度学历分布.csv")
    logger.info("  - output/reports/职业月度学历分布.csv")
    logger.info("  【行业分析】")
    logger.info("  - output/reports/城市行业月度数据.csv")
    logger.info("  - output/reports/行业月度数据.csv")
    
    logger.info("\n📑 Excel汇总报告:")
    logger.info("  - output/reports/广东省招聘数据分析汇总报告.xlsx")
    
    logger.info("\n💡 提示：在浏览器中打开HTML文件查看交互式可视化!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
