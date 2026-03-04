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
    
    # 1. 薪资分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【1/5】薪资分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis import salary_analysis
        salary_analysis.main()
        logger.info(f"\n✅ 薪资分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 薪资分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 技能组合分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【2/5】技能组合分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis import skill_combination
        skill_combination.analyze_skill_combinations()
        logger.info(f"\n✅ 技能组合分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 技能组合分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 时间趋势分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【3/5】时间趋势分析")
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
    logger.info(f"【4/5】词云生成")
    logger.info(f"{'='*80}\n")
    try:
        from src.visualization import wordcloud_generator
        wordcloud_generator.generate_wordcloud_data()
        logger.info(f"\n✅ 词云生成完成!")
    except Exception as e:
        logger.error(f"\n❌ 词云生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 生成整合Excel报告
    logger.info(f"\n{'='*80}")
    logger.info(f"【5/5】生成整合Excel报告")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis.generate_excel_report import ExcelReportGenerator
        generator = ExcelReportGenerator(base_dir)
        excel_file = generator.generate_report()
        logger.info(f"\n✅ Excel报告生成完成!")
    except Exception as e:
        logger.error(f"\n❌ Excel报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        excel_file = None
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("🎉 所有分析完成!")
    logger.info("=" * 80)
    logger.info("\n📄 生成的文本报告:")
    logger.info("  - output/reports/薪资分析报告.txt")
    logger.info("  - output/reports/技能组合分析报告.txt")
    logger.info("  - output/reports/时间趋势分析报告.txt")
    logger.info("  - output/reports/关键词统计.txt")
    
    if excel_file:
        logger.info("\n📊 整合Excel报告:")
        logger.info(f"  - {excel_file.relative_to(base_dir)}")
        logger.info("  包含9个Sheet：概览、技能薪资、学历薪资、经验薪资、城市薪资、")
        logger.info("              技能组合、技能三元组、年度趋势、技能趋势")
    
    logger.info("\n🎨 可视化图表:")
    logger.info("  - output/reports/词云图.html")
    logger.info("  - output/reports/技能关系网络图.html")
    logger.info("  - output/reports/时间趋势图.html")
    logger.info("\n💡 提示：在浏览器中打开HTML文件查看交互式可视化!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
