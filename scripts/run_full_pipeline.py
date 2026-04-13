#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整数据处理与分析流程
从原始数据到最终Excel报告的一键式处理
支持灵活处理样本数据和总体数据
"""

import sys
from pathlib import Path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """运行完整流程：从原始数据到最终报告"""
    
    parser = argparse.ArgumentParser(description='完整数据处理与分析流程')
    parser.add_argument('--skip-nlp', action='store_true',
                       help='跳过NLP处理（如果已经处理过）')
    parser.add_argument('--skip-parsing', action='store_true',
                       help='跳过职业解析（如果已经解析过）')
    parser.add_argument('--skip-integration', action='store_true',
                       help='跳过数据整合（如果已经整合过）')
    parser.add_argument('--input', type=str, default='data',
                       help='原始数据目录（默认：data/）')
    parser.add_argument('--sample', action='store_true',
                       help='使用样本数据模式（默认：全量数据模式）')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    # 根据数据类型设置路径
    data_type = '样本数据' if args.sample else '全量数据'
    nlp_output_dir = 'output/nlp_processed' if args.sample else 'output/nlp_processed_full'
    
    logger.info("=" * 80)
    logger.info("广东省招聘数据 - 完整处理与分析流程")
    logger.info("=" * 80)
    logger.info(f"数据类型: {data_type}")
    logger.info(f"原始数据目录: {args.input}")
    logger.info(f"NLP输出目录: {nlp_output_dir}")
    logger.info(f"跳过NLP处理: {args.skip_nlp}")
    logger.info(f"跳过职业解析: {args.skip_parsing}")
    logger.info(f"跳过数据整合: {args.skip_integration}")
    logger.info("=" * 80)
    
    # ========================================
    # 阶段1：数据预处理（NLP）
    # ========================================
    
    if not args.skip_nlp:
        logger.info(f"\n{'='*80}")
        logger.info(f"【阶段1/4】NLP数据预处理")
        logger.info(f"{'='*80}\n")
        try:
            import process_full_data_nlp
            processor = process_full_data_nlp.FullDataNLPProcessor(
                input_dir=args.input,
                output_dir=nlp_output_dir
            )
            processor.process_all()
            logger.info(f"\n✅ NLP数据预处理完成!")
        except Exception as e:
            logger.error(f"\n❌ NLP数据预处理失败: {e}")
            import traceback
            traceback.print_exc()
            logger.error("\n提示：如果已经处理过，可以使用 --skip-nlp 跳过此步骤")
            return
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"【阶段1/4】NLP数据预处理 - 已跳过")
        logger.info(f"{'='*80}\n")
    
    # ========================================
    # 阶段2：职业名称解析
    # ========================================
    
    if not args.skip_parsing:
        logger.info(f"\n{'='*80}")
        logger.info(f"【阶段2/4】职业名称解析")
        logger.info(f"{'='*80}\n")
        try:
            import parse_all_occupations
            # 传递正确的输入目录
            parser_obj = parse_all_occupations.BatchOccupationParser(
                base_dir=base_dir,
                input_dir=nlp_output_dir
            )
            parser_obj.parse_all()
            logger.info(f"\n✅ 职业名称解析完成!")
        except Exception as e:
            logger.error(f"\n❌ 职业名称解析失败: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("\n⚠️  职业解析失败，但继续执行后续步骤")
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"【阶段2/4】职业名称解析 - 已跳过")
        logger.info(f"{'='*80}\n")
    
    # ========================================
    # 阶段3：数据整合
    # ========================================
    
    if not args.skip_integration:
        logger.info(f"\n{'='*80}")
        logger.info(f"【阶段3/4】数据整合")
        logger.info(f"{'='*80}\n")
        try:
            from src.preprocessing.integrate_occupation import DataIntegrator
            # 传递数据类型参数
            integrator = DataIntegrator(
                base_dir=base_dir,
                use_full_data=not args.sample
            )
            integrator.integrate_all()
            logger.info(f"\n✅ 数据整合完成!")
        except Exception as e:
            logger.error(f"\n❌ 数据整合失败: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("\n⚠️  数据整合失败，但继续执行后续步骤")
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"【阶段3/4】数据整合 - 已跳过")
        logger.info(f"{'='*80}\n")
    
    # ========================================
    # 阶段4：完整分析流程
    # ========================================
    
    logger.info(f"\n{'='*80}")
    logger.info(f"【阶段4/4】完整分析流程")
    logger.info(f"{'='*80}\n")
    
    # 添加src目录到路径
    sys.path.insert(0, str(base_dir / 'src'))
    
    # 1. 职业类别薪资分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.1】职业类别薪资分析")
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
    
    # 2. 学历需求分布分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.2】学历需求分布分析")
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
    
    # 3. 行业景气度分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.3】行业景气度分析")
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
    
    # 4. 时间趋势分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.4】时间趋势分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis import time_trend_analysis
        time_trend_analysis.analyze_time_trends()
        logger.info(f"\n✅ 时间趋势分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 时间趋势分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 词云生成
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.5】词云生成")
    logger.info(f"{'='*80}\n")
    try:
        from src.visualization import wordcloud_generator
        wordcloud_generator.generate_wordcloud_data()
        logger.info(f"\n✅ 词云生成完成!")
    except Exception as e:
        logger.error(f"\n❌ 词云生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 基础薪资分析
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.6】基础薪资分析")
    logger.info(f"{'='*80}\n")
    try:
        from src.analysis import salary_analysis
        salary_analysis.main()
        logger.info(f"\n✅ 基础薪资分析完成!")
    except Exception as e:
        logger.error(f"\n❌ 基础薪资分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 生成规范化汇总表
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.7】生成规范化汇总表")
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
    
    # 8. 生成Excel汇总报告
    logger.info(f"\n{'='*80}")
    logger.info(f"【4.8】生成Excel汇总报告")
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
    
    # ========================================
    # 总结
    # ========================================
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 完整流程执行完成!")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 数据类型: {data_type}")
    logger.info("\n生成的主要输出:")
    logger.info("  【数据处理】")
    logger.info(f"  - {nlp_output_dir}/              # NLP处理后的数据")
    logger.info("  - output/nlp_reports/               # NLP处理报告")
    logger.info("  - output/job_title_parsing/         # 职业解析结果")
    logger.info("  - output/integrated/                # 整合后的数据")
    
    logger.info("\n  【分析报告】")
    logger.info("  - output/reports/职业类别薪资分析报告.txt")
    logger.info("  - output/reports/学历需求分布分析报告.txt")
    logger.info("  - output/reports/行业景气度分析报告.txt")
    logger.info("  - output/reports/时间趋势分析报告.txt")
    logger.info("  - output/reports/薪资分析报告.txt")
    
    logger.info("\n  【可视化图表】")
    logger.info("  - output/reports/职业类别薪资分析图.html")
    logger.info("  - output/reports/行业景气度分析图.html")
    logger.info("  - output/reports/词云图.html")
    logger.info("  - output/reports/时间趋势图.html")
    
    logger.info("\n  【Excel汇总】")
    logger.info("  - output/reports/广东省招聘数据分析汇总报告.xlsx")
    
    logger.info("\n💡 提示：")
    logger.info("  - 如果已经完成NLP处理，下次可以使用 --skip-nlp 跳过")
    logger.info("  - 如果已经完成职业解析，下次可以使用 --skip-parsing 跳过")
    logger.info("  - 如果已经完成数据整合，下次可以使用 --skip-integration 跳过")
    logger.info("  - 使用 --sample 参数处理样本数据")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
