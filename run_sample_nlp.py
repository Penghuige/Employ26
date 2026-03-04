"""
主运行脚本 - 样本数据NLP流程
步骤1: 提取10%样本数据
步骤2: NLP文本预处理
"""

import sys
from pathlib import Path

# 添加src到路径
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir / 'src'))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    logger.info("=" * 80)
    logger.info("广东省招聘数据NLP分析 - 样本数据处理流程")
    logger.info("=" * 80)
    
    # 步骤1: 提取样本数据
    logger.info("\n【步骤1/2】提取10%样本数据（均匀间隔采样）")
    logger.info("-" * 80)
    
    try:
        from preprocessing.sample_data import main as sample_main
        sample_main()
    except Exception as e:
        logger.error(f"样本提取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤2: NLP文本预处理
    logger.info("\n【步骤2/2】NLP文本预处理（清洗、分词、关键词提取）")
    logger.info("-" * 80)
    
    try:
        from ..nlp_analysis.text_preprocessing import main as nlp_main
        nlp_main()
    except Exception as e:
        logger.error(f"NLP处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 样本数据NLP流程完成!")
    logger.info("=" * 80)
    logger.info("\n下一步可以进行:")
    logger.info("  - 技能关键词统计分析")
    logger.info("  - 薪资影响因素分析")
    logger.info("  - 主题建模")
    logger.info("  - 词向量训练")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

