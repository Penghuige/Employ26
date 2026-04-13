"""
数据整合模块
将职业名称解析结果合并到NLP处理后的数据集，并标准化时间、城市字段
支持处理全量数据和样本数据
"""

import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIntegrator:
    """数据整合器"""
    
    def __init__(self, base_dir=None, use_full_data=True):
        """初始化
        
        Args:
            base_dir: 项目根目录
            use_full_data: 是否使用全量数据（True=全量，False=样本）
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.use_full_data = use_full_data
        
        # 根据数据类型设置路径
        if use_full_data:
            # 全量数据路径
            self.nlp_dir = base_dir / 'output' / 'nlp_processed_full'
            self.occupation_dir = base_dir / 'output' / 'job_title_parsing'
        else:
            # 样本数据路径
            self.nlp_dir = base_dir / 'output' / 'nlp_processed'
            self.occupation_dir = base_dir / 'output' / 'job_title_parsing'
        
        self.output_dir = base_dir / 'output' / 'integrated'
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据整合器初始化完成")
        logger.info(f"  数据类型: {'全量数据' if use_full_data else '样本数据'}")
        logger.info(f"  NLP数据目录: {self.nlp_dir}")
        logger.info(f"  职业解析目录: {self.occupation_dir}")
        logger.info(f"  输出目录: {self.output_dir}")
    
    def load_occupation_mapping_from_files(self):
        """从职业解析文件中加载映射（支持全量数据）
        
        Returns:
            dict: {岗位名称: {occupation_core, occupation_category, confidence}}
        """
        logger.info("加载职业解析结果...")
        
        # 查找所有解析结果文件
        parsing_files = list(self.occupation_dir.glob('*_解析.csv'))
        
        if not parsing_files:
            logger.warning(f"未找到职业解析文件: {self.occupation_dir}")
            logger.info("尝试使用样本解析结果...")
            return self.load_occupation_mapping_from_sample()
        
        logger.info(f"找到 {len(parsing_files)} 个职业解析文件")
        
        # 合并所有解析结果
        all_mappings = {}
        
        for parsing_file in parsing_files:
            logger.info(f"  读取: {parsing_file.name}")
            # 指定 dtype 或使用 low_memory=False 避免警告
            df = pd.read_csv(parsing_file, encoding='utf-8', low_memory=False)
            
            # 检查必要字段
            if '岗位名称' not in df.columns:
                logger.warning(f"    ⚠️  缺少'岗位名称'字段，跳过")
                continue
            
            # 构建映射
            for _, row in df.iterrows():
                job_title = row['岗位名称']
                if pd.notna(job_title):
                    all_mappings[job_title] = {
                        'occupation_core': row.get('occupation_core'),
                        'occupation_category': row.get('occupation_category'),
                        'confidence': row.get('occupation_confidence', 1.0)
                    }
        
        logger.info(f"加载职业映射: {len(all_mappings):,} 条")
        return all_mappings
    
    def load_occupation_mapping_from_sample(self):
        """从样本解析结果加载映射（兼容旧版本）
        
        Returns:
            dict: {岗位名称: {occupation_core, occupation_category, confidence}}
        """
        occupation_file = self.occupation_dir / 'sample_parsing_results.csv'
        
        if not occupation_file.exists():
            logger.error(f"职业解析结果文件不存在: {occupation_file}")
            raise FileNotFoundError(f"找不到文件: {occupation_file}")
        
        logger.info(f"加载职业解析结果: {occupation_file}")
        df = pd.read_csv(occupation_file, encoding='utf-8')
        
        # 构建映射字典
        mapping = {}
        for _, row in df.iterrows():
            job_title = row['原始岗位名称']
            mapping[job_title] = {
                'occupation_core': row['职业核心词'],
                'occupation_category': row['类别'],
                'confidence': row['置信度']
            }
        
        logger.info(f"加载职业映射: {len(mapping):,} 条")
        return mapping
    
    def standardize_month(self, date_str):
        """标准化时间为年-月格式
        
        Args:
            date_str: 日期字符串
            
        Returns:
            str: YYYY-MM格式，或None
        """
        if pd.isna(date_str):
            return None
        
        try:
            dt = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(dt):
                return dt.strftime('%Y-%m')
        except:
            pass
        
        return None
    
    def standardize_city(self, city_str):
        """标准化城市名称
        
        Args:
            city_str: 城市字符串
            
        Returns:
            str: 标准化后的城市名，或None
        """
        if pd.isna(city_str):
            return None
        
        city_str = str(city_str)
        
        # 广东省主要城市列表
        cities = [
            '深圳', '广州', '佛山', '东莞', '惠州', '珠海', '中山', '江门', 
            '肇庆', '汕头', '湛江', '茂名', '韶关', '梅州', '清远', '阳江', 
            '河源', '云浮', '潮州', '揭阳', '汕尾'
        ]
        
        # 提取第一个匹配的城市
        for city in cities:
            if city in city_str:
                return city
        
        return '其他'
    
    def standardize_industry(self, industry_str):
        """标准化行业名称
        
        Args:
            industry_str: 行业字符串
            
        Returns:
            str: 标准化后的行业名，或None
        """
        if pd.isna(industry_str):
            return None
        
        industry_str = str(industry_str).strip()
        
        # 去除多余的分隔符和空格
        industry_str = re.sub(r'[,，/、]+', ',', industry_str)
        
        # 取第一个行业
        industries = industry_str.split(',')
        if industries:
            return industries[0].strip()
        
        return None
    
    def integrate_file(self, nlp_file, occupation_mapping):
        """整合单个文件
        
        Args:
            nlp_file: NLP处理后的CSV文件
            occupation_mapping: 职业映射字典
            
        Returns:
            DataFrame: 整合后的数据
        """
        logger.info(f"\n处理文件: {nlp_file.name}")
        
        # 读取NLP数据
        df = pd.read_csv(nlp_file, encoding='utf-8', low_memory=False)
        logger.info(f"  读取数据: {len(df):,} 行")
        
        # 检查必要字段
        if '岗位名称' not in df.columns:
            logger.error(f"  ❌ 缺少'岗位名称'字段")
            return None
        
        # 添加职业字段
        logger.info("  添加职业类别字段...")
        df['occupation_core'] = None
        df['occupation_category'] = None
        df['occupation_confidence'] = None
        
        matched_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  匹配职业"):
            job_title = row['岗位名称']
            if job_title in occupation_mapping:
                occ_info = occupation_mapping[job_title]
                df.at[idx, 'occupation_core'] = occ_info['occupation_core']
                df.at[idx, 'occupation_category'] = occ_info['occupation_category']
                df.at[idx, 'occupation_confidence'] = occ_info['confidence']
                matched_count += 1
        
        match_rate = matched_count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  职业匹配率: {match_rate:.2f}% ({matched_count:,}/{len(df):,})")
        
        # 标准化时间字段
        if '发布时间' in df.columns:
            logger.info("  标准化时间字段...")
            df['publish_month'] = df['发布时间'].apply(self.standardize_month)
            logger.info(f"  时间字段有效率: {df['publish_month'].notna().sum() / len(df) * 100:.2f}%")
        
        # 标准化城市字段
        if '工作城市' in df.columns:
            logger.info("  标准化城市字段...")
            df['city_clean'] = df['工作城市'].apply(self.standardize_city)
            logger.info(f"  城市字段有效率: {df['city_clean'].notna().sum() / len(df) * 100:.2f}%")
        
        # 标准化行业字段
        if '公司行业' in df.columns:
            logger.info("  标准化行业字段...")
            df['industry_clean'] = df['公司行业'].apply(self.standardize_industry)
            logger.info(f"  行业字段有效率: {df['industry_clean'].notna().sum() / len(df) * 100:.2f}%")
        
        return df
    
    def integrate_all(self):
        """整合所有文件"""
        logger.info("=" * 80)
        logger.info("数据整合 - 添加职业类别和标准化字段")
        logger.info("=" * 80)
        
        # 检查NLP目录
        if not self.nlp_dir.exists():
            logger.error(f"❌ NLP数据目录不存在: {self.nlp_dir}")
            logger.info("提示：请先运行 process_full_data_nlp.py 进行NLP处理")
            return
        
        # 加载职业映射
        try:
            occupation_mapping = self.load_occupation_mapping_from_files()
        except Exception as e:
            logger.error(f"❌ 加载职业映射失败: {e}")
            return
        
        # 查找所有NLP处理后的文件
        nlp_files = list(self.nlp_dir.glob('*_NLP处理.csv'))
        
        if not nlp_files:
            logger.warning(f"未找到NLP处理后的文件: {self.nlp_dir}")
            logger.info("提示：请先运行 process_full_data_nlp.py")
            return
        
        logger.info(f"\n找到 {len(nlp_files)} 个文件待处理")
        
        # 处理每个文件
        success_count = 0
        for nlp_file in nlp_files:
            try:
                # 整合数据
                df_integrated = self.integrate_file(nlp_file, occupation_mapping)
                
                if df_integrated is None:
                    continue
                
                # 生成输出文件名
                output_name = nlp_file.name.replace('_NLP处理', '_整合')
                output_file = self.output_dir / output_name
                
                # 保存
                logger.info(f"  保存到: {output_file.name}")
                df_integrated.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                logger.info(f"  ✅ 完成: {output_file.name}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  ❌ 处理失败: {nlp_file.name}")
                logger.error(f"     错误: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ 数据整合完成! 成功处理 {success_count}/{len(nlp_files)} 个文件")
        logger.info("=" * 80)
        logger.info(f"\n整合后的数据保存在: {self.output_dir}")
        logger.info("\n新增字段:")
        logger.info("  - occupation_core: 职业核心词（如：工程师、经理、专员）")
        logger.info("  - occupation_category: 职业类别（如：技术类、管理类、销售类）")
        logger.info("  - occupation_confidence: 职业匹配置信度")
        logger.info("  - publish_month: 发布月份 (YYYY-MM)")
        logger.info("  - city_clean: 标准化城市名")
        logger.info("  - industry_clean: 标准化行业名")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据整合')
    parser.add_argument('--sample', action='store_true',
                       help='使用样本数据（默认使用全量数据）')
    
    args = parser.parse_args()
    
    integrator = DataIntegrator(use_full_data=not args.sample)
    integrator.integrate_all()


if __name__ == '__main__':
    main()
