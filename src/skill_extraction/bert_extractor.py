"""
BERT命名实体识别模块
使用BERT模型进行技能实体识别
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from collections import Counter
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BERTSkillExtractor:
    """基于BERT的技能实体识别器"""
    
    def __init__(self, base_dir=None):
        """初始化"""
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.nlp_dir = self.base_dir / 'output' / 'nlp_processed'
        self.output_dir = self.base_dir / 'output' / 'skill_extraction'
        self.model_dir = self.base_dir / 'models'
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        
        logger.info(f"BERT技能提取器初始化完成")
    
    def check_gpu(self):
        """检查GPU可用性"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU可用: {gpu_name}")
            logger.info(f"   显存: {gpu_memory:.1f} GB")
            return True
        else:
            logger.warning("⚠️  GPU不可用，将使用CPU（速度较慢）")
            return False
    
    def load_model(self, model_name="ckiplab/bert-base-chinese-ner"):
        """加载BERT-NER模型
        
        Args:
            model_name: 模型名称或路径
        """
        logger.info("\n" + "=" * 80)
        logger.info("加载BERT-NER模型")
        logger.info("=" * 80)
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            logger.info(f"模型: {model_name}")
            logger.info("正在下载/加载模型...")
            
            # 检查GPU
            device = 0 if self.check_gpu() else -1
            
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # 创建NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                aggregation_strategy="simple"  # 自动合并子词
            )
            
            logger.info("✅ 模型加载完成!")
            
            return True
            
        except ImportError as e:
            logger.error("❌ 缺少依赖库，请安装:")
            logger.error("   pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.error("提示: 如果是网络问题，可以:")
            logger.error("   1. 使用镜像源下载模型")
            logger.error("   2. 手动下载模型到本地，然后指定路径")
            return False
    
    def extract_entities_from_text(self, text, max_length=512):
        """从文本中提取实体
        
        Args:
            text: 输入文本
            max_length: 最大文本长度
        """
        if self.ner_pipeline is None:
            logger.error("模型未加载")
            return []
        
        if not text or len(text.strip()) == 0:
            return []
        
        # 截断过长文本
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            # 使用pipeline提取实体
            results = self.ner_pipeline(text)
            
            # 提取实体词
            entities = []
            for item in results:
                entity_text = item['word'].strip()
                entity_type = item['entity_group']
                score = item['score']
                
                # 过滤太短的实体和低置信度的
                if len(entity_text) >= 2 and score >= 0.5:
                    entities.append({
                        'text': entity_text,
                        'type': entity_type,
                        'score': score
                    })
            
            return entities
            
        except Exception as e:
            logger.debug(f"提取失败: {e}")
            return []
    
    def load_data(self, use_sample=True, max_rows=None):
        """加载数据
        
        Args:
            use_sample: 是否使用样本数据
            max_rows: 最大行数（用于快速测试）
        """
        logger.info("\n" + "=" * 80)
        logger.info("加载数据")
        logger.info("=" * 80)
        
        all_data = []
        
        # 读取NLP处理后的数据
        csv_files = list(self.nlp_dir.glob('*样本_1%.csv'))
        
        if not csv_files:
            logger.error(f"未找到数据文件: {self.nlp_dir}")
            return pd.DataFrame()
        
        for csv_file in csv_files:
            logger.info(f"读取: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file, usecols=['岗位名称', '岗位描述'])
                
                if max_rows:
                    df = df.head(max_rows)
                
                all_data.append(df)
                logger.info(f"  加载: {len(df):,} 行")
                
                # 如果是样本模式，只读取第一个文件
                if use_sample:
                    logger.info("  样本模式：只使用第一个文件")
                    break
                    
            except Exception as e:
                logger.error(f"读取文件失败: {csv_file.name}, 错误: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"总共加载: {len(df):,} 行")
        
        return df
    
    def batch_extract_skills(self, df, batch_size=100):
        """批量提取技能
        
        Args:
            df: 数据DataFrame
            batch_size: 批处理大小
        """
        logger.info("\n" + "=" * 80)
        logger.info("批量提取技能实体")
        logger.info("=" * 80)
        
        if self.ner_pipeline is None:
            logger.error("模型未加载")
            return []
        
        all_entities = []
        skill_counter = Counter()
        
        total = len(df)
        logger.info(f"待处理: {total:,} 条岗位描述")
        logger.info(f"批处理大小: {batch_size}")
        
        # 批量处理
        for i in range(0, total, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            logger.info(f"处理进度: {i+1}-{min(i+batch_size, total)}/{total}")
            
            for idx, row in batch_df.iterrows():
                job_title = row['岗位名称']
                job_desc = row['岗位描述']
                
                if pd.isna(job_desc):
                    continue
                
                # 提取实体
                entities = self.extract_entities_from_text(str(job_desc))
                
                if entities:
                    all_entities.append({
                        '岗位名称': job_title,
                        '提取实体': entities
                    })
                    
                    # 统计技能频率
                    for entity in entities:
                        skill_counter[entity['text']] += 1
        
        logger.info(f"\n提取完成:")
        logger.info(f"  处理岗位: {total:,} 个")
        logger.info(f"  提取实体: {len(all_entities):,} 个岗位有实体")
        logger.info(f"  唯一技能: {len(skill_counter):,} 个")
        
        return all_entities, skill_counter
    
    def save_results(self, all_entities, skill_counter):
        """保存提取结果
        
        Args:
            all_entities: 所有提取的实体
            skill_counter: 技能计数器
        """
        logger.info("\n" + "=" * 80)
        logger.info("保存提取结果")
        logger.info("=" * 80)
        
        # 1. 保存详细提取结果（JSON）
        json_file = self.output_dir / 'bert_extracted_entities.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_entities, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存: {json_file}")
        
        # 2. 保存技能统计（CSV）
        csv_file = self.output_dir / 'bert_skill_statistics.csv'
        skill_df = pd.DataFrame([
            {'技能': skill, '出现次数': count}
            for skill, count in skill_counter.most_common()
        ])
        skill_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"技能统计已保存: {csv_file}")
        
        # 3. 生成jieba词典格式
        dict_file = self.output_dir / 'bert_extracted_skills.txt'
        with open(dict_file, 'w', encoding='utf-8') as f:
            for skill, count in skill_counter.most_common():
                # 根据出现频率设置词频
                freq = min(count * 100, 20000)
                f.write(f"{skill} {freq} nz\n")
        logger.info(f"jieba词典已保存: {dict_file}")
        
        return dict_file
    
    def generate_report(self, skill_counter):
        """生成分析报告"""
        report_file = self.output_dir / 'bert_extraction_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BERT命名实体识别报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"提取技能总数: {len(skill_counter):,} 个\n")
            f.write(f"总出现次数: {sum(skill_counter.values()):,} 次\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Top 100 高频技能\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (skill, count) in enumerate(skill_counter.most_common(100), 1):
                f.write(f"{i:3d}. {skill:30s} - {count:6,} 次\n")
        
        logger.info(f"分析报告已保存: {report_file}")
        
        return report_file
    
    def run_pipeline(self, use_sample=True, max_rows=1000):
        """运行完整流水线
        
        Args:
            use_sample: 是否使用样本数据
            max_rows: 最大处理行数（用于快速测试）
        """
        logger.info("\n" + "=" * 80)
        logger.info("BERT技能提取流水线")
        logger.info("=" * 80)
        
        # 1. 加载模型
        success = self.load_model()
        if not success:
            logger.error("模型加载失败，流程终止")
            logger.info("\n提示: 请先安装依赖:")
            logger.info("  pip install transformers torch")
            logger.info("\n如果有GPU，还需要安装:")
            logger.info("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return None
        
        # 2. 加载数据
        df = self.load_data(use_sample=use_sample, max_rows=max_rows)
        
        if df.empty:
            logger.error("数据加载失败")
            return None
        
        # 3. 批量提取技能
        all_entities, skill_counter = self.batch_extract_skills(df, batch_size=100)
        
        # 4. 保存结果
        dict_file = self.save_results(all_entities, skill_counter)
        
        # 5. 生成报告
        report_file = self.generate_report(skill_counter)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ BERT技能提取完成!")
        logger.info("=" * 80)
        logger.info(f"提取技能: {len(skill_counter):,} 个")
        logger.info(f"技能词典: {dict_file}")
        logger.info(f"分析报告: {report_file}")
        
        return skill_counter


def main():
    """主函数"""
    # 创建提取器
    extractor = BERTSkillExtractor()
    
    # 运行流水线（使用样本数据，快速测试）
    skill_counter = extractor.run_pipeline(
        use_sample=True,   # 使用样本数据
        max_rows=1000      # 只处理1000行（快速测试）
    )
    
    return skill_counter


if __name__ == '__main__':
    main()

