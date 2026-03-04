"""
技能提取主流程
整合Word2Vec和BERT两种方法，生成最终的扩展词典
"""

import pandas as pd
from pathlib import Path
import logging
import json
from collections import Counter

from word2vec_extractor import Word2VecSkillExtractor
from bert_extractor import BERTSkillExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SkillExtractionPipeline:
    """技能提取流水线"""
    
    def __init__(self, base_dir=None):
        """初始化"""
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.output_dir = self.base_dir / 'output' / 'skill_extraction'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化两个提取器
        self.w2v_extractor = Word2VecSkillExtractor(base_dir)
        self.bert_extractor = BERTSkillExtractor(base_dir)
        
        logger.info("技能提取流水线初始化完成")
    
    def run_word2vec_stage(self, use_sample=True):
        """阶段1：Word2Vec快速扩展
        
        Args:
            use_sample: 是否使用样本数据
        """
        logger.info("\n" + "=" * 80)
        logger.info("【阶段1】Word2Vec快速扩展")
        logger.info("=" * 80)
        
        try:
            expanded_skills = self.w2v_extractor.run_pipeline(
                use_sample=use_sample,
                train_new=True
            )
            
            if expanded_skills:
                logger.info("✅ Word2Vec阶段完成")
                return expanded_skills
            else:
                logger.warning("⚠️  Word2Vec阶段未返回结果")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Word2Vec阶段失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_bert_stage(self, use_sample=True, max_rows=1000):
        """阶段2：BERT命名实体识别
        
        Args:
            use_sample: 是否使用样本数据
            max_rows: 最大处理行数
        """
        logger.info("\n" + "=" * 80)
        logger.info("【阶段2】BERT命名实体识别")
        logger.info("=" * 80)
        
        try:
            skill_counter = self.bert_extractor.run_pipeline(
                use_sample=use_sample,
                max_rows=max_rows
            )
            
            if skill_counter:
                logger.info("✅ BERT阶段完成")
                return skill_counter
            else:
                logger.warning("⚠️  BERT阶段未返回结果")
                return Counter()
                
        except Exception as e:
            logger.error(f"❌ BERT阶段失败: {e}")
            import traceback
            traceback.print_exc()
            return Counter()
    
    def merge_results(self, w2v_skills, bert_skills):
        """合并两个阶段的结果
        
        Args:
            w2v_skills: Word2Vec扩展的技能字典
            bert_skills: BERT提取的技能计数器
        """
        logger.info("\n" + "=" * 80)
        logger.info("【阶段3】合并结果")
        logger.info("=" * 80)
        
        # 收集所有技能
        all_skills = set()
        skill_sources = {}  # 记录技能来源
        
        # 1. 添加Word2Vec的种子技能
        seed_skills = self.w2v_extractor.seed_skills
        for skill in seed_skills:
            all_skills.add(skill)
            skill_sources[skill] = ['seed']
        
        logger.info(f"种子技能: {len(seed_skills)} 个")
        
        # 2. 添加Word2Vec扩展的技能
        w2v_count = 0
        if w2v_skills:
            for skill, similar_list in w2v_skills.items():
                for word, score in similar_list:
                    all_skills.add(word)
                    if word not in skill_sources:
                        skill_sources[word] = []
                    skill_sources[word].append(f'w2v({score:.2f})')
                    w2v_count += 1
        
        logger.info(f"Word2Vec扩展: {w2v_count} 个")
        
        # 3. 添加BERT提取的技能
        bert_count = 0
        if bert_skills:
            for skill, count in bert_skills.items():
                all_skills.add(skill)
                if skill not in skill_sources:
                    skill_sources[skill] = []
                skill_sources[skill].append(f'bert({count}次)')
                bert_count += 1
        
        logger.info(f"BERT提取: {bert_count} 个")
        
        # 4. 统计
        logger.info(f"\n合并结果:")
        logger.info(f"  总技能数: {len(all_skills)} 个")
        
        # 统计来源
        both_methods = sum(1 for sources in skill_sources.values() 
                          if any('w2v' in s for s in sources) and any('bert' in s for s in sources))
        only_w2v = sum(1 for sources in skill_sources.values() 
                      if any('w2v' in s for s in sources) and not any('bert' in s for s in sources))
        only_bert = sum(1 for sources in skill_sources.values() 
                       if any('bert' in s for s in sources) and not any('w2v' in s for s in sources))
        
        logger.info(f"  两种方法都发现: {both_methods} 个")
        logger.info(f"  仅Word2Vec: {only_w2v} 个")
        logger.info(f"  仅BERT: {only_bert} 个")
        
        return all_skills, skill_sources
    
    def save_final_dictionary(self, all_skills, skill_sources):
        """保存最终的扩展词典
        
        Args:
            all_skills: 所有技能集合
            skill_sources: 技能来源信息
        """
        logger.info("\n" + "=" * 80)
        logger.info("保存最终词典")
        logger.info("=" * 80)
        
        # 1. 保存jieba词典格式
        dict_file = self.output_dir / 'final_expanded_skills.txt'
        with open(dict_file, 'w', encoding='utf-8') as f:
            for skill in sorted(all_skills):
                f.write(f"{skill} 20000 nz\n")
        
        logger.info(f"jieba词典: {dict_file}")
        logger.info(f"  包含技能: {len(all_skills)} 个")
        
        # 2. 保存详细信息（JSON）
        detail_file = self.output_dir / 'final_skills_detail.json'
        skill_detail = {
            skill: {
                'sources': skill_sources.get(skill, [])
            }
            for skill in sorted(all_skills)
        }
        
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(skill_detail, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细信息: {detail_file}")
        
        # 3. 生成对比报告
        report_file = self.output_dir / 'final_extraction_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("技能提取最终报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"总技能数: {len(all_skills)} 个\n\n")
            
            # 统计来源
            both = [s for s, sources in skill_sources.items() 
                   if any('w2v' in src for src in sources) and any('bert' in src for src in sources)]
            only_w2v = [s for s, sources in skill_sources.items() 
                       if any('w2v' in src for src in sources) and not any('bert' in src for src in sources)]
            only_bert = [s for s, sources in skill_sources.items() 
                        if any('bert' in src for src in sources) and not any('w2v' in src for src in sources)]
            
            f.write(f"两种方法都发现: {len(both)} 个\n")
            f.write(f"仅Word2Vec发现: {len(only_w2v)} 个\n")
            f.write(f"仅BERT发现: {len(only_bert)} 个\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("技能列表（按字母排序）\n")
            f.write("=" * 80 + "\n\n")
            
            for skill in sorted(all_skills):
                sources = skill_sources.get(skill, [])
                sources_str = ', '.join(sources)
                f.write(f"{skill:30s} - {sources_str}\n")
        
        logger.info(f"对比报告: {report_file}")
        
        return dict_file
    
    def run_full_pipeline(self, use_sample=True, max_rows=1000):
        """运行完整流水线
        
        Args:
            use_sample: 是否使用样本数据
            max_rows: BERT最大处理行数
        """
        logger.info("\n" + "=" * 80)
        logger.info("技能提取完整流水线")
        logger.info("=" * 80)
        logger.info(f"模式: {'样本数据' if use_sample else '全量数据'}")
        logger.info(f"BERT处理行数: {max_rows}")
        
        # 阶段1：Word2Vec
        w2v_skills = self.run_word2vec_stage(use_sample=use_sample)
        
        # 阶段2：BERT
        bert_skills = self.run_bert_stage(use_sample=use_sample, max_rows=max_rows)
        
        # 阶段3：合并结果
        all_skills, skill_sources = self.merge_results(w2v_skills, bert_skills)
        
        # 阶段4：保存最终词典
        dict_file = self.save_final_dictionary(all_skills, skill_sources)
        
        # 最终总结
        logger.info("\n" + "=" * 80)
        logger.info("🎉 技能提取流水线完成!")
        logger.info("=" * 80)
        logger.info(f"最终技能数: {len(all_skills)} 个")
        logger.info(f"最终词典: {dict_file}")
        logger.info("\n生成的文件:")
        logger.info(f"  - {self.output_dir / 'word2vec_expanded_skills.txt'}")
        logger.info(f"  - {self.output_dir / 'bert_extracted_skills.txt'}")
        logger.info(f"  - {self.output_dir / 'final_expanded_skills.txt'}")
        logger.info(f"  - {self.output_dir / 'final_extraction_report.txt'}")
        logger.info("=" * 80)
        
        return all_skills


def main():
    """主函数"""
    # 创建流水线
    pipeline = SkillExtractionPipeline()
    
    # 运行完整流水线
    # 使用样本数据，快速验证可行性
    all_skills = pipeline.run_full_pipeline(
        use_sample=True,   # 使用样本数据
        max_rows=1000      # BERT只处理1000行（快速测试）
    )
    
    return all_skills


if __name__ == '__main__':
    main()

