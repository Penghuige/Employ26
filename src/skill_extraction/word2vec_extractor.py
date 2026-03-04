"""
Word2Vec技能扩展模块
使用Word2Vec发现相似技能，扩展技能词典
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from gensim.models import Word2Vec
from collections import Counter
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Word2VecSkillExtractor:
    """基于Word2Vec的技能扩展器"""
    
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
        
        self.model = None
        
        # 基础技能种子词
        self.seed_skills = [
            # 编程语言
            'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go', 'SQL',
            # AI/大数据
            '机器学习', '深度学习', '数据分析', '大数据', '云计算', '人工智能',
            # 前后端
            '前端', '后端', '全栈', 'Web', 'APP',
            # 数据库
            'MySQL', 'Redis', 'MongoDB', 'Oracle',
            # 框架
            'Vue', 'React', 'Angular', 'Spring', 'Django',
            # 运维
            'Linux', 'Docker', 'Kubernetes',
            # 办公
            'Excel', 'PPT', 'Word', 'Photoshop', 'AutoCAD',
            # 软技能
            '产品经理', '项目管理', '市场营销', '数据运营'
        ]
        
        logger.info(f"Word2Vec技能扩展器初始化完成")
        logger.info(f"基础种子技能: {len(self.seed_skills)} 个")
    
    def load_data(self, use_sample=True):
        """加载数据
        
        Args:
            use_sample: 是否使用样本数据（默认True，用于验证）
        """
        logger.info("=" * 80)
        logger.info("加载数据")
        logger.info("=" * 80)
        
        all_sentences = []
        
        # 读取NLP处理后的数据
        csv_files = list(self.nlp_dir.glob('*样本_1%.csv'))
        
        if not csv_files:
            logger.error(f"未找到数据文件: {self.nlp_dir}")
            return []
        
        for csv_file in csv_files:
            logger.info(f"读取: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file, usecols=['岗位描述_分词'])
                
                # 提取分词后的句子
                for text in df['岗位描述_分词'].dropna():
                    words = text.split()
                    if len(words) > 3:  # 至少3个词
                        all_sentences.append(words)
                
                logger.info(f"  提取句子: {len(all_sentences):,} 条")
                
                # 如果是样本模式，只读取第一个文件
                if use_sample:
                    logger.info("  样本模式：只使用第一个文件")
                    break
                    
            except Exception as e:
                logger.error(f"读取文件失败: {csv_file.name}, 错误: {e}")
                continue
        
        logger.info(f"总共加载句子: {len(all_sentences):,} 条")
        
        return all_sentences
    
    def train_word2vec(self, sentences, save_model=True):
        """训练Word2Vec模型
        
        Args:
            sentences: 分词后的句子列表
            save_model: 是否保存模型
        """
        logger.info("\n" + "=" * 80)
        logger.info("训练Word2Vec模型")
        logger.info("=" * 80)
        
        if not sentences:
            logger.error("没有训练数据")
            return None
        
        logger.info(f"训练数据: {len(sentences):,} 条句子")
        
        # 训练参数
        params = {
            'vector_size': 300,      # 词向量维度
            'window': 5,             # 上下文窗口大小
            'min_count': 5,          # 最小词频
            'workers': 4,            # 并行线程数
            'epochs': 10,            # 训练轮数
            'sg': 1,                 # Skip-gram模型
            'negative': 5,           # 负采样
            'seed': 42               # 随机种子
        }
        
        logger.info(f"训练参数: {params}")
        
        # 训练模型
        logger.info("开始训练...")
        self.model = Word2Vec(sentences, **params)
        
        logger.info(f"训练完成!")
        logger.info(f"词汇表大小: {len(self.model.wv):,} 个词")
        
        # 保存模型
        if save_model:
            model_path = self.model_dir / 'word2vec_skills.model'
            self.model.save(str(model_path))
            logger.info(f"模型已保存: {model_path}")
        
        return self.model
    
    def load_model(self, model_path=None):
        """加载已训练的模型"""
        if model_path is None:
            model_path = self.model_dir / 'word2vec_skills.model'
        
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            return None
        
        logger.info(f"加载模型: {model_path}")
        self.model = Word2Vec.load(str(model_path))
        logger.info(f"模型加载完成，词汇表大小: {len(self.model.wv):,}")
        
        return self.model
    
    def find_similar_skills(self, skill, topn=20, threshold=0.5):
        """查找相似技能
        
        Args:
            skill: 技能词
            topn: 返回top-n个相似词
            threshold: 相似度阈值
        """
        if self.model is None:
            logger.error("模型未加载")
            return []
        
        try:
            similar = self.model.wv.most_similar(skill, topn=topn)
            # 过滤相似度低的
            filtered = [(word, score) for word, score in similar if score >= threshold]
            return filtered
        except KeyError:
            logger.warning(f"词汇不在模型中: {skill}")
            return []
    
    def expand_skill_dictionary(self, topn=20, threshold=0.5):
        """扩展技能词典
        
        Args:
            topn: 每个种子技能查找top-n个相似词
            threshold: 相似度阈值
        """
        logger.info("\n" + "=" * 80)
        logger.info("扩展技能词典")
        logger.info("=" * 80)
        
        if self.model is None:
            logger.error("模型未加载")
            return {}
        
        expanded_skills = {}
        all_new_skills = set()
        
        logger.info(f"基础种子技能: {len(self.seed_skills)} 个")
        logger.info(f"相似度阈值: {threshold}")
        logger.info(f"每个技能查找: Top {topn}")
        
        for skill in self.seed_skills:
            similar_skills = self.find_similar_skills(skill, topn=topn, threshold=threshold)
            
            if similar_skills:
                expanded_skills[skill] = similar_skills
                new_skills = [word for word, score in similar_skills]
                all_new_skills.update(new_skills)
                
                logger.info(f"\n{skill}:")
                for word, score in similar_skills[:5]:  # 只显示前5个
                    logger.info(f"  - {word:20s} (相似度: {score:.3f})")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"扩展结果:")
        logger.info(f"  原始技能: {len(self.seed_skills)} 个")
        logger.info(f"  新发现技能: {len(all_new_skills)} 个")
        logger.info(f"  总计: {len(self.seed_skills) + len(all_new_skills)} 个")
        logger.info("=" * 80)
        
        return expanded_skills
    
    def save_expanded_dictionary(self, expanded_skills, output_file=None):
        """保存扩展后的词典
        
        Args:
            expanded_skills: 扩展的技能字典
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = self.output_dir / 'word2vec_expanded_skills.json'
        
        # 保存JSON格式
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(expanded_skills, f, ensure_ascii=False, indent=2)
        
        logger.info(f"扩展词典已保存: {output_file}")
        
        # 生成jieba自定义词典格式
        dict_file = self.output_dir / 'word2vec_expanded_skills.txt'
        
        all_skills = set(self.seed_skills)
        for skill, similar_list in expanded_skills.items():
            for word, score in similar_list:
                all_skills.add(word)
        
        with open(dict_file, 'w', encoding='utf-8') as f:
            for skill in sorted(all_skills):
                # 格式：词汇 词频 词性
                f.write(f"{skill} 20000 nz\n")
        
        logger.info(f"jieba词典已保存: {dict_file}")
        logger.info(f"总计技能数: {len(all_skills)}")
        
        return dict_file
    
    def generate_report(self, expanded_skills):
        """生成分析报告"""
        report_file = self.output_dir / 'word2vec_expansion_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Word2Vec技能扩展报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"基础种子技能: {len(self.seed_skills)} 个\n\n")
            
            # 统计新发现的技能
            all_new_skills = set()
            for similar_list in expanded_skills.values():
                for word, score in similar_list:
                    all_new_skills.add(word)
            
            f.write(f"新发现技能: {len(all_new_skills)} 个\n")
            f.write(f"总计技能: {len(self.seed_skills) + len(all_new_skills)} 个\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("详细扩展结果\n")
            f.write("=" * 80 + "\n\n")
            
            for skill in self.seed_skills:
                if skill in expanded_skills:
                    f.write(f"\n{skill}:\n")
                    f.write("-" * 40 + "\n")
                    
                    for word, score in expanded_skills[skill]:
                        f.write(f"  {word:30s} 相似度: {score:.3f}\n")
        
        logger.info(f"分析报告已保存: {report_file}")
        
        return report_file
    
    def run_pipeline(self, use_sample=True, train_new=True):
        """运行完整流水线
        
        Args:
            use_sample: 是否使用样本数据
            train_new: 是否训练新模型（False则加载已有模型）
        """
        logger.info("\n" + "=" * 80)
        logger.info("Word2Vec技能扩展流水线")
        logger.info("=" * 80)
        
        # 1. 加载数据
        sentences = self.load_data(use_sample=use_sample)
        
        if not sentences:
            logger.error("数据加载失败")
            return None
        
        # 2. 训练或加载模型
        if train_new:
            self.train_word2vec(sentences, save_model=True)
        else:
            self.load_model()
        
        if self.model is None:
            logger.error("模型加载失败")
            return None
        
        # 3. 扩展技能词典
        expanded_skills = self.expand_skill_dictionary(topn=20, threshold=0.5)
        
        # 4. 保存结果
        dict_file = self.save_expanded_dictionary(expanded_skills)
        
        # 5. 生成报告
        report_file = self.generate_report(expanded_skills)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Word2Vec技能扩展完成!")
        logger.info("=" * 80)
        logger.info(f"扩展词典: {dict_file}")
        logger.info(f"分析报告: {report_file}")
        
        return expanded_skills


def main():
    """主函数"""
    # 创建扩展器
    extractor = Word2VecSkillExtractor()
    
    # 运行流水线（使用样本数据）
    expanded_skills = extractor.run_pipeline(
        use_sample=True,   # 使用样本数据验证
        train_new=True     # 训练新模型
    )
    
    return expanded_skills


if __name__ == '__main__':
    main()

