"""
快速测试脚本 - 验证技能提取模块
不需要GPU，只测试基本功能
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("技能提取模块 - 快速测试")
print("=" * 80)

# 测试1：检查数据文件
print("\n【测试1】检查数据文件")
print("-" * 80)

base_dir = Path(__file__).parent
nlp_dir = base_dir / 'output' / 'nlp_processed'

csv_files = list(nlp_dir.glob('*样本_1%.csv'))
if csv_files:
    print(f"✅ 找到 {len(csv_files)} 个数据文件:")
    for f in csv_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   - {f.name} ({size_mb:.1f} MB)")
else:
    print(f"❌ 未找到数据文件: {nlp_dir}")
    print("   请先运行: python src/nlp_analysis/text_preprocessing.py")
    sys.exit(1)

# 测试2：检查依赖
print("\n【测试2】检查依赖")
print("-" * 80)

dependencies = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'gensim': 'gensim',
    'tqdm': 'tqdm',
    'transformers': 'transformers (可选，用于BERT)',
    'torch': 'torch (可选，用于BERT)'
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError:
        if module in ['transformers', 'torch']:
            print(f"⚠️  {name} - 未安装（BERT功能不可用）")
        else:
            print(f"❌ {name} - 未安装")
            missing.append(module)

if missing:
    print(f"\n请安装缺失的依赖:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

# 测试3：检查GPU（如果有torch）
print("\n【测试3】检查GPU")
print("-" * 80)

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU可用: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
    else:
        print("⚠️  GPU不可用，将使用CPU")
        print("   BERT提取会较慢，但Word2Vec不受影响")
except ImportError:
    print("⚠️  torch未安装，无法检测GPU")
    print("   Word2Vec可以正常使用")

# 测试4：测试Word2Vec模块导入
print("\n【测试4】测试模块导入")
print("-" * 80)

try:
    from src.skill_extraction import Word2VecSkillExtractor
    print("✅ Word2VecSkillExtractor 导入成功")
    
    # 创建实例
    extractor = Word2VecSkillExtractor(base_dir)
    print(f"✅ 初始化成功，种子技能: {len(extractor.seed_skills)} 个")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from src.skill_extraction import BERTSkillExtractor
    print("✅ BERTSkillExtractor 导入成功")
except Exception as e:
    print(f"⚠️  BERTSkillExtractor 导入失败: {e}")
    print("   （如果未安装transformers，这是正常的）")

try:
    from src.skill_extraction import SkillExtractionPipeline
    print("✅ SkillExtractionPipeline 导入成功")
except Exception as e:
    print(f"❌ SkillExtractionPipeline 导入失败: {e}")

# 测试5：测试数据加载
print("\n【测试5】测试数据加载")
print("-" * 80)

try:
    from src.skill_extraction import Word2VecSkillExtractor
    
    extractor = Word2VecSkillExtractor(base_dir)
    sentences = extractor.load_data(use_sample=True)
    
    if sentences:
        print(f"✅ 数据加载成功")
        print(f"   句子数: {len(sentences):,}")
        print(f"   示例句子: {' '.join(sentences[0][:10])}...")
    else:
        print(f"❌ 数据加载失败")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

print("\n✅ 基础功能正常，可以运行Word2Vec")
print("\n运行命令:")
print("  python src/skill_extraction/word2vec_extractor.py")

try:
    import transformers
    import torch
    print("\n✅ BERT依赖已安装，可以运行完整流水线")
    print("\n运行命令:")
    print("  python src/skill_extraction/run_extraction_pipeline.py")
except ImportError:
    print("\n⚠️  BERT依赖未安装，只能运行Word2Vec")
    print("\n如需使用BERT，请安装:")
    print("  pip install transformers torch")

print("\n" + "=" * 80)








