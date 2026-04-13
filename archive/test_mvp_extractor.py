"""
测试MVP技能抽取器
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("测试MVP技能抽取器")
print("=" * 80)

# 1. 测试词典管理器
print("\n【步骤1】测试词典管理器")
print("-" * 80)

try:
    sys.path.insert(0, str(Path(__file__).parent / 'dicts'))
    from dicts.dict_manager import DictManager
    
    manager = DictManager()
    
    # 加载技能种子
    skills = manager.load_skill_seeds()
    print(f"✅ 技能种子: {len(skills)} 个")
    
    # 加载黑名单
    blacklists = manager.load_blacklists()
    total_blacklist = sum(len(words) for words in blacklists.values())
    print(f"✅ 黑名单词: {total_blacklist} 个")
    
    # 加载同义词
    synonyms = manager.load_synonyms()
    print(f"✅ 同义词组: {len(synonyms)} 组")
    
    print("\n词典管理器测试通过！")
    
except Exception as e:
    print(f"❌ 词典管理器测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 测试MVP抽取器
print("\n【步骤2】测试MVP抽取器")
print("-" * 80)

try:
    from src.skill_extraction.mvp_skill_extractor import MVPSkillExtractor
    
    print("✅ 模块导入成功")
    
    # 创建抽取器
    extractor = MVPSkillExtractor()
    print("✅ 抽取器初始化成功")
    
    # 运行流水线（使用样本数据，限制1000行快速测试）
    print("\n开始运行MVP抽取流水线...")
    print("（使用样本数据，限制1000行进行快速测试）")
    print("=" * 80)
    
    skills = extractor.run_pipeline(
        use_sample=True,
        max_rows=1000,
        min_confidence=0.7,
        min_doc_count=3
    )
    
    if skills:
        print("\n" + "=" * 80)
        print("✅ MVP技能抽取测试通过！")
        print("=" * 80)
        print(f"\n最终技能数: {len(skills)} 个")
        
        # 显示分类统计
        from collections import Counter
        categories = Counter(s['category'] for s in skills.values())
        print("\n分类统计:")
        for cat, count in categories.most_common():
            print(f"  {cat:20s}: {count:4d} 个")
        
        # 显示Top 10技能
        sorted_skills = sorted(
            skills.items(),
            key=lambda x: x[1]['doc_count'],
            reverse=True
        )
        print("\nTop 10 技能:")
        for i, (skill, info) in enumerate(sorted_skills[:10], 1):
            print(f"  {i:2d}. {skill:15s} "
                  f"文档数:{info['doc_count']:4d}  "
                  f"置信度:{info['confidence']:.3f}  "
                  f"分类:{info['category']}")
        
        print("\n生成的文件:")
        print("  - output/skill_extraction/mvp_skills.json")
        print("  - output/skill_extraction/mvp_skills_jieba.txt")
        print("  - output/skill_extraction/mvp_skills.csv")
        print("  - output/skill_extraction/mvp_report.txt")
        
        print("\n下一步:")
        print("  1. 查看 mvp_report.txt 了解详细统计")
        print("  2. 打开 mvp_skills.csv 用Excel查看结果")
        print("  3. 随机抽取100个技能进行人工评估")
        print("  4. 如果精度达标，运行完整数据集（去掉max_rows限制）")
        
    else:
        print("\n❌ 技能抽取失败")
        sys.exit(1)
    
except Exception as e:
    print(f"\n❌ MVP抽取器测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("所有测试完成！")
print("=" * 80)
