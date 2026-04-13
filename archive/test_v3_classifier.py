"""
测试V3技能分类器
"""

import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("测试V3技能分类器（机器学习方案）")
print("=" * 80)

try:
    from src.skill_extraction.v3_skill_classifier import SkillClassifierV3
    
    print("\n[OK] 模块导入成功")
    
    # 创建分类器
    print("\n初始化V3分类器...")
    classifier = SkillClassifierV3()
    print("[OK] 分类器初始化成功")
    
    # 运行流水线
    print("\n" + "=" * 80)
    print("开始运行V3流水线...")
    print("（使用样本数据，5000条进行快速测试）")
    print("=" * 80)
    
    new_skills = classifier.run_pipeline(
        use_sample=True,
        max_rows=5000
    )
    
    if new_skills:
        print("\n" + "=" * 80)
        print("[OK] V3技能分类器测试通过！")
        print("=" * 80)
        
        print(f"\n发现新技能: {len(new_skills)} 个")
        
        # 按类别统计
        from collections import Counter
        categories = Counter(s['category'] for s in new_skills.values())
        print("\n按类别统计:")
        for cat, count in categories.most_common():
            print(f"  {cat:20s}: {count:4d} 个")
        
        # 显示高置信度技能
        sorted_skills = sorted(
            new_skills.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        print("\nTop 20 高置信度新技能:")
        for i, (word, info) in enumerate(sorted_skills[:20], 1):
            print(f"  {i:2d}. {word:20s} "
                  f"置信度:{info['confidence']:.3f}  "
                  f"文档数:{info['doc_freq']:4d}  "
                  f"类别:{info['category']}")
        
        print("\n生成的文件:")
        print("  - output/skill_extraction/v3_discovered_skills.txt")
        print("  - models/skill_classifier_v3.pkl")
        print("  - models/word2vec_skills.model")
        
        print("\n下一步:")
        print("  1. 查看 v3_discovered_skills.txt 中发现的新技能")
        print("  2. 人工审核高置信度技能")
        print("  3. 将确认的技能添加到 dicts/skill_seeds.txt")
        print("  4. 测试单个词的预测：")
        print("     >>> classifier.predict_skill('Rust')")
        print("     >>> classifier.predict_skill('沟通能力')")
        
        # 测试几个词
        print("\n" + "=" * 80)
        print("测试单词预测")
        print("=" * 80)
        
        test_words = ['Rust', 'Golang', 'TypeScript', '沟通能力', '负责', '项目管理']
        
        for word in test_words:
            is_skill, confidence, category = classifier.predict_skill(word)
            status = "[SKILL]" if is_skill else "[NOT-SKILL]"
            print(f"{word:15s} -> {status}  置信度:{confidence:.3f}  {category}")
        
    else:
        print("\n[ERROR] 未发现新技能")
    
except Exception as e:
    print(f"\n[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
