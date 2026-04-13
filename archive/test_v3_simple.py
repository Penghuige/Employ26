# -*- coding: utf-8 -*-
"""
测试V3技能分类器 - 简化版
"""

import sys
from pathlib import Path

# 设置UTF-8输出
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Test V3 Skill Classifier (Machine Learning)")
print("=" * 80)

try:
    from src.skill_extraction.v3_skill_classifier import SkillClassifierV3
    
    print("\nOK: Module imported")
    
    # 创建分类器
    print("\nInitializing V3 classifier...")
    classifier = SkillClassifierV3()
    print("OK: Classifier initialized")
    
    # 运行流水线
    print("\n" + "=" * 80)
    print("Running V3 pipeline...")
    print("(Using sample data, 5000 rows for quick test)")
    print("=" * 80)
    
    new_skills = classifier.run_pipeline(
        use_sample=True,
        max_rows=5000
    )
    
    if new_skills:
        print("\n" + "=" * 80)
        print("SUCCESS: V3 Skill Classifier Test Passed!")
        print("=" * 80)
        
        print(f"\nDiscovered new skills: {len(new_skills)}")
        
        # 按类别统计
        from collections import Counter
        categories = Counter(s['category'] for s in new_skills.values())
        print("\nBy category:")
        for cat, count in categories.most_common():
            print(f"  {cat:20s}: {count:4d}")
        
        # 显示高置信度技能
        sorted_skills = sorted(
            new_skills.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        print("\nTop 20 high confidence new skills:")
        for i, (word, info) in enumerate(sorted_skills[:20], 1):
            print(f"  {i:2d}. {word:20s} "
                  f"conf:{info['confidence']:.3f}  "
                  f"docs:{info['doc_freq']:4d}  "
                  f"cat:{info['category']}")
        
        print("\nGenerated files:")
        print("  - output/skill_extraction/v3_discovered_skills.txt")
        print("  - models/skill_classifier_v3.pkl")
        print("  - models/word2vec_skills.model")
        
        print("\nNext steps:")
        print("  1. Check v3_discovered_skills.txt")
        print("  2. Review high confidence skills")
        print("  3. Add confirmed skills to dicts/skill_seeds.txt")
        print("  4. Test single word prediction:")
        print("     >>> classifier.predict_skill('Rust')")
        
        # 测试几个词
        print("\n" + "=" * 80)
        print("Test single word prediction")
        print("=" * 80)
        
        test_words = ['Rust', 'Golang', 'TypeScript', 'communication', 'responsible', 'project management']
        
        for word in test_words:
            is_skill, confidence, category = classifier.predict_skill(word)
            status = "OK" if is_skill else "NO"
            print(f"{word:20s} -> {status}  conf:{confidence:.3f}  {category}")
        
    else:
        print("\nNO new skills discovered")
    
except Exception as e:
    print(f"\nERROR: Test failed")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
