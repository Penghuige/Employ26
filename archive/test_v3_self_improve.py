"""
测试V3自我迭代功能
"""

import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("测试V3技能分类器 - 自我迭代功能")
print("=" * 80)

try:
    from src.skill_extraction.v3_skill_classifier import SkillClassifierV3
    from src.skill_extraction.v3_self_improve import create_self_improving_pipeline
    
    print("\n[OK] 模块导入成功")
    
    # 1. 加载已训练的分类器
    print("\n" + "=" * 80)
    print("步骤1: 加载V3分类器")
    print("=" * 80)
    
    classifier = SkillClassifierV3()
    
    # 加载已保存的模型
    model_file = Path('models/skill_classifier_v3.pkl')
    if model_file.exists():
        import joblib
        classifier.skill_classifier = joblib.load(model_file)
        print(f"[OK] 已加载模型: {model_file}")
    else:
        print("[WARN] 模型文件不存在，需要先运行 test_v3_classifier.py")
        sys.exit(1)
    
    # 加载Word2Vec模型
    w2v_file = Path('models/word2vec_skills.model')
    if w2v_file.exists():
        from gensim.models import Word2Vec
        classifier.word2vec_model = Word2Vec.load(str(w2v_file))
        print(f"[OK] 已加载Word2Vec: {w2v_file}")
    
    # 2. 读取发现的新技能
    print("\n" + "=" * 80)
    print("步骤2: 读取发现的新技能")
    print("=" * 80)
    
    skills_file = Path('output/skill_extraction/v3_discovered_skills.txt')
    new_skills = {}
    
    with open(skills_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 解析: Mysql 30000 nz database # 置信度:0.967 文档数:13
            parts = line.split('#')
            if len(parts) < 2:
                continue
            
            word_info = parts[0].strip().split()
            if len(word_info) < 4:
                continue
            
            word = word_info[0]
            category = word_info[3]
            
            # 解析置信度和文档数
            meta = parts[1].strip()
            confidence = float(meta.split('置信度:')[1].split()[0])
            doc_freq = int(meta.split('文档数:')[1])
            
            new_skills[word] = {
                'confidence': confidence,
                'doc_freq': doc_freq,
                'category': category
            }
    
    print(f"[OK] 读取到 {len(new_skills)} 个新技能")
    
    # 3. 创建自我改进流水线
    print("\n" + "=" * 80)
    print("步骤3: 自动验证新技能")
    print("=" * 80)
    
    pipeline = create_self_improving_pipeline(classifier)
    validator = pipeline['validator']
    iterator = pipeline['iterator']
    learner = pipeline['learner']
    
    # 自动验证
    validation_results = iterator.auto_validate_batch(new_skills)
    
    print(f"\n验证结果:")
    print(f"  高置信度（自动通过）: {len(validation_results['high_confidence'])} 个")
    print(f"  需要人工审核: {len(validation_results['need_review'])} 个")
    print(f"  自动拒绝: {len(validation_results['rejected'])} 个")
    
    # 4. 显示详细结果
    print("\n" + "=" * 80)
    print("高置信度技能（自动通过）")
    print("=" * 80)
    
    for item in validation_results['high_confidence']:
        word = item['word']
        score = item['validation_score']
        info = item['info']
        reason = item['reason']
        print(f"  {word:20s} 验证分数:{score:.3f}  原始:{info['confidence']:.3f}  {reason}")
    
    print("\n" + "=" * 80)
    print("需要人工审核")
    print("=" * 80)
    
    for item in validation_results['need_review']:
        word = item['word']
        score = item['validation_score']
        info = item['info']
        reason = item['reason']
        print(f"  {word:20s} 验证分数:{score:.3f}  原始:{info['confidence']:.3f}  {reason}")
    
    print("\n" + "=" * 80)
    print("自动拒绝")
    print("=" * 80)
    
    for item in validation_results['rejected']:
        word = item['word']
        score = item['validation_score']
        info = item['info']
        reason = item['reason']
        print(f"  {word:20s} 验证分数:{score:.3f}  原始:{info['confidence']:.3f}  {reason}")
    
    # 5. 生成验证报告
    print("\n" + "=" * 80)
    print("步骤4: 生成验证报告")
    print("=" * 80)
    
    report_file = Path('output/skill_extraction/v3_validation_report.txt')
    iterator.generate_review_report(validation_results, report_file)
    print(f"[OK] 验证报告: {report_file}")
    
    # 6. 主动学习 - 选择需要标注的样本
    print("\n" + "=" * 80)
    print("步骤5: 主动学习 - 选择不确定样本")
    print("=" * 80)
    
    # 模拟更多候选词（实际应该从数据中提取）
    uncertain_samples = learner.select_uncertain_samples(new_skills, n_samples=5)
    print(f"[OK] 选择了 {len(uncertain_samples)} 个不确定样本用于标注")
    
    for word, info in uncertain_samples:
        print(f"  {word:20s} 置信度:{info['confidence']:.3f}  文档数:{info['doc_freq']:4d}")
    
    # 生成标注任务
    annotation_file = Path('output/skill_extraction/v3_annotation_task.txt')
    learner.generate_annotation_task(uncertain_samples, annotation_file)
    print(f"[OK] 标注任务: {annotation_file}")
    
    # 7. 模拟增量训练
    print("\n" + "=" * 80)
    print("步骤6: 增量训练（模拟）")
    print("=" * 80)
    
    # 自动确认高置信度技能
    confirmed_skills = [item['word'] for item in validation_results['high_confidence']]
    
    if confirmed_skills:
        print(f"将 {len(confirmed_skills)} 个高置信度技能加入训练集:")
        for skill in confirmed_skills:
            print(f"  - {skill}")
        
        # 注意：这里不实际执行增量训练，因为需要重新加载数据
        print("\n[INFO] 增量训练需要重新加载数据，这里仅演示流程")
        print("[INFO] 实际使用时调用: iterator.incremental_train(confirmed_skills)")
    else:
        print("[INFO] 没有高置信度技能需要加入训练集")
    
    # 8. 总结
    print("\n" + "=" * 80)
    print("自我迭代流程总结")
    print("=" * 80)
    
    print("\n自动验收策略:")
    print("  1. 多维度验证: 置信度 + 特征模式 + 文档频率 + 类别一致性")
    print("  2. 三级分类: 高置信度（自动通过）/ 需审核 / 自动拒绝")
    print("  3. 黑名单过滤: 排除泛化词（如Server、能力等）")
    print("  4. 特征加分: 驼峰命名、全大写、包含数字等")
    
    print("\n迭代改进流程:")
    print("  1. 自动验证 -> 生成验证报告")
    print("  2. 高置信度技能 -> 自动加入训练集")
    print("  3. 不确定样本 -> 主动学习标注")
    print("  4. 人工审核 -> 增量训练")
    print("  5. 重复循环 -> 持续改进")
    
    print("\n质量保证:")
    print(f"  - 原始准确率: ~60% (6/10)")
    print(f"  - 自动验证后: 高置信度 {len(validation_results['high_confidence'])} 个")
    print(f"  - 需人工审核: {len(validation_results['need_review'])} 个")
    print(f"  - 自动拒绝: {len(validation_results['rejected'])} 个")
    
    print("\n下一步:")
    print("  1. 查看验证报告: output/skill_extraction/v3_validation_report.txt")
    print("  2. 审核需要人工确认的技能")
    print("  3. 完成标注任务: output/skill_extraction/v3_annotation_task.txt")
    print("  4. 运行增量训练，持续改进模型")
    
except Exception as e:
    print(f"\n[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
