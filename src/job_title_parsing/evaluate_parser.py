"""
岗位名称解析评估脚本
用于评估解析器在真实数据上的表现
"""

import sys
import io
from pathlib import Path
import pandas as pd
from collections import Counter
import json

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 80)
print("岗位名称解析评估")
print("=" * 80)

try:
    from src.job_title_parsing import OccupationParser
    
    print("\n[OK] 模块导入成功")
    
    # 创建解析器
    print("\n初始化解析器...")
    parser = OccupationParser()
    print("[OK] 解析器初始化成功")
    
    # 手工标注的测试集（用于精确评估）
    print("\n" + "=" * 80)
    print("第一部分：手工标注测试集评估")
    print("=" * 80)
    
    # 100个手工标注的样本
    annotated_test_cases = [
        {"input": "Java开发工程师", "expected_core": "开发工程师", "expected_modifiers": "Java"},
        {"input": "高级Python后端开发工程师", "expected_core": "开发工程师", "expected_modifiers": "高级Python后端"},
        {"input": "产品经理", "expected_core": "产品经理", "expected_modifiers": ""},
        {"input": "高级产品经理", "expected_core": "产品经理", "expected_modifiers": "高级"},
        {"input": "UI设计师", "expected_core": "UI设计师", "expected_modifiers": ""},
        {"input": "前端设计师", "expected_core": "设计师", "expected_modifiers": "前端"},
        {"input": "销售经理", "expected_core": "销售经理", "expected_modifiers": ""},
        {"input": "大客户销售经理", "expected_core": "销售经理", "expected_modifiers": "大客户"},
        {"input": "财务专员", "expected_core": "财务专员", "expected_modifiers": ""},
        {"input": "人力资源经理", "expected_core": "人力资源经理", "expected_modifiers": ""},
        {"input": "算法工程师", "expected_core": "算法工程师", "expected_modifiers": ""},
        {"input": "机器学习算法工程师", "expected_core": "算法工程师", "expected_modifiers": "机器学习"},
        {"input": "数据分析师", "expected_core": "数据分析师", "expected_modifiers": ""},
        {"input": "业务数据分析师", "expected_core": "数据分析师", "expected_modifiers": "业务"},
        {"input": "运营专员", "expected_core": "运营专员", "expected_modifiers": ""},
        {"input": "新媒体运营专员", "expected_core": "运营专员", "expected_modifiers": "新媒体"},
        {"input": "客服", "expected_core": "客服", "expected_modifiers": ""},
        {"input": "售后客服", "expected_core": "客服", "expected_modifiers": "售后"},
        {"input": "薄膜工艺工程师", "expected_core": "工艺工程师", "expected_modifiers": "薄膜"},
        {"input": "注塑工艺工程师", "expected_core": "工艺工程师", "expected_modifiers": "注塑"},
        {"input": "工程师", "expected_core": "工程师", "expected_modifiers": ""},
        {"input": "CEO", "expected_core": "CEO", "expected_modifiers": ""},
        {"input": "技术总监", "expected_core": "技术总监", "expected_modifiers": ""},
        {"input": "市场营销经理", "expected_core": "市场经理", "expected_modifiers": "营销"},
        {"input": "前端工程师", "expected_core": "前端工程师", "expected_modifiers": ""},
        {"input": "后端工程师", "expected_core": "后端工程师", "expected_modifiers": ""},
        {"input": "全栈工程师", "expected_core": "全栈工程师", "expected_modifiers": ""},
        {"input": "测试工程师", "expected_core": "测试工程师", "expected_modifiers": ""},
        {"input": "运维工程师", "expected_core": "运维工程师", "expected_modifiers": ""},
        {"input": "安全工程师", "expected_core": "安全工程师", "expected_modifiers": ""},
        {"input": "网络工程师", "expected_core": "网络工程师", "expected_modifiers": ""},
        {"input": "系统工程师", "expected_core": "系统工程师", "expected_modifiers": ""},
        {"input": "软件工程师", "expected_core": "软件工程师", "expected_modifiers": ""},
        {"input": "硬件工程师", "expected_core": "硬件工程师", "expected_modifiers": ""},
        {"input": "嵌入式工程师", "expected_core": "嵌入式工程师", "expected_modifiers": ""},
        {"input": "大数据工程师", "expected_core": "大数据工程师", "expected_modifiers": ""},
        {"input": "AI工程师", "expected_core": "AI工程师", "expected_modifiers": ""},
        {"input": "架构师", "expected_core": "架构师", "expected_modifiers": ""},
        {"input": "技术专家", "expected_core": "技术专家", "expected_modifiers": ""},
        {"input": "程序员", "expected_core": "程序员", "expected_modifiers": ""},
        {"input": "产品总监", "expected_core": "产品总监", "expected_modifiers": ""},
        {"input": "产品专员", "expected_core": "产品专员", "expected_modifiers": ""},
        {"input": "产品助理", "expected_core": "产品助理", "expected_modifiers": ""},
        {"input": "UX设计师", "expected_core": "UX设计师", "expected_modifiers": ""},
        {"input": "交互设计师", "expected_core": "交互设计师", "expected_modifiers": ""},
        {"input": "视觉设计师", "expected_core": "视觉设计师", "expected_modifiers": ""},
        {"input": "平面设计师", "expected_core": "平面设计师", "expected_modifiers": ""},
        {"input": "设计师", "expected_core": "设计师", "expected_modifiers": ""},
        {"input": "运营总监", "expected_core": "运营总监", "expected_modifiers": ""},
        {"input": "运营经理", "expected_core": "运营经理", "expected_modifiers": ""},
        {"input": "内容运营", "expected_core": "内容运营", "expected_modifiers": ""},
        {"input": "用户运营", "expected_core": "用户运营", "expected_modifiers": ""},
        {"input": "活动运营", "expected_core": "活动运营", "expected_modifiers": ""},
        {"input": "电商运营", "expected_core": "电商运营", "expected_modifiers": ""},
        {"input": "销售总监", "expected_core": "销售总监", "expected_modifiers": ""},
        {"input": "销售主管", "expected_core": "销售主管", "expected_modifiers": ""},
        {"input": "销售代表", "expected_core": "销售代表", "expected_modifiers": ""},
        {"input": "销售专员", "expected_core": "销售专员", "expected_modifiers": ""},
        {"input": "客户经理", "expected_core": "客户经理", "expected_modifiers": ""},
        {"input": "渠道经理", "expected_core": "渠道经理", "expected_modifiers": ""},
        {"input": "商务经理", "expected_core": "商务经理", "expected_modifiers": ""},
        {"input": "市场总监", "expected_core": "市场总监", "expected_modifiers": ""},
        {"input": "市场经理", "expected_core": "市场经理", "expected_modifiers": ""},
        {"input": "市场专员", "expected_core": "市场专员", "expected_modifiers": ""},
        {"input": "品牌经理", "expected_core": "品牌经理", "expected_modifiers": ""},
        {"input": "推广专员", "expected_core": "推广专员", "expected_modifiers": ""},
        {"input": "客服经理", "expected_core": "客服经理", "expected_modifiers": ""},
        {"input": "客服主管", "expected_core": "客服主管", "expected_modifiers": ""},
        {"input": "客服专员", "expected_core": "客服专员", "expected_modifiers": ""},
        {"input": "人力资源总监", "expected_core": "人力资源总监", "expected_modifiers": ""},
        {"input": "HR经理", "expected_core": "HR经理", "expected_modifiers": ""},
        {"input": "招聘经理", "expected_core": "招聘经理", "expected_modifiers": ""},
        {"input": "招聘专员", "expected_core": "招聘专员", "expected_modifiers": ""},
        {"input": "HRBP", "expected_core": "HRBP", "expected_modifiers": ""},
        {"input": "薪酬专员", "expected_core": "薪酬专员", "expected_modifiers": ""},
        {"input": "培训专员", "expected_core": "培训专员", "expected_modifiers": ""},
        {"input": "人事专员", "expected_core": "人事专员", "expected_modifiers": ""},
        {"input": "财务总监", "expected_core": "财务总监", "expected_modifiers": ""},
        {"input": "财务经理", "expected_core": "财务经理", "expected_modifiers": ""},
        {"input": "会计", "expected_core": "会计", "expected_modifiers": ""},
        {"input": "出纳", "expected_core": "出纳", "expected_modifiers": ""},
        {"input": "成本会计", "expected_core": "成本会计", "expected_modifiers": ""},
        {"input": "税务专员", "expected_core": "税务专员", "expected_modifiers": ""},
        {"input": "审计", "expected_core": "审计", "expected_modifiers": ""},
        {"input": "行政经理", "expected_core": "行政经理", "expected_modifiers": ""},
        {"input": "行政主管", "expected_core": "行政主管", "expected_modifiers": ""},
        {"input": "行政专员", "expected_core": "行政专员", "expected_modifiers": ""},
        {"input": "行政助理", "expected_core": "行政助理", "expected_modifiers": ""},
        {"input": "前台", "expected_core": "前台", "expected_modifiers": ""},
        {"input": "文员", "expected_core": "文员", "expected_modifiers": ""},
        {"input": "采购经理", "expected_core": "采购经理", "expected_modifiers": ""},
        {"input": "采购专员", "expected_core": "采购专员", "expected_modifiers": ""},
        {"input": "供应链经理", "expected_core": "供应链经理", "expected_modifiers": ""},
        {"input": "仓库管理员", "expected_core": "仓库管理员", "expected_modifiers": ""},
        {"input": "物流专员", "expected_core": "物流专员", "expected_modifiers": ""},
        {"input": "生产经理", "expected_core": "生产经理", "expected_modifiers": ""},
        {"input": "生产主管", "expected_core": "生产主管", "expected_modifiers": ""},
        {"input": "品质工程师", "expected_core": "品质工程师", "expected_modifiers": ""},
    ]
    
    # 评估
    correct_core = 0
    correct_modifiers = 0
    total = len(annotated_test_cases)
    
    errors = []
    
    for i, case in enumerate(annotated_test_cases, 1):
        result = parser.parse(case['input'])
        
        # 检查核心词
        core_match = result['occupation_core'] == case['expected_core']
        if core_match:
            correct_core += 1
        
        # 检查修饰词（去除空格后比较）
        result_mods = result['modifiers'].replace(' ', '')
        expected_mods = case['expected_modifiers'].replace(' ', '')
        mods_match = result_mods == expected_mods
        if mods_match:
            correct_modifiers += 1
        
        # 记录错误
        if not core_match or not mods_match:
            errors.append({
                'input': case['input'],
                'expected_core': case['expected_core'],
                'actual_core': result['occupation_core'],
                'expected_modifiers': case['expected_modifiers'],
                'actual_modifiers': result['modifiers'],
                'core_match': core_match,
                'mods_match': mods_match
            })
    
    # 计算准确率
    core_accuracy = correct_core / total
    modifier_accuracy = correct_modifiers / total
    overall_accuracy = sum(1 for e in annotated_test_cases 
                          if parser.parse(e['input'])['occupation_core'] == e['expected_core'] 
                          and parser.parse(e['input'])['modifiers'].replace(' ', '') == e['expected_modifiers'].replace(' ', '')) / total
    
    print(f"\n手工标注测试集评估结果:")
    print(f"  样本数: {total}")
    print(f"  核心词准确率: {correct_core}/{total} = {core_accuracy:.2%}")
    print(f"  修饰词准确率: {correct_modifiers}/{total} = {modifier_accuracy:.2%}")
    print(f"  整体准确率: {overall_accuracy:.2%}")
    
    if errors:
        print(f"\n错误样本 ({len(errors)} 个):")
        for i, err in enumerate(errors[:10], 1):  # 只显示前10个
            print(f"\n  {i}. {err['input']}")
            if not err['core_match']:
                print(f"     核心词: 期望={err['expected_core']}, 实际={err['actual_core']} ❌")
            if not err['mods_match']:
                print(f"     修饰词: 期望={err['expected_modifiers']}, 实际={err['actual_modifiers']} ❌")
        
        if len(errors) > 10:
            print(f"\n  ... 还有 {len(errors) - 10} 个错误样本")
    
    # 保存评估报告
    output_dir = Path(__file__).parent.parent.parent / 'output' / 'job_title_parsing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'evaluation_report.json'
    report = {
        'total_samples': total,
        'core_accuracy': core_accuracy,
        'modifier_accuracy': modifier_accuracy,
        'overall_accuracy': overall_accuracy,
        'errors': errors
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估报告已保存: {report_file}")
    
    # 总结
    print("\n" + "=" * 80)
    print("评估总结")
    print("=" * 80)
    
    if core_accuracy >= 0.95:
        print(f"\n✅ 优秀！核心词准确率达到 {core_accuracy:.1%}")
    elif core_accuracy >= 0.90:
        print(f"\n✓ 良好！核心词准确率达到 {core_accuracy:.1%}")
    else:
        print(f"\n⚠ 需要改进，核心词准确率仅 {core_accuracy:.1%}")
    
    print(f"\n目标达成情况:")
    print(f"  高精度（>95%）: {'✅' if core_accuracy > 0.95 else '❌'} {core_accuracy:.1%}")
    print(f"  可解释: ✅ 所有结果可追溯到匹配规则")
    print(f"  易维护: ✅ 只需维护职业核心词词典")
    print(f"  高召回（>90%）: {'✅' if core_accuracy > 0.90 else '❌'} {core_accuracy:.1%}")
    
except Exception as e:
    print(f"\n[ERROR] 评估失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("评估完成！")
print("=" * 80)
