"""
技能组合分析模块
分析哪些技能经常一起出现，发现技能组合模式
"""

import pandas as pd
from pathlib import Path
import logging
from collections import Counter, defaultdict
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_skill_combinations():
    """分析技能组合"""
    base_dir = Path('d:/pythonProject/leisure/Employ26')
    nlp_dir = base_dir / 'output' / 'nlp_processed'
    output_dir = base_dir / 'output' / 'reports'
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("技能组合分析")
    logger.info("=" * 80)
    
    # 定义关注的技能
    target_skills = [
        'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go', 'SQL',
        '机器学习', '深度学习', '数据分析', '大数据', '云计算', '人工智能',
        '前端', '后端', '全栈', 'Web', 'APP',
        'MySQL', 'Redis', 'MongoDB', 'Oracle',
        'Vue', 'React', 'Angular', 'Spring', 'Django',
        'Linux', 'Docker', 'Kubernetes',
        'Excel', 'PPT', 'Word', 'Photoshop', 'AutoCAD',
        '产品经理', '项目管理', '市场营销', '数据运营'
    ]
    
    # 读取数据
    all_skill_sets = []
    
    for csv_file in nlp_dir.glob('*.csv'):
        logger.info(f"读取: {csv_file.name}")
        df = pd.read_csv(csv_file, usecols=['关键词'])
        
        for keywords_str in df['关键词'].dropna():
            keywords = keywords_str.split(',')
            # 只保留目标技能
            job_skills = [skill for skill in keywords if skill in target_skills]
            if len(job_skills) >= 2:  # 至少有2个技能
                all_skill_sets.append(set(job_skills))
    
    logger.info(f"找到 {len(all_skill_sets):,} 个包含多个技能的岗位")
    
    # 统计技能对的共现次数
    logger.info("\n分析技能对共现...")
    skill_pairs = Counter()
    
    for skill_set in all_skill_sets:
        # 生成所有技能对
        for pair in combinations(sorted(skill_set), 2):
            skill_pairs[pair] += 1
    
    # 统计技能三元组
    logger.info("分析技能三元组...")
    skill_triples = Counter()
    
    for skill_set in all_skill_sets:
        if len(skill_set) >= 3:
            for triple in combinations(sorted(skill_set), 3):
                skill_triples[triple] += 1
    
    # 输出结果
    logger.info("\n" + "=" * 80)
    logger.info("Top 50 技能组合（两个技能）")
    logger.info("=" * 80)
    
    for i, (pair, count) in enumerate(skill_pairs.most_common(50), 1):
        logger.info(f"{i:2d}. {pair[0]:15s} + {pair[1]:15s} - {count:6,} 次")
    
    logger.info("\n" + "=" * 80)
    logger.info("Top 30 技能组合（三个技能）")
    logger.info("=" * 80)
    
    for i, (triple, count) in enumerate(skill_triples.most_common(30), 1):
        logger.info(f"{i:2d}. {triple[0]:12s} + {triple[1]:12s} + {triple[2]:12s} - {count:5,} 次")
    
    # 分析每个技能最常搭配的其他技能
    logger.info("\n" + "=" * 80)
    logger.info("每个技能最常搭配的其他技能")
    logger.info("=" * 80)
    
    skill_companions = defaultdict(Counter)
    
    for skill_set in all_skill_sets:
        for skill in skill_set:
            for other_skill in skill_set:
                if skill != other_skill:
                    skill_companions[skill][other_skill] += 1
    
    # 输出主要技能的搭配
    main_skills = ['Python', 'Java', 'JavaScript', '数据分析', '前端', '后端', 
                   '产品经理', '项目管理', 'Excel', 'SQL']
    
    for skill in main_skills:
        if skill in skill_companions:
            logger.info(f"\n{skill} 最常搭配:")
            for other_skill, count in skill_companions[skill].most_common(10):
                logger.info(f"  - {other_skill:20s}: {count:5,} 次")
    
    # 保存报告
    report_file = output_dir / '技能组合分析报告.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("广东省招聘数据 - 技能组合分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"分析岗位数: {len(all_skill_sets):,}\n\n")
        
        f.write("Top 100 技能组合（两个技能）:\n")
        f.write("-" * 80 + "\n")
        for i, (pair, count) in enumerate(skill_pairs.most_common(100), 1):
            f.write(f"{i:3d}. {pair[0]:20s} + {pair[1]:20s} - {count:8,} 次\n")
        
        f.write("\n\nTop 50 技能组合（三个技能）:\n")
        f.write("-" * 80 + "\n")
        for i, (triple, count) in enumerate(skill_triples.most_common(50), 1):
            f.write(f"{i:3d}. {triple[0]:15s} + {triple[1]:15s} + {triple[2]:15s} - {count:6,} 次\n")
        
        f.write("\n\n每个技能最常搭配的其他技能:\n")
        f.write("-" * 80 + "\n")
        
        for skill in target_skills:
            if skill in skill_companions:
                f.write(f"\n{skill}:\n")
                for other_skill, count in skill_companions[skill].most_common(15):
                    f.write(f"  {other_skill:25s}: {count:8,} 次\n")
    
    logger.info(f"\n报告已保存到: {report_file}")
    
    # 生成技能关系网络图（HTML）
    logger.info("\n生成技能关系网络图...")
    
    # 选择Top技能对
    top_pairs = skill_pairs.most_common(100)
    
    # 构建节点和边
    nodes = set()
    edges = []
    
    for (skill1, skill2), count in top_pairs:
        nodes.add(skill1)
        nodes.add(skill2)
        edges.append({
            'source': skill1,
            'target': skill2,
            'value': count
        })
    
    nodes_list = [{'name': node, 'symbolSize': 20} for node in nodes]
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>技能关系网络图</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        #chart {{
            width: 100%;
            height: 800px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>广东省招聘数据 - 技能关系网络图</h1>
    <div id="chart"></div>
    
    <script>
        var chart = echarts.init(document.getElementById('chart'));
        
        var option = {{
            title: {{
                text: '技能共现关系网络',
                subtext: '线条粗细表示共现频率',
                top: 'top',
                left: 'center'
            }},
            tooltip: {{
                formatter: function(params) {{
                    if (params.dataType === 'edge') {{
                        return params.data.source + ' + ' + params.data.target + ': ' + params.data.value + ' 次';
                    }}
                    return params.data.name;
                }}
            }},
            series: [{{
                type: 'graph',
                layout: 'force',
                data: {nodes_list},
                edges: {edges},
                roam: true,
                label: {{
                    show: true,
                    position: 'right',
                    formatter: '{{b}}'
                }},
                labelLayout: {{
                    hideOverlap: true
                }},
                scaleLimit: {{
                    min: 0.4,
                    max: 2
                }},
                lineStyle: {{
                    color: 'source',
                    curveness: 0.3
                }},
                emphasis: {{
                    focus: 'adjacency',
                    lineStyle: {{
                        width: 10
                    }}
                }},
                force: {{
                    repulsion: 100,
                    edgeLength: [50, 150]
                }}
            }}]
        }};
        
        chart.setOption(option);
        
        window.addEventListener('resize', function() {{
            chart.resize();
        }});
    </script>
</body>
</html>
"""
    
    html_file = output_dir / '技能关系网络图.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"网络图已保存到: {html_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 技能组合分析完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    analyze_skill_combinations()

