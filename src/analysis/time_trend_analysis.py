"""
旧版时间趋势分析模块。

用途:
- 基于 `output/nlp_processed/*.csv` 统计招聘量、热门岗位和关键词技能的时间变化趋势。
- 输出文本报告和 HTML 趋势图，适合做早期的宏观浏览。

前置依赖:
- 需要旧版 NLP 结果目录 `output/nlp_processed`。
- 数据里应包含 `发布时间`、`岗位名称`、`关键词` 等字段。

输出文件:
- `output/reports/时间趋势分析报告.txt`
- `output/reports/时间趋势图.html`

运行方式:
- `python -m src.analysis.time_trend_analysis`
- 或 `python src/analysis/time_trend_analysis.py`

维护说明:
- 本脚本没有使用 `preprocessing/integrate_occupation.py` 产出的标准化月份、职业和行业字段。
- 相比当前 `industry_trend_analysis.py` 与 `occupation_salary_analysis.py`，它的数据口径更旧，适合作为补充探索而非主报告。
"""

import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_time_trends():
    """分析时间趋势"""
    base_dir = Path(__file__).parent.parent.parent
    nlp_dir = base_dir / 'output' / 'nlp_processed'
    output_dir = base_dir / 'output' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("时间趋势分析")
    logger.info("=" * 80)
    
    # 读取数据
    all_data = []
    for csv_file in nlp_dir.glob('*.csv'):
        logger.info(f"读取: {csv_file.name}")
        df = pd.read_csv(csv_file, usecols=['发布时间', '岗位名称', '关键词'])
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"总数据: {len(df):,} 行")
    
    # 转换日期
    df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')
    df = df[df['发布时间'].notna()].copy()
    
    df['年份'] = df['发布时间'].dt.year
    df['月份'] = df['发布时间'].dt.to_period('M')
    df['季度'] = df['发布时间'].dt.to_period('Q')
    
    logger.info(f"有效日期数据: {len(df):,} 行")
    
    # 1. 招聘量趋势
    logger.info("\n" + "=" * 80)
    logger.info("招聘量趋势分析")
    logger.info("=" * 80)
    
    # 按年统计
    yearly_counts = df.groupby('年份').size()
    logger.info("\n按年统计:")
    for year, count in yearly_counts.items():
        logger.info(f"  {year}年: {count:8,} 个岗位")
    
    # 按季度统计
    quarterly_counts = df.groupby('季度').size()
    logger.info("\n按季度统计 (Top 20):")
    for quarter, count in quarterly_counts.head(20).items():
        logger.info(f"  {quarter}: {count:8,} 个岗位")
    
    # 按月统计
    monthly_counts = df.groupby('月份').size()
    logger.info("\n按月统计 (最近24个月):")
    for month, count in monthly_counts.tail(24).items():
        logger.info(f"  {month}: {count:8,} 个岗位")
    
    # 2. 技能需求趋势
    logger.info("\n" + "=" * 80)
    logger.info("技能需求趋势分析")
    logger.info("=" * 80)
    
    target_skills = [
        'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go',
        '机器学习', '深度学习', '数据分析', '大数据', '云计算', '人工智能',
        '前端', '后端', 'Vue', 'React', 'Spring',
        'Excel', 'PPT', 'Word', '产品经理', '项目管理'
    ]
    
    # 按年统计技能需求
    skill_by_year = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        year = row['年份']
        if pd.notna(row['关键词']):
            keywords = row['关键词'].split(',')
            for skill in target_skills:
                if skill in keywords:
                    skill_by_year[year][skill] += 1
    
    logger.info("\n技能需求年度变化:")
    for skill in ['Python', 'Java', 'JavaScript', '数据分析', '机器学习', '深度学习', '人工智能']:
        logger.info(f"\n{skill}:")
        for year in sorted(skill_by_year.keys()):
            count = skill_by_year[year][skill]
            total = yearly_counts[year]
            percentage = count / total * 100 if total > 0 else 0
            logger.info(f"  {year}年: {count:6,} 次 ({percentage:.2f}%)")
    
    # 3. 热门岗位趋势
    logger.info("\n" + "=" * 80)
    logger.info("热门岗位趋势")
    logger.info("=" * 80)
    
    top_jobs = df['岗位名称'].value_counts().head(20).index.tolist()
    
    job_by_year = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        year = row['年份']
        job = row['岗位名称']
        if job in top_jobs:
            job_by_year[year][job] += 1
    
    logger.info("\nTop 10 岗位年度变化:")
    for job in top_jobs[:10]:
        logger.info(f"\n{job}:")
        for year in sorted(job_by_year.keys()):
            count = job_by_year[year][job]
            logger.info(f"  {year}年: {count:6,} 次")
    
    # 生成趋势图HTML
    logger.info("\n生成趋势图...")
    
    # 准备月度数据
    monthly_data = []
    for month, count in monthly_counts.items():
        monthly_data.append([str(month), count])
    
    # 准备技能趋势数据
    skill_trend_data = {}
    for skill in ['Python', 'Java', 'JavaScript', '数据分析', '机器学习', '人工智能']:
        skill_trend_data[skill] = []
        for year in sorted(skill_by_year.keys()):
            count = skill_by_year[year][skill]
            skill_trend_data[skill].append([year, count])
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>时间趋势分析</title>
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
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart {{
            width: 100%;
            height: 500px;
        }}
    </style>
</head>
<body>
    <h1>广东省招聘数据 - 时间趋势分析</h1>
    
    <div class="chart-container">
        <h2>招聘量月度趋势</h2>
        <div id="monthly-chart" class="chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>技能需求年度趋势</h2>
        <div id="skill-chart" class="chart"></div>
    </div>
    
    <script>
        // 月度趋势图
        var monthlyChart = echarts.init(document.getElementById('monthly-chart'));
        var monthlyData = {monthly_data};
        
        monthlyChart.setOption({{
            title: {{
                text: '招聘量月度变化',
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis'
            }},
            xAxis: {{
                type: 'category',
                data: monthlyData.map(item => item[0]),
                axisLabel: {{
                    rotate: 45
                }}
            }},
            yAxis: {{
                type: 'value',
                name: '岗位数量'
            }},
            series: [{{
                data: monthlyData.map(item => item[1]),
                type: 'line',
                smooth: true,
                areaStyle: {{
                    color: 'rgba(66, 133, 244, 0.2)'
                }},
                lineStyle: {{
                    color: 'rgb(66, 133, 244)',
                    width: 2
                }},
                itemStyle: {{
                    color: 'rgb(66, 133, 244)'
                }}
            }}]
        }});
        
        // 技能趋势图
        var skillChart = echarts.init(document.getElementById('skill-chart'));
        var skillData = {skill_trend_data};
        
        var series = [];
        var colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272'];
        var i = 0;
        
        for (var skill in skillData) {{
            series.push({{
                name: skill,
                type: 'line',
                data: skillData[skill].map(item => item[1]),
                smooth: true,
                lineStyle: {{
                    width: 3
                }},
                itemStyle: {{
                    color: colors[i % colors.length]
                }}
            }});
            i++;
        }}
        
        var years = [];
        for (var skill in skillData) {{
            years = skillData[skill].map(item => item[0]);
            break;
        }}
        
        skillChart.setOption({{
            title: {{
                text: '主要技能需求年度变化',
                left: 'center'
            }},
            tooltip: {{
                trigger: 'axis'
            }},
            legend: {{
                data: Object.keys(skillData),
                top: 30
            }},
            xAxis: {{
                type: 'category',
                data: years
            }},
            yAxis: {{
                type: 'value',
                name: '需求次数'
            }},
            series: series
        }});
        
        window.addEventListener('resize', function() {{
            monthlyChart.resize();
            skillChart.resize();
        }});
    </script>
</body>
</html>
"""
    
    html_file = output_dir / '时间趋势图.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"趋势图已保存到: {html_file}")
    
    # 保存报告
    report_file = output_dir / '时间趋势分析报告.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("广东省招聘数据 - 时间趋势分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("招聘量年度统计:\n")
        f.write("-" * 80 + "\n")
        for year, count in yearly_counts.items():
            f.write(f"{year}年: {count:10,} 个岗位\n")
        
        f.write("\n\n技能需求年度变化:\n")
        f.write("-" * 80 + "\n")
        for skill in target_skills:
            f.write(f"\n{skill}:\n")
            for year in sorted(skill_by_year.keys()):
                count = skill_by_year[year][skill]
                total = yearly_counts[year]
                percentage = count / total * 100 if total > 0 else 0
                f.write(f"  {year}年: {count:8,} 次 ({percentage:.2f}%)\n")
    
    logger.info(f"报告已保存到: {report_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 时间趋势分析完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    analyze_time_trends()
