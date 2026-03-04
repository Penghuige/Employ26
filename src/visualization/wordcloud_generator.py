"""
词云生成模块
生成技能、岗位、行业的词云图
"""

import pandas as pd
from pathlib import Path
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_wordcloud_data():
    """生成词云数据"""
    base_dir = Path('d:/pythonProject/leisure/Employ26')
    nlp_dir = base_dir / 'output' / 'nlp_processed'
    output_dir = base_dir / 'output' / 'reports'
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("词云数据生成")
    logger.info("=" * 80)
    
    # 读取数据
    all_keywords = []
    all_jobs = []
    all_industries = []
    
    for csv_file in nlp_dir.glob('*.csv'):
        logger.info(f"读取: {csv_file.name}")
        df = pd.read_csv(csv_file, usecols=['岗位名称', '关键词', '公司行业'])
        
        # 收集关键词
        keywords_list = df['关键词'].dropna().str.split(',').tolist()
        for kw_list in keywords_list:
            all_keywords.extend(kw_list)
        
        # 收集岗位名称
        all_jobs.extend(df['岗位名称'].dropna().tolist())
        
        # 收集行业
        industries_list = df['公司行业'].dropna().str.split(',').tolist()
        for ind_list in industries_list:
            all_industries.extend(ind_list)
    
    logger.info(f"总关键词: {len(all_keywords):,}")
    logger.info(f"总岗位: {len(all_jobs):,}")
    logger.info(f"总行业: {len(all_industries):,}")
    
    # 统计频率
    keyword_counter = Counter(all_keywords)
    job_counter = Counter(all_jobs)
    industry_counter = Counter(all_industries)
    
    # 过滤掉HTML标签和无意义词
    stop_words = {'br', 'div', 'span', 'p', '00', '30', '18', '岗位职责', '任职要求'}
    keyword_counter_filtered = Counter({k: v for k, v in keyword_counter.items() if k not in stop_words and len(k) > 1})
    
    # 保存词云数据
    logger.info("\n保存词云数据...")
    
    # 1. 关键词词云数据
    keyword_file = output_dir / '词云数据_关键词.txt'
    with open(keyword_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:500]:
            f.write(f"{word} {count}\n")
    logger.info(f"关键词词云数据: {keyword_file}")
    
    # 2. 岗位词云数据
    job_file = output_dir / '词云数据_岗位.txt'
    with open(job_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(job_counter.items(), key=lambda x: x[1], reverse=True)[:300]:
            f.write(f"{word} {count}\n")
    logger.info(f"岗位词云数据: {job_file}")
    
    # 3. 行业词云数据
    industry_file = output_dir / '词云数据_行业.txt'
    with open(industry_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(industry_counter.items(), key=lambda x: x[1], reverse=True)[:200]:
            f.write(f"{word} {count}\n")
    logger.info(f"行业词云数据: {industry_file}")
    
    # 生成HTML词云（使用echarts）
    logger.info("\n生成HTML词云...")
    
    # 转换为JavaScript数组格式
    import json
    
    keyword_data_js = json.dumps([{"name": k, "value": v} for k, v in keyword_counter_filtered.most_common(200)])
    job_data_js = json.dumps([{"name": k, "value": v} for k, v in job_counter.most_common(150)])
    industry_data_js = json.dumps([{"name": k, "value": v} for k, v in industry_counter.most_common(100)])
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>广东省招聘数据 - 词云图</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
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
            height: 600px;
        }}
    </style>
</head>
<body>
    <h1>广东省招聘数据 - 词云分析</h1>
    
    <div class="chart-container">
        <h2>技能关键词词云</h2>
        <div id="keyword-cloud" class="chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>热门岗位词云</h2>
        <div id="job-cloud" class="chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>行业分布词云</h2>
        <div id="industry-cloud" class="chart"></div>
    </div>
    
    <script>
        // 关键词数据
        var keywordData = {keyword_data_js};
        var keywordChart = echarts.init(document.getElementById('keyword-cloud'));
        keywordChart.setOption({{
            tooltip: {{}},
            series: [{{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center',
                top: 'center',
                width: '90%',
                height: '90%',
                sizeRange: [12, 60],
                rotationRange: [-90, 90],
                rotationStep: 45,
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {{
                    fontFamily: 'Microsoft YaHei',
                    fontWeight: 'bold',
                    color: function () {{
                        return 'rgb(' + [
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160)
                        ].join(',') + ')';
                    }}
                }},
                emphasis: {{
                    textStyle: {{
                        shadowBlur: 10,
                        shadowColor: '#333'
                    }}
                }},
                data: keywordData
            }}]
        }});
        
        // 岗位数据
        var jobData = {job_data_js};
        var jobChart = echarts.init(document.getElementById('job-cloud'));
        jobChart.setOption({{
            tooltip: {{}},
            series: [{{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center',
                top: 'center',
                width: '90%',
                height: '90%',
                sizeRange: [12, 60],
                rotationRange: [-90, 90],
                rotationStep: 45,
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {{
                    fontFamily: 'Microsoft YaHei',
                    fontWeight: 'bold',
                    color: function () {{
                        return 'rgb(' + [
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160)
                        ].join(',') + ')';
                    }}
                }},
                emphasis: {{
                    textStyle: {{
                        shadowBlur: 10,
                        shadowColor: '#333'
                    }}
                }},
                data: jobData
            }}]
        }});
        
        // 行业数据
        var industryData = {industry_data_js};
        var industryChart = echarts.init(document.getElementById('industry-cloud'));
        industryChart.setOption({{
            tooltip: {{}},
            series: [{{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center',
                top: 'center',
                width: '90%',
                height: '90%',
                sizeRange: [12, 60],
                rotationRange: [-90, 90],
                rotationStep: 45,
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {{
                    fontFamily: 'Microsoft YaHei',
                    fontWeight: 'bold',
                    color: function () {{
                        return 'rgb(' + [
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160)
                        ].join(',') + ')';
                    }}
                }},
                emphasis: {{
                    textStyle: {{
                        shadowBlur: 10,
                        shadowColor: '#333'
                    }}
                }},
                data: industryData
            }}]
        }});
        
        // 响应式
        window.addEventListener('resize', function() {{
            keywordChart.resize();
            jobChart.resize();
            industryChart.resize();
        }});
    </script>
</body>
</html>
"""
    
    html_file = output_dir / '词云图.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML词云: {html_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 词云生成完成!")
    logger.info("=" * 80)
    logger.info(f"\n请在浏览器中打开: {html_file}")


if __name__ == '__main__':
    generate_wordcloud_data()

