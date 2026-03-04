# 广东省招聘数据NLP分析项目

## 项目简介

本项目对广东省三大招聘网站（智联招聘、猎聘网、前程无忧）2022-2025年的招聘数据进行深度NLP分析，总数据量约5GB。

## 数据来源

- 智联招聘_广东省_202203_202506
- 广东省招聘数据_猎聘网_202201_202506
- 广东省招聘数据_前程无忧_202201_202506

## 数据字段

- 发布时间
- 岗位名称
- 工作城市
- 薪资水平
- 学历要求
- 经验要求
- 岗位描述（核心NLP分析字段）
- 公司名称
- 公司规模
- 公司行业

## 分析目标

1. **技能需求分析**：识别市场最需要的技能
2. **薪资影响因素**：分析什么因素影响薪资
3. **岗位分类**：构建标准化岗位分类体系
4. **趋势预测**：预测未来招聘和技能需求趋势
5. **地域分析**：不同城市的就业机会和薪资差异
6. **行业洞察**：各行业的人才需求特点

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 清洗数据
python src/preprocessing/clean_data.py --input data/ --output output/processed/

# 合并数据
python src/preprocessing/merge_data.py --input output/processed/ --output output/merged_data.csv
```

### 3. NLP分析

```bash
# 文本预处理
python src/nlp_analysis/text_preprocessing.py --input output/processed/ --output output/nlp_processed/

# 提取关键词
python src/nlp_analysis/extract_keywords.py --input output/nlp_processed/ --output output/reports/keywords.csv
```

### 4. 数据分析

```bash
# 薪资分析
python src/analysis/salary_analysis.py --input output/processed/ --output output/reports/salary_report.html

# 趋势分析
python src/analysis/trend_analysis.py --input output/processed/ --output output/reports/trend_report.html
```

### 5. 生成报告

```bash
python src/visualization/generate_report.py --input output/reports/ --output output/final_report.html
```

## 项目结构

```
Employ26/
├── data/                    # 原始数据（5GB）
├── output/                  # 输出结果
│   ├── split_out/          # 分割后的数据
│   ├── processed/          # 预处理后的数据
│   ├── models/             # 训练的模型
│   └── reports/            # 分析报告
├── src/                     # 源代码
│   ├── preprocessing/      # 数据预处理
│   ├── nlp_analysis/       # NLP分析模块
│   ├── analysis/           # 数据分析
│   ├── modeling/           # 机器学习
│   ├── visualization/      # 可视化
│   └── utils/              # 工具函数
├── notebooks/              # Jupyter notebooks
├── tests/                  # 单元测试
├── docs/                   # 文档
├── requirements.txt        # 依赖
├── .cursorrules           # Cursor规则
└── README.md
```

## 核心功能

### NLP分析
- ✅ 中文分词（jieba）
- ✅ 关键词提取（TF-IDF、TextRank）
- ✅ 主题建模（LDA）
- ✅ 词向量训练（Word2Vec）
- ✅ 命名实体识别
- ✅ 文本聚类

### 数据分析
- ✅ 薪资分析与预测
- ✅ 技能需求统计
- ✅ 时间序列分析
- ✅ 地域分布分析
- ✅ 行业趋势分析

### 可视化
- ✅ 词云图
- ✅ 交互式图表（Plotly）
- ✅ 技能关系网络图
- ✅ 时间序列动态图
- ✅ HTML分析报告

## 技术栈

- **数据处理**：pandas, dask, vaex
- **NLP**：jieba, gensim, transformers, sklearn
- **可视化**：matplotlib, seaborn, plotly, pyecharts
- **机器学习**：scikit-learn, xgboost, lightgbm

## 注意事项

1. **大数据处理**：5GB数据使用分块读取，避免内存溢出
2. **中文处理**：使用jieba分词，需要自定义词典
3. **性能优化**：使用并行处理和缓存机制
4. **结果保存**：及时保存中间结果，支持断点续传

## 详细文档

- [完整分析方案](./NLP分析方案.md)
- [Cursor规则](./.cursorrules)
- [技能库](./.cursor/skills.md)
- [常用命令](./.cursor/commands.md)

## 作者

广东省招聘数据分析项目组

## 许可证

MIT License








