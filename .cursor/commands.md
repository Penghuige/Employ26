# 招聘数据NLP分析 - 常用命令

## 数据预处理命令

### 数据清洗
```bash
# 清洗所有数据文件
python src/preprocessing/clean_data.py --input data/ --output output/processed/

# 清洗单个文件
python src/preprocessing/clean_data.py --input data/智联招聘_广东省_202203_202506.csv --output output/processed/zhilian_cleaned.csv
```

### 数据合并
```bash
# 合并三个数据源
python src/preprocessing/merge_data.py --input output/processed/ --output output/merged_data.csv
```

### 数据分割
```bash
# 将大文件分割成小文件（已有split.py）
python src/split.py --input data/智联招聘_广东省_202203_202506.csv --output output/split_out/ --rows 10000
```

## NLP分析命令

### 文本预处理
```bash
# 对岗位描述进行分词和清洗
python src/nlp_analysis/text_preprocessing.py --input output/processed/ --output output/nlp_processed/
```

### 关键词提取
```bash
# 提取技能关键词
python src/nlp_analysis/extract_keywords.py --input output/nlp_processed/ --output output/reports/keywords.csv --method tfidf

# 使用TextRank提取
python src/nlp_analysis/extract_keywords.py --input output/nlp_processed/ --output output/reports/keywords.csv --method textrank
```

### 主题建模
```bash
# LDA主题建模
python src/nlp_analysis/topic_modeling.py --input output/nlp_processed/ --num_topics 20 --output output/models/lda_model
```

### 词向量训练
```bash
# 训练Word2Vec模型
python src/nlp_analysis/train_word2vec.py --input output/nlp_processed/ --vector_size 100 --output output/models/word2vec.model
```

### 技能图谱构建
```bash
# 构建技能共现网络
python src/nlp_analysis/skill_network.py --input output/reports/keywords.csv --output output/reports/skill_network.html
```

## 数据分析命令

### 薪资分析
```bash
# 薪资统计分析
python src/analysis/salary_analysis.py --input output/processed/ --output output/reports/salary_report.html

# 薪资预测模型训练
python src/analysis/salary_prediction.py --input output/processed/ --output output/models/salary_model.pkl
```

### 趋势分析
```bash
# 时间序列分析
python src/analysis/trend_analysis.py --input output/processed/ --output output/reports/trend_report.html

# 行业趋势分析
python src/analysis/industry_analysis.py --input output/processed/ --output output/reports/industry_report.html
```

### 地域分析
```bash
# 城市就业分析
python src/analysis/city_analysis.py --input output/processed/ --output output/reports/city_report.html
```

## 可视化命令

### 生成词云
```bash
# 生成技能词云
python src/visualization/generate_wordcloud.py --input output/reports/keywords.csv --output output/reports/wordcloud.png --column skill

# 生成岗位词云
python src/visualization/generate_wordcloud.py --input output/processed/ --output output/reports/job_wordcloud.png --column 岗位名称
```

### 生成报告
```bash
# 生成完整分析报告
python src/visualization/generate_report.py --input output/reports/ --output output/final_report.html

# 生成PDF报告
python src/visualization/generate_report.py --input output/reports/ --output output/final_report.pdf --format pdf
```

## 模型评估命令

### 评估薪资预测模型
```bash
python src/evaluation/evaluate_salary_model.py --model output/models/salary_model.pkl --test_data output/processed/test_data.csv
```

### 评估主题模型
```bash
python src/evaluation/evaluate_topic_model.py --model output/models/lda_model --output output/reports/topic_evaluation.txt
```

## 工具命令

### 数据统计
```bash
# 查看数据基本信息
python src/utils/data_info.py --input data/

# 查看数据质量报告
python src/utils/data_quality.py --input output/processed/ --output output/reports/quality_report.html
```

### 性能测试
```bash
# 测试处理速度
python src/utils/benchmark.py --input data/智联招聘_广东省_202203_202506.csv --method chunk

# 内存使用分析
python src/utils/memory_profiler.py --script src/nlp_analysis/extract_keywords.py
```

## Jupyter Notebook命令

### 启动Notebook
```bash
# 启动Jupyter Lab
jupyter lab --notebook-dir=notebooks/

# 启动Jupyter Notebook
jupyter notebook notebooks/
```

### 运行特定Notebook
```bash
# 运行探索性分析
jupyter nbconvert --execute --to html notebooks/01_exploratory_analysis.ipynb

# 批量运行所有notebook
jupyter nbconvert --execute --to html notebooks/*.ipynb
```

## 环境管理命令

### 安装依赖
```bash
# 安装所有依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 更新依赖
```bash
# 更新requirements.txt
pip freeze > requirements.txt
```

## 测试命令

### 运行单元测试
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_preprocessing.py

# 运行测试并生成覆盖率报告
pytest --cov=src tests/
```

## 快捷组合命令

### 完整分析流程
```bash
# 一键运行完整分析流程
python run_full_pipeline.py --input data/ --output output/

# 仅运行NLP分析
python run_full_pipeline.py --input data/ --output output/ --steps nlp

# 仅运行可视化
python run_full_pipeline.py --input output/processed/ --output output/ --steps visualization
```

