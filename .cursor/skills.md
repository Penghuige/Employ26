# 招聘数据NLP分析 - 技能库

## 数据处理技能

### 大文件分块读取
```python
import pandas as pd

def read_large_csv_chunks(filepath, chunksize=10000):
    """分块读取大型CSV文件"""
    for chunk in pd.read_csv(filepath, chunksize=chunksize, encoding='utf-8'):
        yield chunk
```

### 并行处理
```python
from multiprocessing import Pool
import pandas as pd

def parallel_process(data, func, n_cores=4):
    """并行处理数据"""
    with Pool(n_cores) as pool:
        result = pool.map(func, data)
    return result
```

### 内存优化
```python
def optimize_dataframe_memory(df):
    """优化DataFrame内存使用"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
    return df
```

## NLP技能

### 中文文本预处理
```python
import jieba
import re

def preprocess_chinese_text(text):
    """中文文本预处理"""
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 分词
    words = jieba.cut(text)
    # 去除停用词
    stopwords = load_stopwords()
    words = [w for w in words if w not in stopwords and len(w) > 1]
    return ' '.join(words)
```

### 技能关键词提取
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_skills_tfidf(job_descriptions, top_n=20):
    """使用TF-IDF提取技能关键词"""
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    feature_names = vectorizer.get_feature_names_out()
    
    # 获取每个文档的top关键词
    skills = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        skills.append([feature_names[i] for i in top_indices])
    return skills
```

### 主题建模
```python
from gensim import corpora, models

def topic_modeling(texts, num_topics=10):
    """LDA主题建模"""
    # 创建词典
    dictionary = corpora.Dictionary(texts)
    # 创建语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    # LDA模型
    lda_model = models.LdaModel(corpus, num_topics=num_topics, 
                                id2word=dictionary, passes=10)
    return lda_model, dictionary, corpus
```

### 词向量训练
```python
from gensim.models import Word2Vec

def train_word2vec(sentences, vector_size=100):
    """训练Word2Vec词向量"""
    model = Word2Vec(sentences, vector_size=vector_size, 
                     window=5, min_count=5, workers=4)
    return model
```

## 薪资分析技能

### 薪资范围解析
```python
import re

def parse_salary(salary_str):
    """解析薪资字符串"""
    if pd.isna(salary_str):
        return None, None
    
    # 匹配各种薪资格式
    patterns = [
        r'(\d+\.?\d*)-(\d+\.?\d*)万',
        r'(\d+)-(\d+)元',
        r'(\d+)k-(\d+)k',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, salary_str, re.IGNORECASE)
        if match:
            min_sal = float(match.group(1))
            max_sal = float(match.group(2))
            # 统一转换为月薪（元）
            if '万' in salary_str:
                min_sal *= 10000
                max_sal *= 10000
            elif 'k' in salary_str.lower():
                min_sal *= 1000
                max_sal *= 1000
            return min_sal, max_sal
    return None, None
```

### 薪资预测模型
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_salary_predictor(X, y):
    """训练薪资预测模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    return model, score
```

## 可视化技能

### 词云生成
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, output_path):
    """生成词云图"""
    wc = WordCloud(
        font_path='simhei.ttf',  # 中文字体
        width=1600, height=800,
        background_color='white',
        max_words=200
    ).generate(text)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 交互式图表
```python
import plotly.express as px

def create_interactive_chart(df, x, y, title):
    """创建交互式图表"""
    fig = px.scatter(df, x=x, y=y, title=title, 
                     hover_data=df.columns)
    fig.update_layout(template='plotly_white')
    return fig
```

## 性能监控技能

### 内存监控
```python
import psutil
import os

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"内存使用: {mem_info.rss / 1024 / 1024:.2f} MB")
```

### 进度条
```python
from tqdm import tqdm

def process_with_progress(items):
    """带进度条的处理"""
    results = []
    for item in tqdm(items, desc="处理中"):
        result = process_item(item)
        results.append(result)
    return results
```

