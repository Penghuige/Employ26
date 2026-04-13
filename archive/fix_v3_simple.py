# -*- coding: utf-8 -*-
"""
修复V3分类器的特征提取问题
"""

import sys
from pathlib import Path

# 设置UTF-8输出
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取原文件
file_path = Path('d:/PythonProjects/Employ26/src/skill_extraction/v3_skill_classifier.py')
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换extract_features方法中的类型转换
replacements = [
    ('features.append(len(word))', 'features.append(float(len(word)))'),
    ('features.append(int(word.isupper()))', 'features.append(float(word.isupper()))'),
    ('features.append(int(word.islower()))', 'features.append(float(word.islower()))'),
    ('features.append(int(any(c.isdigit() for c in word)))', 'features.append(float(any(c.isdigit() for c in word)))'),
    ('features.append(int(word.isalpha()))', 'features.append(float(word.isalpha()))'),
    ('features.append(int(pos == tag))', 'features.append(float(pos == tag))'),
    ('features.extend(word_vec.tolist())', 'features.extend([float(x) for x in word_vec.tolist()])'),
    ('                    sim = self.word2vec_model.wv.similarity(word, seed)\n                    similarities.append(sim)',
     '                    try:\n                        sim = self.word2vec_model.wv.similarity(word, seed)\n                        similarities.append(float(sim))\n                    except:\n                        pass'),
    ('features.append(np.mean(similarities))', 'features.append(float(np.mean(similarities)))'),
    ('features.append(np.max(similarities))', 'features.append(float(np.max(similarities)))'),
    ('features.append(np.min(similarities))', 'features.append(float(np.min(similarities)))'),
    ('features.append(np.std(similarities))', 'features.append(float(np.std(similarities)))'),
    ('features.append(len([s for s in similarities if s > 0.7]))', 'features.append(float(len([s for s in similarities if s > 0.7])))'),
    ('features.append(doc_freq)', 'features.append(float(doc_freq))'),
    ('doc_freq_ratio = doc_freq / self.total_docs if self.total_docs > 0 else 0',
     'doc_freq_ratio = doc_freq / self.total_docs if self.total_docs > 0 else 0.0'),
    ('features.append(doc_freq_ratio)', 'features.append(float(doc_freq_ratio))'),
    ('features.append(total_freq)', 'features.append(float(total_freq))'),
    ('avg_freq = total_freq / doc_freq if doc_freq > 0 else 0',
     'avg_freq = total_freq / doc_freq if doc_freq > 0 else 0.0'),
    ('features.append(avg_freq)', 'features.append(float(avg_freq))'),
    ('tfidf = (total_freq / self.total_docs) * math.log(self.total_docs / (doc_freq + 1)) if self.total_docs > 0 else 0',
     'tfidf = (total_freq / self.total_docs) * math.log(self.total_docs / (doc_freq + 1)) if self.total_docs > 0 else 0.0'),
    ('features.append(tfidf)', 'features.append(float(tfidf))'),
    ('        return np.array(features)', '''        # 确保返回的是固定长度的numpy数组
        features_array = np.array(features, dtype=np.float64)
        
        # 验证维度
        if len(features_array) != 125:
            logger.warning(f"特征维度不正确: {len(features_array)}, 词: {word}")
            # 补齐或截断到125维
            if len(features_array) < 125:
                features_array = np.pad(features_array, (0, 125 - len(features_array)), 'constant')
            else:
                features_array = features_array[:125]
        
        return features_array'''),
]

count = 0
# 应用替换
for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        count += 1
        print(f"OK: {old[:40]}...")
    else:
        print(f"SKIP: {old[:40]}...")

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nDone! Fixed {count} places")
print("Run: python test_v3_classifier.py")
