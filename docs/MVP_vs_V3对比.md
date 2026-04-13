# MVP vs V3 技能抽取方案对比

## 📊 核心对比

| 维度 | MVP方案 | V3方案 |
|------|---------|--------|
| **核心技术** | 精确匹配 + 黑名单 | 机器学习分类器 |
| **能否发现新词** | ❌ 否 | ✅ 是 |
| **精度** | 90% | 85-88% |
| **召回率** | 65% | 75-80% |
| **训练时间** | 无需训练 | 5-10分钟 |
| **推理速度** | 极快 | 中等 |
| **维护成本** | 中（需维护词典） | 低（自动学习） |
| **可解释性** | 高（规则明确） | 中（特征重要性） |
| **适用场景** | 已知技能抽取 | 新技能发现 |

---

## 🎯 方法论对比

### MVP方案：规则 + 词典

```python
# 判断逻辑
if word in skill_seeds:
    return True, 1.0, category
elif word in blacklist:
    return False, 0.0, "黑名单"
else:
    return False, 0.0, "不在词典中"
```

**优点**：
- ✅ 高精度（90%+）
- ✅ 速度快
- ✅ 可解释性强
- ✅ 无需训练

**缺点**：
- ❌ 无法发现新词
- ❌ 需要维护词典
- ❌ 召回率受限于词典大小

---

### V3方案：机器学习

```python
# 判断逻辑
features = extract_features(word)  # 125维特征
probability = classifier.predict_proba(features)
if probability[1] >= threshold:
    return True, probability[1], infer_category(word)
else:
    return False, probability[1], "置信度过低"
```

**优点**：
- ✅ 可以发现新词
- ✅ 自动学习模式
- ✅ 更高召回率
- ✅ 可持续更新

**缺点**：
- ❌ 精度略低（85-88%）
- ❌ 需要训练时间
- ❌ 推理速度较慢
- ❌ 可解释性较弱

---

## 🔍 技术细节对比

### 特征使用

| 特征类型 | MVP | V3 |
|---------|-----|-----|
| 精确匹配 | ✅ 主要方法 | ✅ 作为特征之一 |
| 黑名单 | ✅ 硬过滤 | ✅ 训练负样本 |
| 词向量 | ❌ 不使用 | ✅ 100维特征 |
| 统计特征 | ❌ 不使用 | ✅ 文档频率、TF-IDF |
| 词性特征 | ❌ 不使用 | ✅ 10维one-hot |
| 相似度特征 | ❌ 不使用 | ✅ 与种子词相似度 |

### 判断机制

**MVP**：
```
word → 精确匹配词典 → 是/否
       ↓
    黑名单过滤 → 是/否
       ↓
    规则检查 → 是/否
```

**V3**：
```
word → 特征提取（125维）
       ↓
    随机森林分类器
       ↓
    置信度评分（0-1）
       ↓
    阈值判断 → 是/否
```

---

## 📈 性能对比

### 精度-召回率曲线

```
精度
 │
 │  MVP ●
 │      │
90%│      │
 │      │
 │      │    V3 ●
85%│      │    │
 │      │    │
 │      │    │
 │      └────┘
 │
 └──────────────────── 召回率
    65%      75%
```

### 适用场景

**MVP适合**：
- ✅ 技能词典相对稳定
- ✅ 需要极高精度
- ✅ 实时性要求高
- ✅ 资源受限环境

**V3适合**：
- ✅ 技能快速变化
- ✅ 需要发现新技能
- ✅ 可接受训练时间
- ✅ 追求更高召回率

---

## 💡 组合使用策略

### 策略1：两阶段流水线

```python
# 阶段1：MVP抽取已知技能（高精度）
mvp_skills = mvp_extractor.extract_from_dataset(df)

# 阶段2：V3发现新技能（高召回）
remaining_texts = filter_out_mvp_skills(texts, mvp_skills)
v3_skills = v3_classifier.discover_new_skills(remaining_texts)

# 合并结果
all_skills = {**mvp_skills, **v3_skills}
```

**优点**：
- 已知技能用MVP（90%精度）
- 新技能用V3发现
- 综合精度和召回率都高

### 策略2：置信度加权

```python
# MVP结果：置信度1.0
mvp_results = {word: {'confidence': 1.0, 'source': 'mvp'} 
               for word in mvp_skills}

# V3结果：置信度0.85-0.95
v3_results = {word: {'confidence': conf, 'source': 'v3'} 
              for word, conf in v3_skills.items()}

# 合并，MVP优先
all_skills = {**v3_results, **mvp_results}
```

### 策略3：主动学习循环

```
1. MVP抽取已知技能
   ↓
2. V3发现候选新技能
   ↓
3. 人工审核高置信度候选
   ↓
4. 确认的技能加入种子词典
   ↓
5. 重新训练V3模型
   ↓
6. 回到步骤1（定期循环）
```

---

## 🎓 使用建议

### 场景1：初次使用

**推荐**：先用MVP

```bash
# 1. 测试MVP
python test_mvp_extractor.py

# 2. 评估效果
# 如果召回率足够 → 继续使用MVP
# 如果召回率不足 → 升级到V3
```

### 场景2：技能变化快

**推荐**：使用V3

```bash
# 定期运行V3发现新技能
python test_v3_classifier.py

# 审核并更新词典
# 重新训练模型
```

### 场景3：生产环境

**推荐**：组合使用

```python
# 在线服务：使用MVP（速度快）
def extract_skills_online(text):
    return mvp_extractor.extract_from_text(text)

# 离线分析：使用V3（发现新技能）
def discover_new_skills_offline(texts):
    return v3_classifier.discover_new_skills(texts)

# 定期更新：主动学习循环
def update_skill_dict():
    new_skills = discover_new_skills_offline(recent_texts)
    reviewed_skills = human_review(new_skills)
    update_seed_dict(reviewed_skills)
    retrain_v3_model()
```

---

## 📊 实际案例对比

### 案例1：抽取"Python"

**MVP**：
```python
word = "Python"
# 精确匹配种子词典
if word in skill_seeds:
    return True, 1.0, 'programming_language'

结果：✅ 是技能，置信度1.0，类别programming_language
速度：0.001ms
```

**V3**：
```python
word = "Python"
# 提取125维特征
features = extract_features(word)
# 随机森林预测
probability = classifier.predict_proba(features)

结果：✅ 是技能，置信度0.98，类别programming_language
速度：5ms
```

### 案例2：判断"Rust"（新技能）

**MVP**：
```python
word = "Rust"
# 不在种子词典中
if word not in skill_seeds:
    return False, 0.0, "不在词典中"

结果：❌ 不是技能（漏报）
速度：0.001ms
```

**V3**：
```python
word = "Rust"
# 提取特征：
# - 与Python相似度：0.82
# - 词性：英文名词
# - 文档频率：150
# 随机森林预测
probability = 0.92

结果：✅ 是技能，置信度0.92，类别programming_language
速度：5ms
```

### 案例3：判断"沟通能力"（非技能）

**MVP**：
```python
word = "沟通能力"
# 在黑名单中
if word in blacklist['soft_skills']:
    return False, 0.0, "黑名单(soft_skills)"

结果：✅ 不是技能（正确）
速度：0.001ms
```

**V3**：
```python
word = "沟通能力"
# 提取特征：
# - 与种子词相似度：0.35（低）
# - 词性：名词
# - 在训练负样本中
# 随机森林预测
probability = 0.15

结果：✅ 不是技能，置信度0.15（正确）
速度：5ms
```

---

## 🔧 参数调优对比

### MVP调优

```python
# 主要通过维护词典
1. 添加新技能到 skill_seeds.txt
2. 添加噪声词到 blacklist_*.txt
3. 添加同义词到 synonyms.txt

# 无需重新训练，立即生效
```

### V3调优

```python
# 主要通过调整参数
1. 置信度阈值：threshold (0.80-0.95)
2. 最小文档频率：min_freq (5-20)
3. 随机森林参数：n_estimators, max_depth

# 需要重新训练模型
```

---

## 💰 成本对比

### 开发成本

| 阶段 | MVP | V3 |
|------|-----|-----|
| 初始开发 | 2-3天 | 5-7天 |
| 词典构建 | 1-2天 | 1天（复用MVP） |
| 测试验证 | 1天 | 2天 |
| **总计** | **4-6天** | **8-10天** |

### 运行成本

| 资源 | MVP | V3 |
|------|-----|-----|
| CPU | 极低 | 中等 |
| 内存 | 极低（<100MB） | 中等（~500MB） |
| 磁盘 | 极低（<10MB） | 中等（~200MB） |
| 训练时间 | 无 | 5-10分钟 |
| 推理速度 | 0.001ms/词 | 5ms/词 |

### 维护成本

| 任务 | MVP | V3 |
|------|-----|-----|
| 添加新技能 | 编辑文本文件 | 重新训练模型 |
| 更新黑名单 | 编辑文本文件 | 重新训练模型 |
| 定期审核 | 每月 | 每季度 |
| 人工工作量 | 中 | 低 |

---

## 🎯 决策树

```
需要发现新技能？
├─ 否 → 使用MVP
│        - 高精度
│        - 速度快
│        - 维护简单
│
└─ 是 → 技能变化频率？
         ├─ 低（每季度） → 使用MVP + 定期人工更新
         │                  - 成本最低
         │                  - 精度最高
         │
         └─ 高（每月/每周） → 使用V3
                              - 自动发现
                              - 持续学习
                              - 或组合使用MVP+V3
```

---

## 📝 总结

### MVP方案

**核心**：规则 + 词典  
**优势**：高精度、速度快、可解释  
**劣势**：无法发现新词  
**适合**：已知技能抽取、实时场景

### V3方案

**核心**：机器学习分类器  
**优势**：可发现新词、自动学习  
**劣势**：精度略低、需要训练  
**适合**：新技能发现、离线分析

### 最佳实践

**推荐组合使用**：
1. 在线服务用MVP（高精度、速度快）
2. 离线分析用V3（发现新技能）
3. 定期主动学习（更新词典、重训模型）

这样可以兼顾**精度、召回率和维护成本**！

---

**最后更新**: 2026-03-07  
**版本**: v1.0
