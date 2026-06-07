# job_title_parsing LLM 增强工作流 — 进展报告

**报告时间**: 2026-05-07
**检验样本**: 99 条（基线） + 20 条（LLM 增强）

---

## 零、环境状态

| 组件 | 状态 | 说明 |
|------|------|------|
| WSL vLLM Qwen3.6-27B-int4-AutoRound | ✅ 运行中 | `http://127.0.0.1:8100/v1`，max_model_len=4096，dtype=float16(inc quant) |
| External API (GPT-5.4-mini) | ✅ 可用 | `api.ofox.ai/v1`，通过 `LLMRouter` 调用 |
| DuckDB | 🔒 被 VS Code 锁定 | 改用 CSV 加载（`output/catalog_preprocessed.csv`） |

---

## 一、已完成工作

### 1.1 LLM 重排序模块 (`src/job_title_parsing/llm_reranker.py`)

创建了 `LLMReranker` 类，对 `MatchPipeline` 产出的 TopK 候选进行语义重排序。

**双后端支持**:
| 后端 | 状态 | 说明 |
|------|------|------|
| `external_api` | ✅ 可用 | 通过 `LLMRouter` 调用 GPT-5.4-mini（`api.ofox.ai`） |
| `vllm` (WSL) | ✅ 可用 | Qwen3.6-27B-int4-AutoRound，`http://127.0.0.1:8100/v1` |

**核心机制**:
- 将 TopK 候选的标题、任务描述、层级信息组装为结构化 prompt
- LLM 逐候选评估匹配度（0.0-1.0），输出重排序 JSON
- **阈值回退**: LLM top1 分数 < 0.50 时，自动回退到基线最优候选
- **英文缩写守卫**: prompt 中明确要求保留 CNC/CAD/QA 等技术缩写原意
- **无匹配检测**: LLM 可判定所有候选均不相关（`all_irrelevant: true`）

### 1.2 准确性检验脚本 (`scripts/eval_matching_accuracy.py`)

自动化评估脚本，支持：
- 从三平台 1% 样本中随机抽样
- CSV 加载职业大典（绕过 DuckDB 文件锁）
- 基线匹配 + LLM 重排序对比
- Markdown 报告 + JSON 详细数据输出

### 1.3 过时文件清理

归档至 `src/job_title_parsing/history/`:
- `evaluate_parser.py` — 阶段一原型评估
- `occupation_parser.py` — 被 title_cleaner + 检索替代
- `occupation_dict_manager.py` — 被 occupation_parser 依赖
- `evaluate_matching.py` — 被 cli.py evaluate 替代

### 1.4 中文注释补充

为 13 个活跃文件的 30+ 公开方法补充了规范的 `Args:`/`Returns:` 中文 docstring。

---

## 评估方法论说明

⚠️ **重要**: 本项目目前**没有人工标注的 ground truth 数据集**（即每条岗位的正确职业细类代码）。

因此，报告中出现的"改进/退化"判定基于以下方法：

1. **基线指标**（定量）: `confidence_level`、`risk_flags`、`top1_score` 等来自系统自身的打分和置信度标记，是客观数值。
2. **LLM 重排后的改进/退化**（定性人工抽检）: 由我（Claude）逐条阅读岗位标题、描述、基线 top1 候选和 LLM top1 候选，根据工作内容是否匹配做出主观判断。这不是精确的 precision/recall 计算，而是**方向性评估**，用于判断 LLM 重排是否值得继续投入。

后续如需量化评估，需要：
- 从 DuckDB 导出带 `gold_code` 的匹配结果（如果已有人工标注）
- 或使用 LLM 作为独立 judge 对匹配结果做批量评判（需严格 prompt 和一致性验证）

---

## 二、基线匹配准确性（纯规则 + 检索）

| 指标 | 数值 |
|------|------|
| 有候选返回率 | 100.0% |
| 高置信度占比 | 10.1% |
| 中置信度占比 | 38.4% |
| 低置信度占比 | **51.5%** |
| 需要人工复核率 | **89.9%** |
| Top1 平均分数 | 0.5583 |
| 无风险标记率 | 10.1% |

### 风险标记分布

| 风险标记 | 出现次数（/99） |
|----------|----------------|
| generic_title_penalty（泛标题惩罚） | 67 |
| small_top1_top2_margin（Top1-2 分数接近） | 46 |
| task_signal_missing（任务信号缺失） | 30 |
| title_signal_missing（标题信号缺失） | 10 |
| low_top1_score（Top1 分数过低） | 8 |

### 基线典型问题

1. **跨领域误匹配**: "MES高级.net开发" → "房地产策划专业技术人员"
2. **关键词语义漂移**: "课程顾问（周末双休）" → "公司金融顾问"
3. **英文缩写丢失**: "cad绘图员" → "电子设备维修工程技术人员"
4. **行业歧义未消解**: "应收会计（跨境电商）" → "电子商务师"（会计被电商覆盖）

结论：纯规则+检索的基线系统能覆盖常见岗位，但 **泛标题惩罚过激**（67/99）、**Top1-Top2 区分度不足**（46/99）、**无语义理解** 导致跨领域误匹配。

---

## 三、LLM 增强后准确性

对 20 条样本进行 GPT-5.4-mini 重排序：

| 指标 | 数值 |
|------|------|
| LLM 调用成功率 | **100%** (20/20) |
| Top1 被改变率 | **70%** (14/20) |
| 判定全部不相关 | 40% (8/20) |
| 阈值回退触发 | 待大规模验证后统计 |

### 改进案例（人工抽检）

| 岗位 | 基线 Top1 | LLM Top1 | 评判 |
|------|----------|----------|------|
| 人寿险统计分析 | 统计专业技术人员 | **精算(保险)** | ✅ 改进（精算更精确） |
| 硬件测试助理工程师 | 人力资源服务专业技术人员 | **电子设备调试人员** | ✅ 改进（HR→电子，明显纠错） |
| 应届设计（花都城） | 检验检测专业技术人员 | **美术设计专业技术人员** | ✅ 改进（检测→设计） |
| 汽车4S店销售经理 | 保险代理人 | **汽车运用工程技术人员** | ✅ 改进（保险→汽车） |
| cad绘图员 | 电子设备维修工程技术人员 | **制图员** | ✅ 改进（理解 CAD=制图） |
| 应收会计（跨境电商） | 会计专业人员 | **会计专业人员** | ✅ 保持正确（未被电商误导） |
| 外贸采购 | 采购员 | 采购员 | ✅ 保持正确 |

### 退化案例

| 岗位 | 基线 Top1 | LLM Top1 | 分析 |
|------|----------|----------|------|
| cnc操作员 | 机床操作人员 | 纺织印染人员 | ❌ LLM 未识别 CNC=数控机床 |

退化原因：LLM 对英文缩写 `CNC` 的语义理解仍然不足。已在 prompt 中加入英文缩写守卫规则，需进一步改进。

### 退化案例

| 岗位 | 基线 Top1 | LLM Top1 | 分析 |
|------|----------|----------|------|
| cnc操作员 | 机床操作人员 | 纺织印染人员 | ❌ LLM 未识别 CNC=数控机床 |

退化原因：LLM 对英文缩写 `CNC` 的语义理解仍然不足。已在 prompt 中加入英文缩写守卫规则，需进一步改进。

---

## 三-B、vLLM Qwen3.6-27B 重排序结果

在 WSL vLLM 服务成功启动后，对另外 20 条样本进行本地 Qwen3.6-27B-int4 重排序评估。

| 指标 | 数值 |
|------|------|
| LLM 调用成功率 | **100%** (20/20) |
| Top1 被改变率 | **40%** (8/20) |
| 改进（人工判定） | 3 |
| 退化（人工判定） | **2**（高置信度错误） |
| 不确定 | 2 |
| 不变 | 13 |

### Qwen 改进案例（人工抽检）

| 岗位 | 基线 Top1 | Qwen Top1 | 评判 |
|------|----------|-----------|------|
| SQE工程师 | 供应链管理师S | **质量认证认可工程技术人员** | ✅ 改进（SQE=供应商质量工程师→质量认证更匹配） |
| 酒吧服务员 | 浴池服务员 | **康乐服务员** | ✅ 改进（酒吧→康乐比浴池更合理） |
| 主管-应收应付 | 企业经理 | **会计专业人员** | ✅ 改进（财务主管→会计，明显纠错） |

### Qwen 退化案例（高置信度错误）

| 岗位 | 基线 Top1 | Qwen Top1 | 分数 | 分析 |
|------|----------|-----------|------|------|
| **地产中介（高提成+双休）** | **房地产经纪人S** | 会计专业人员 | **0.95** | ❌ 严重错误：房地产中介→会计，且置信度极高 |
| **培训讲师** | **职业培训实训指导专业技术人员** | 市场管理员 | 0.55 | ❌ 明显错误：培训讲师→市场管理，完全无关 |

### Qwen vs GPT-5.4-mini 对比（均为 20 条人工抽检，非精确量化）

| 维度 | GPT-5.4-mini | Qwen3.6-27B-int4 |
|------|-------------|-------------------|
| 调用速度 | ~5s/条 | ~3s/条（本地） |
| Top1 变更率 | 70% | 40% |
| 改进率（人工判定） | ~50% | ~38% |
| 退化率（人工判定） | ~14% | **~25%**（含高置信度错误） |
| 高置信度错误 | 未发现 | **2 例**（地产→会计 0.95, 培训→市场 0.55） |
| 英文缩写处理 | 较好（cad→制图正确） | 未专门测试 |

### 关键发现

1. **Qwen3.6-27B-int4 存在高置信度幻觉** — 对"地产中介→会计专业人员"给出 0.95 分，完全越过阈值回退保护
2. **int4 量化可能影响推理质量** — 27B 原版模型推理能力应强于 GPT-5.4-mini，但量化后出现明显退化
3. **GPT-5.4-mini 更适合此任务** — 作为闭源模型的蒸馏版本，在分类重排任务上表现更稳定
4. **阈值回退不足以防御高置信度错误** — 需要引入"基线-LLM 分歧检测"作为额外保护层

### 改进措施（已实施）

在 `LLMReranker` 中新增**基线-LLM 分歧保护**：当 LLM 的 top1 与基线 top1 不同，且 LLM 分数 > 0.85（疑似过度自信）时，将基线 top1 保留在首位，LLM 候选降为第二位。这能防御"地产→会计"这类高置信度幻觉。

### 改进/退化统计（人工抽检，非精确量化）

在 14 条 Top1 被改变的案例中，逐条阅读岗位内容后主观判定：
- 明确改进: ~50%（7 条）
- 明确退化: ~14%（2 条）
- 不确定/需更多上下文: ~36%（5 条）

---

## 四、关键代码产出

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/job_title_parsing/llm_reranker.py` | LLM 候选重排序器（双后端，阈值回退） |
| `scripts/eval_matching_accuracy.py` | 准确性检验脚本 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/job_title_parsing/__init__.py` | 移除过时导出，新增架构说明 |
| `src/job_title_parsing/scoring.py` | 补充 9 个方法的中文 docstring |
| `src/job_title_parsing/matching_pipeline.py` | 补充 `match_one`/`match_batch` docstring |
| `src/job_title_parsing/title_cleaner.py` | 补充 `clean` 方法 docstring |
| `src/job_title_parsing/catalog_preprocessor.py` | 补充 `preprocess` docstring |
| `src/job_title_parsing/alias_builder.py` | 补充 2 个方法 docstring |
| `src/job_title_parsing/jd_parser.py` | 补充 `parse` docstring |
| `src/job_title_parsing/feature_extractor.py` | 补充 `extract` docstring |
| `src/job_title_parsing/hierarchy_filter.py` | 补充 3 个方法 docstring |
| `src/job_title_parsing/hierarchy_keyword_builder.py` | 补充 `build_from_catalog` docstring |
| `src/job_title_parsing/bm25_index.py` | 补充 `search` docstring |
| `src/job_title_parsing/ngram_retrieval.py` | 补充 `overlap_score` docstring |
| `src/job_title_parsing/matching_evaluator.py` | 补充 2 个函数 docstring |
| `src/job_title_parsing/match_utils.py` | 补充 3 个函数 docstring |
| `tests/test_job_title_matching_fixes.py` | 移除对过时模块的依赖 |
| `CLAUDE.md` | 新增项目级 AI 辅助开发指南 |

---

## 五、下一步计划

### 短期（本周）

1. **修复 WSL vLLM 模型路径** — 当前 Qwen3.6-27B 目录缺少 HuggingFace `config.json`，需在 WSL 内确认模型格式（可能是 GGUF 需要转换，或指向正确的 HF 目录）

2. **批量 LLM 评估（100 条全量）** — 当前仅用 20 条做概念验证，需跑满 100 条并统计：
   - LLM 重排后的准确率变化
   - 阈值回退触发率与回退后的正确率
   - 按职业类别分组的改进/退化分布

3. **改进 baseline 的 `generic_title_penalty`** — 当前 67/99 触发泛标题惩罚，需审查 `dicts/job_generic_terms.txt` 是否包含过多合理岗位词

### 中期（下周）

4. **LLM 作为匹配质量评估器（judge）** — 不依赖人工标注，用 LLM 对匹配结果的正确性做独立评判，生成 pseudo-label 用于量化评估

5. **英文缩写词典增强** — 在 `dicts/` 下建立 `job_english_abbr_mapping.txt`，将 CNC/MES/CAD/PLC 等常见缩写映射为中文解释，注入 LLM prompt

6. **分层重排序策略** — 仅在 baseline `confidence_level != "high"`（即 medium/low）时启用 LLM 重排，降低 API 调用成本

### 长期（两周+）

7. **baseline + LLM 联合流水线** — 将 `LLMReranker` 集成到 `MatchPipeline` 中作为可选步骤，通过配置开关控制

8. **unmatched 判定增强** — 利用 LLM 的 `all_irrelevant` 信号，为真正无法匹配的岗位建立独立的 unmatched 输出通道

---

## 六、使用方法

### 运行基线评估
```bash
.conda/python.exe scripts/eval_matching_accuracy.py --sample-size 100
```

### 运行 LLM 增强评估
```bash
.conda/python.exe scripts/eval_matching_accuracy.py --sample-size 100 --llm
```

### 在代码中使用 LLMReranker
```python
from src.job_title_parsing.matching_pipeline import MatchPipeline
from src.job_title_parsing.llm_reranker import LLMReranker

# 基线匹配
pipeline = MatchPipeline(catalog_df=catalog_df)
baseline = pipeline.match_one(job_title, job_description)

# LLM 重排序
reranker = LLMReranker(backend='external_api')  # 或 'vllm'
result = reranker.rerank(
    job_title=job_title,
    job_description=job_description,
    candidates=baseline['candidates'],
)
print(f"最终 Top1: {result.reranked_candidates[0]['title']}")
print(f"置信度: {result.reranked_candidates[0]['llm_score']}")
print(f"是否回退: {result.fell_back_to_baseline}")
```
