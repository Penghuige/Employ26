# `src/penghui` 目录说明
# @ PengHui 2026-06-08
## 1. 目录定位

`src/penghui/` 目前是一组围绕“第二轮人工标注数据”和“RAG 检索模型微调”的实验脚本集合，主要用于：

- 复现第二轮数据有效性分析
- 分析人工标注与 DeepSeek 重标结果的分歧
- 基于不同样本筛选策略微调 BGE 检索模型
- 对多个微调版本做离线对比评估

这批脚本整体更接近“实验工作台”，而不是已经模块化、可长期维护的正式流水线。它们之间存在较多重复逻辑，适合先保留实验结论，再逐步收敛成公共模块。

## 2. 共用输入与产出

大多数脚本依赖以下数据：

- 标注数据：`data/project-4-at-2026-05-27-01-51-7cceb9ba.json`
- DeepSeek 重标结果：`output/deepseek_relabel/deepseek_relabel_raw.jsonl`
- 职业词典：`data/中国职业大典.xlsx`
- 基础向量模型：`config.paths.get_project_paths().bge_model_path`

常见输出位置：

- 文本报告：`output/*.txt`
- JSON 结果：`output/*.json`
- 微调模型与评估文件：`output/rag_round2_training/`

## 3. 脚本总览

| 脚本 | 主要作用 | 典型输出 | 当前定位 |
| --- | --- | --- | --- |
| `reproduce_round2_validity.py` | 复现第二轮数据有效性分析 | `output/round2_validity_report.txt` | 数据集整体体检 |
| `deep_analysis_round2.py` | 统计任务级/标注级 TopK 命中与多数意见情况 | 控制台输出 | 轻量分析脚本 |
| `disagreement_deep_analysis.py` | 深挖人类与 DeepSeek 分歧模式 | `output/disagreement_analysis.txt` | 分歧诊断 |
| `multidim_validation.py` | 用多信号给样本打质量分层 | `output/multidim_validation_report.txt`、`output/multidim_validation_results.json` | 标注质检 |
| `train_rag_round2.py` | v1：直接用第二轮标注训练基础检索模型 | 模型目录、`evaluation_results.json` | 基线微调方案 |
| `train_rag_round2_v3.py` | v3：用 Gold/Silver 样本训练 | 模型目录、`evaluation_v3.json` | 噪声过滤方案 |
| `train_rag_round2_v4.py` | v4：按分歧与语义排名筛正负样本 | 模型目录、`evaluation_v4.json` | 中等强度过滤方案 |
| `train_rag_weighted.py` | 置信分层加权训练 | 模型目录、`evaluation_weighted.json` | 质量加权方案 |
| `eval_models_multimetric.py` | 比较 baseline、v1、v3、v4 多项指标 | `output/model_comparison.txt` | 模型横向评估 |

## 4. 各脚本说明

### `reproduce_round2_validity.py`

作用：

- 复现第二轮数据集有效性报告
- 统计单标注/多标注分布、pairwise agreement、majority 存在率
- 对 DeepSeek 与人类多数意见的一致性做对照
- 单独分析 `is_validation_sample=1` 的验证样本

优点：

- 覆盖面最全，适合先了解第二轮数据质量
- 输出结构完整，适合当作目录中的总览入口

不足：

- 报告逻辑集中在一个超长 `main()` 中，可复用性弱
- 统计口径写死在脚本里，不支持参数化选择数据文件或输出位置
- 没有把关键统计过程拆成公共函数，后续别的脚本重复实现了相似逻辑

建议修复：

- 拆分为“数据加载 / 标注聚合 / 指标统计 / 报告写出”四层函数
- 增加命令行参数，允许指定输入文件和输出文件
- 把 majority、pairwise agreement、DeepSeek 对齐统计抽到公共工具模块

### `deep_analysis_round2.py`

作用：

- 基于任务多数意见统计 RAG 候选 Top1-Top5 命中率
- 区分单标注任务、多标注任务、逐标注样本三种口径
- 给出 `NONE` 选择比例和多标注平均跟随率

优点：

- 逻辑相对独立，适合快速回答“候选召回质量是否够用”
- 已使用 `pathlib`，结构比其他脚本更简洁

不足：

- 结果只打印到控制台，没有沉淀报告文件
- 数据文件名仍然硬编码到脚本常量
- 与其他脚本重复解析 Label Studio 标注格式

建议修复：

- 增加文本或 JSON 输出
- 改成从统一配置读取数据源
- 复用公共 `parse_choice` / `load_annotations` 工具函数

### `disagreement_deep_analysis.py`

作用：

- 识别“人类与 DeepSeek 不一致”的任务
- 从语义排名、层级距离、职业大类冲突、候选排名等角度分析错误模式
- 输出分歧样本特征统计和代表案例

优点：

- 对“分歧来自哪里”解释性较强
- 已经开始利用职业层级信息，而不是只看 Top1 准确率

不足：

- 采用逐条 `model.encode([anchor])` 的方式做语义分析，运行成本较高
- 使用追加写文件并在开始时删除旧文件，流程可读性一般
- 与 `multidim_validation.py`、`eval_models_multimetric.py` 共用的大量词典与分歧逻辑没有抽象

建议修复：

- 批量编码 anchor，减少重复推理
- 改成先收集 `output_lines`，最后一次性写文件
- 抽出职业层级距离、DeepSeek 对齐分析等公共函数

### `multidim_validation.py`

作用：

- 为每条样本计算多维质量信号
- 综合语义排名、大类关键词、标注员一致性、DeepSeek 一致性、标注员历史质量、`NONE` 比例等信号
- 输出质检报告和结构化 JSON 结果

优点：

- 是本目录里最接近“样本质量评分器”的脚本
- 输出了 JSON，便于后续继续筛样本或可视化

不足：

- 单脚本承担了特征工程、评分、案例展示、文件输出多个职责
- 大类关键词硬编码在脚本中，不方便调参与版本管理
- 每条任务单独编码 anchor，性能较差
- 缺少对评分规则的显式配置说明，复现实验时不够透明

建议修复：

- 把关键词字典迁移到 `dicts/` 或配置文件
- 把“信号计算”和“tier 判定”拆成独立模块
- 增加评分配置对象，避免阈值散落在代码中
- 批量向量化计算 semantic rank

### `train_rag_round2.py`

作用：

- 作为 v1 基线方案，直接从第二轮标注中抽取正样本对
- 默认把多标注任务整体放入测试集，单标注任务按比例切分
- 用 `MultipleNegativesRankingLoss` 微调基础 BGE 模型
- 输出微调模型与评估 JSON

优点：

- 流程清晰，是后续各版本训练脚本的起点
- 有相对完整的训练/测试切分与 baseline 对比

不足：

- 只使用正样本，没有显式 hard negative
- 训练、评估、数据抽取都写在同一个脚本里
- 测试集可能包含训练集中未覆盖的职业，结果会受分布影响
- 代码里仍有部分“BGE-M3”字样，但实际模型目录命名是 `bge-large`，命名不够统一

建议修复：

- 明确模型命名，避免 `BGE-M3` 与 `bge-large` 混用
- 把数据构造、训练、评估拆分
- 为测试切分策略补充固定配置和说明

### `train_rag_round2_v3.py`

作用：

- 只保留 DeepSeek 与人类一致的数据
- 多标注一致样本记为 Gold，单标注一致样本记为 Silver
- 用 Gold + Silver 训练，并把其余有效样本作为测试集

优点：

- 核心思路简单，噪声控制直观
- 比 v1 更强调标签可信度

不足：

- 测试集定义为“所有未入选 Gold/Silver 的样本”，天然更难，和 v1 结果并不完全同口径
- 仍与其他训练脚本重复了多数数据加载与解析逻辑
- 依赖 `output/rag_round2_training/bge-large-round2-finetuned` 作为对比模型时，没有显式检查该模型是否存在

建议修复：

- 明确记录训练集/测试集口径，避免横向比较误读
- 在运行前检查对比模型目录和输入文件是否存在
- 与 v1/v4/weighted 共用一套数据准备模块

### `train_rag_round2_v4.py`

作用：

- 用 “DeepSeek 一致 + 语义排名靠前” 构造正样本
- 用 “DeepSeek 分歧 + 语义排名靠后” 识别 hard negative 候选
- 正样本参与训练，负样本和中间样本参与评估

优点：

- 比 v3 多引入了语义排名这一层过滤
- 负样本集合有助于分析模型是否在可疑样本上过拟合

不足：

- 在预编码和训练前初始化模型时把设备直接写成 `cuda`，无 GPU 环境会直接失败
- `compute_semantic_rank()` 对每条样本单独编码，数据量大时很慢
- “负样本不参与训练，只用于评估”的策略写在代码里，但没有抽成可配置实验参数

建议修复：

- 改为自动选择 `cuda` / `cpu`
- 批量计算 anchor embedding 和 semantic rank
- 把正负样本阈值与测试集采样规模改成可配置参数

### `train_rag_weighted.py`

作用：

- 基于多维信号给样本打 `S/A/B/C/D` 质量等级
- 通过 oversampling 近似实现不同样本权重
- 输出加权训练模型及评估结果

优点：

- 是本目录里最完整的“样本打分 -> 加权训练”方案
- 将样本质量问题直接映射到训练权重，实验方向明确

不足：

- 多处把设备直接写成 `cuda`，CPU 环境不可运行
- 语义排名仍然按样本逐条编码，开销大
- oversample 倍数写死在代码中，不利于复现实验和做网格搜索
- 训练集/测试集划分混合了规则筛选和随机补样，统计口径需要更清楚地记录

建议修复：

- 先修正硬编码 GPU 问题
- 将 oversample 系数、测试集规模、tier 规则参数化
- 输出训练集和测试集的样本构成摘要，便于后续复盘

### `eval_models_multimetric.py`

作用：

- 对 baseline、v1、v3、v4 多个模型做统一评估
- 评估候选命中、候选排序、人类与 DeepSeek 分歧仲裁、层级准确率、MRR 等指标
- 输出横向比较报告

优点：

- 是目录中最适合做“最终模型对比”的脚本
- 指标维度比单纯看 Top1 更完整

不足：

- `MODEL_PATHS` 写死在脚本里，新增模型需要手改代码
- 指标定义没有单独文档化，不同训练版本之间的比较口径容易混淆
- 没有检查模型目录是否存在，缺模型时会在运行阶段才失败

建议修复：

- 改为从参数或配置文件读取待评估模型列表
- 为每个指标补一段简短说明并沉淀为固定报告模板
- 运行前做输入文件、模型目录、依赖完整性检查

## 5. 脚本之间的关系

可以把这批脚本理解为三层：

1. 数据质量分析层  
   `reproduce_round2_validity.py`、`deep_analysis_round2.py`

2. 分歧与样本筛选层  
   `disagreement_deep_analysis.py`、`multidim_validation.py`

3. 训练与评估层  
   `train_rag_round2.py`、`train_rag_round2_v3.py`、`train_rag_round2_v4.py`、`train_rag_weighted.py`、`eval_models_multimetric.py`

推荐阅读顺序：

1. 先看 `reproduce_round2_validity.py`
2. 再看 `multidim_validation.py` 和 `disagreement_deep_analysis.py`
3. 然后看 `train_rag_round2.py`、`train_rag_round2_v3.py`、`train_rag_round2_v4.py`、`train_rag_weighted.py`
4. 最后用 `eval_models_multimetric.py` 做横向对比

## 6. 当前共性问题

这批脚本的共性问题比较明显，优先级大致如下：

### P0：先修复，否则稳定性较差

- 多个脚本把输入文件名写死为单个导出文件，不支持切换批次数据
- `train_rag_round2_v4.py`、`train_rag_weighted.py` 直接硬编码 `device="cuda"`，无 GPU 环境无法运行
- 不同脚本都各自实现了一遍标注解析、majority 计算、词典加载，容易出现口径漂移

### P1：影响复现与维护

- 大量阈值、关键词、采样规模、oversample 倍数直接写在脚本中
- 模型路径、输出文件名、对比模型列表没有做统一配置
- 训练集/测试集构造规则在不同版本间差异较大，但缺少统一说明

### P2：影响性能与工程质量

- 多个脚本对每条任务逐条 `model.encode([anchor])`，运行效率偏低
- 缺少基础测试，至少应覆盖标注解析、majority、tier 判定等核心逻辑
- 报告写出方式不统一，有的只打印控制台，有的边跑边追加写文本

## 7. 推荐修复顺序

建议按下面顺序收敛：

1. 抽公共模块  
   新建如 `src/penghui/common.py`，统一实现：
   - 标注文件加载
   - `parse_choice()`
   - majority / pairwise agreement 统计
   - 职业词典加载
   - DeepSeek 结果加载

2. 给脚本加 CLI 参数  
   至少支持：
   - `--annotation-file`
   - `--deepseek-file`
   - `--dict-file`
   - `--output-dir` / `--output-file`
   - `--device`

3. 消除硬编码 GPU  
   统一改成自动选择 `cuda` 或 `cpu`。

4. 向量化语义排名计算  
   预先批量编码 anchor，避免每条任务单独推理。

5. 统一实验配置  
   将 tier 规则、语义排名阈值、测试集抽样规模、oversample 系数抽成配置对象。

6. 补最小测试  
   至少覆盖：
   - 候选选择解析
   - 多标注 majority 判定
   - tier 计算结果
   - 数据切分基本约束

## 8. 建议的后续归档方式

如果这批实验已经完成阶段性结论，后续建议做一次轻量整理：

- 保留 1 个总览分析入口
- 保留 1 个样本筛选入口
- 保留 1 个基础训练入口
- 保留 1 个多模型评估入口
- 其他历史版本转入 `archive/` 或标注为 `experimental/`

这样可以减少“v1 / v3 / v4 / weighted”继续并列扩散，避免后续维护成本持续上升。
