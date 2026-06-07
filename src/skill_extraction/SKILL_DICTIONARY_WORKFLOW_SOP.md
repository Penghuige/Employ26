# 技能词典工作流标准操作流程（SOP）

## 目标

本 SOP 用于规范 `src/skill_extraction` 的迭代流程，确保每一轮都遵循相同的控制流：

1. 运行基线词典匹配。
2. 运行回归评估。
3. 判断当前错误模式。
4. 仅应用保守性修改。
5. 重新运行匹配与评估。
6. 当指标达到目标或工作流被阻塞时停止。

固定框架本身不发生变化。每一轮中只有修复动作可以根据观察到的错误模式进行调整。

## 入口命令

### 1. 仅运行基线

```bash
python -m src.skill_extraction.skill_dictionary_workflow baseline
```

用于刷新当前基线，并写入工作流状态，但不会修改词典。

### 2. 运行完整标准化工作流

```bash
python -m src.skill_extraction.skill_dictionary_workflow run
```

该命令会执行已配置的迭代工作流，并写入持久化状态文件。

### 3. 查看工作流状态

```bash
python -m src.skill_extraction.skill_dictionary_workflow status
```

该命令会读取最新的工作流状态，并显示当前阶段、指标和下一步动作。

## 状态与产物

- 工作流状态：
  `output/skill_extraction/reports/dictionary_iteration/workflow_state.json`
- 每轮迭代报告：
  `output/skill_extraction/reports/dictionary_iteration/`
- 回归评估汇总：
  `output/skill_extraction/reports/regression_eval/`
- 词典规则：
  `config/skill_dictionary_iteration.json`

## 固定阶段

### 阶段 A：基线评估

始终运行：

- `match_flat_skills_to_duckdb.py`
- `regression_eval.py`

输出内容：

- 基线 precision / recall / F1
- 主要 false positives
- 主要 false negatives
- 错误模式分类

### 阶段 B：错误模式分类

工作流会将当前轮次归类为以下模式之一：

- `precision_first`
  误报占主导。优先采用过滤规则、别名清理和上下文约束。
- `recall_first`
  漏报占主导。优先采用保守性补充、规范名合并和受控别名。
- `balanced`
  两侧问题相近。优先采用最小化的混合修复。

### 阶段 C：保守性修复

允许的动作：

- 添加高置信度候选技能
- 合并规范同义词
- 添加短期 allowlist 条目
- 添加上下文匹配规则
- 屏蔽过于泛化的技能名称
- 屏蔽不安全别名

禁止的动作：

- 整体性重写整份词典
- 将完整词典发送给 API 模型
- 将低置信度候选直接加入主词典
- 在没有新规则的情况下重复此前已被否决的错误模式

### 阶段 D：重新评估

每一轮修复后，始终重新运行：

- 词典匹配
- 回归评估

随后比较：

- precision delta
- recall delta
- F1 delta

## 模型职责

### 本地 LLM

用于：

- 候选发现
- 低成本抽取探测

当结构化输出质量不稳定时，不应将本地 LLM 作为最终裁决者。

### `gpt-5.4-mini`

仅用于：

- 高不确定性候选裁决
- 边界样本复核
- 小样本质量审计

约束：

- 绝不审查完整词典
- 默认上限为 `max_api_reviews`
- 提示词中只能包含当前样本、候选列表和最小必要证据

## 停止条件

当以下任一条件成立时，工作流停止：

1. `precision >= precision_target`
2. `recall >= recall_target`
3. `f1 >= f1_target`
4. 没有任何经 API 审核的候选被保留，且工作流策略要求停止
5. 指标提升低于工作流阈值，且工作流策略要求停止
6. 达到最大轮次

目标值与停止阈值定义在：

- `config/skill_dictionary_iteration.json`

## 为什么修复动作可以变化

框架是固定的，但修复动作不能机械地重复同一套模板，因为后期轮次的错误类型与前期轮次不同。

典型序列如下：

- 前期轮次：泛化误报、同义词规范化、明显缺失的工具项
- 中期轮次：长尾缺失的技术术语
- 后期轮次：短缩写、依赖上下文的别名、两个汉字的中文术语

这些问题需要不同的修复方式。如果每一轮都重复同样的“抽取候选 + 审核 + 合并”动作，就会对 recall 过拟合，并重新引入 precision 失效问题。

## 推荐操作规则

建议采用以下决策策略：

- 如果问题是泛化误报，则添加过滤规则或上下文规则。
- 如果问题是规范同义词遗漏，则添加合并规则或别名。
- 如果问题是明确的长尾硬技能，则保守地加入。
- 如果问题是有歧义的短缩写，则保留在审核队列中，除非上下文能够约束。
- 如果某个问题经过一轮审核后仍然存在歧义，则停止并升级为人工确认。

## 当前控制器

标准化控制器实现于：

- [skill_dictionary_workflow.py](/D:/PythonProjects/Employ26/src/skill_extraction/skill_dictionary_workflow.py)

其提供的能力包括：

- 基线执行
- 持久化工作流状态
- 轮次历史
- 基于目标的停止条件
- 显式的下一步动作决策
