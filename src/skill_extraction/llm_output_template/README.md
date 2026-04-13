# LLM 输出目录规范模板

这是一份仓库内置模板，和 `init_llm_output_layout.py` 生成的目录结构保持一致。

## 推荐结构

```text
llm_outputs/
  README.md
  train/
    round_00/
      raw/
      json/
      imported/
  supplement/
    round_01/
      raw/
      json/
      imported/
  examples/
    train_result.example.json
    supplement_result.example.json
```

## 规范说明

- `raw/` 保存模型原始返回，允许有解释文字、Markdown、代码块。
- `json/` 保存整理后的标准 JSON，建议优先从这里导入。
- `imported/` 保存已导入词典的文件备份。
- `examples/` 作为标准格式参考。

## 命名建议

- 训练结果：
  `train__{prompt_file_key}__{model_name}.json`
- 补词结果：
  `supplement__round_01__{prompt_file_key}__{model_name}.json`

## 导入建议

```bash
python -m src.skill_extraction.import_llm_results --input output/skill_extraction/llm_outputs/train/round_00/json
python -m src.skill_extraction.import_llm_results --input output/skill_extraction/llm_outputs/supplement/round_01/json
```
