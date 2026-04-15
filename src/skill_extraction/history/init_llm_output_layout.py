"""
初始化 LLM 输出目录规范模板。

默认会在 `output/skill_extraction/llm_outputs` 下生成建议目录结构、
说明文档和示例 JSON，方便后续直接执行导入脚本。
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TEMPLATE_README = """# LLM 输出目录规范

建议把所有给技能词典流程使用的 LLM 返回结果统一放在这个目录下。

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

## 各目录用途

- `train/round_00/raw/`
  保存训练阶段的原始 LLM 输出，允许带解释文字、Markdown、代码块。
- `train/round_00/json/`
  保存人工整理后的标准 JSON。
- `train/round_00/imported/`
  保存已经导入过词典的结果备份，避免重复导入。
- `supplement/round_XX/raw/`
  保存第 XX 轮补词阶段的原始 LLM 输出。
- `supplement/round_XX/json/`
  保存第 XX 轮补词阶段整理后的标准 JSON。
- `supplement/round_XX/imported/`
  保存已经导入过词典的补词结果备份。
- `examples/`
  放标准输出示例，便于提示词和人工检查统一格式。

## 文件命名建议

- 训练阶段：
  `train__{prompt_file_key}__model_name.json`
- 补词阶段：
  `supplement__round_01__{prompt_file_key}__model_name.json`

示例：

- `train__c1cff79d64_商业_服务业人员___购销人员___推销_展销人员___推销员__qwen3.json`
- `supplement__round_01__c1cff79d64_商业_服务业人员___购销人员___推销_展销人员___推销员__qwen3.json`

## 导入建议

训练阶段导入：

```bash
python -m src.skill_extraction.history.import_llm_results --input output/skill_extraction/llm_outputs/train/round_00/json
```

补词阶段导入：

```bash
python -m src.skill_extraction.history.import_llm_results --input output/skill_extraction/llm_outputs/supplement/round_01/json
```

如果你希望先预览导入结果：

```bash
python -m src.skill_extraction.history.import_llm_results --input output/skill_extraction/llm_outputs/train/round_00/json --dry-run
```

## 约束

- 一个文件尽量只放一个职业细类的一个 payload。
- `detail_path` 必须和 prompt 中一致，不要手改。
- 训练阶段优先输出 `skills`。
- 补词阶段优先输出 `missing_skills`。
- 导入完成后，建议把已导入文件移到同轮的 `imported/` 目录。
"""


TRAIN_EXAMPLE = """{
  "detail_path": "商业、服务业人员 > 购销人员 > 推销、展销人员 > 推销员",
  "detail_name": "推销员",
  "skills": [
    {
      "name": "客户开发",
      "aliases": ["客户拓展", "客户开拓"],
      "skill_type": "销售技能",
      "notes": "偏核心"
    },
    {
      "name": "Excel",
      "aliases": ["excel"],
      "skill_type": "办公软件",
      "notes": ""
    }
  ]
}
"""


SUPPLEMENT_EXAMPLE = """{
  "detail_path": "商业、服务业人员 > 购销人员 > 推销、展销人员 > 推销员",
  "missing_skills": [
    {
      "name": "CRM",
      "aliases": ["crm系统", "客户关系管理系统"],
      "skill_type": "业务工具",
      "notes": "来源于 round_01 补词"
    }
  ]
}
"""


def _write_text_if_missing(path: Path, content: str, force: bool) -> None:
    """在目标文件不存在时写入内容。"""
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def initialize_layout(target_dir: Path, force: bool = False) -> None:
    """创建目录模板。"""
    directories = [
        target_dir,
        target_dir / "train" / "round_00" / "raw",
        target_dir / "train" / "round_00" / "json",
        target_dir / "train" / "round_00" / "imported",
        target_dir / "supplement" / "round_01" / "raw",
        target_dir / "supplement" / "round_01" / "json",
        target_dir / "supplement" / "round_01" / "imported",
        target_dir / "examples",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    _write_text_if_missing(target_dir / "README.md", DEFAULT_TEMPLATE_README, force=force)
    _write_text_if_missing(target_dir / "examples" / "train_result.example.json", TRAIN_EXAMPLE, force=force)
    _write_text_if_missing(
        target_dir / "examples" / "supplement_result.example.json",
        SUPPLEMENT_EXAMPLE,
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="初始化 LLM 输出目录规范模板")
    parser.add_argument(
        "--target",
        default=r"output\skill_extraction\llm_outputs",
        help="模板输出目录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果模板文件已存在则覆盖",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args()
    target_dir = Path(args.target)
    initialize_layout(target_dir=target_dir, force=args.force)
    print(f"initialized: {target_dir}")


if __name__ == "__main__":
    main()
