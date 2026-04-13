# `src/analysis` 目录说明

这个目录现在同时包含两套分析思路:

- 主链路: 基于 `output/integrated/*_整合_*.csv` 的标准化分析脚本，适合当前项目继续维护。
- 旧链路: 基于 `output/nlp_processed/*.csv` 的早期关键词分析脚本，适合快速探索或兼容旧数据。

## 推荐执行顺序

1. 先运行 [`integrate_occupation.py`](/d:/PythonProjects/Employ26/src/preprocessing/integrate_occupation.py)，生成 `output/integrated`。
2. 再运行 [`occupation_salary_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/occupation_salary_analysis.py)。
3. 再运行 [`education_distribution_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/education_distribution_analysis.py)。
4. 再运行 [`industry_trend_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/industry_trend_analysis.py)。
5. 如需规范交付表，再运行 [`generate_standardized_tables.py`](/d:/PythonProjects/Employ26/src/analysis/generate_standardized_tables.py)。
6. 如需最终汇总 Excel，再运行 [`generate_excel_summary.py`](/d:/PythonProjects/Employ26/src/analysis/generate_excel_summary.py)。

## 脚本状态

| 脚本 | 数据来源 | 主要产物 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| `occupation_salary_analysis.py` | `output/integrated` | 薪资报告、CSV、HTML | 主链路 | 当前最完整的薪资分析入口 |
| `education_distribution_analysis.py` | `output/integrated` | 学历分布 CSV、TXT | 主链路 | 与职业/职业类别标准化字段直接配套 |
| `industry_trend_analysis.py` | `output/integrated` | 行业趋势 CSV、TXT、HTML | 主链路 | 直接消费 `city_clean`、`industry_clean` |
| `generate_standardized_tables.py` | `output/reports` + `output/integrated` | 规范化 CSV | 二次汇总 | 负责统一交付列名 |
| `generate_excel_summary.py` | `output/reports` | 汇总 Excel | 二次汇总 | 汇总多个分析结果，不直接做原始分析 |
| `salary_analysis.py` | `output/nlp_processed` | `薪资分析报告.txt` | 旧链路 | 没有使用标准化职业字段 |
| `skill_combination.py` | `output/nlp_processed` | 技能组合 TXT、HTML | 旧链路 | 关键词共现分析，未接入新词典流程 |
| `time_trend_analysis.py` | `output/nlp_processed` | 时间趋势 TXT、HTML | 旧链路 | 适合宏观浏览，但口径偏旧 |

## 与 `src` 其他目录对比后的结论

### 1. 不重复、且仍然值得保留的部分

- `analysis` 主链路脚本和 [`src/preprocessing/integrate_occupation.py`](/d:/PythonProjects/Employ26/src/preprocessing/integrate_occupation.py) 是上下游关系，不是重复关系。
- `analysis` 和 [`src/skill_extraction/occupation_skill_pipeline.py`](/d:/PythonProjects/Employ26/src/skill_extraction/occupation_skill_pipeline.py) 也不重复。
  `skill_extraction` 做的是职业技能词典构建与覆盖率评估，`analysis` 做的是招聘数据统计报表。
- `generate_standardized_tables.py`、`generate_excel_summary.py` 虽然不“分析”原始数据，但负责交付层整理，仍然有价值。

### 2. 有明显重叠或偏旧的部分

- [`salary_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/salary_analysis.py)、[`skill_combination.py`](/d:/PythonProjects/Employ26/src/analysis/skill_combination.py)、[`time_trend_analysis.py`](/d:/PythonProjects/Employ26/src/analysis/time_trend_analysis.py) 仍然依赖 `output/nlp_processed`，没有接入更完整的 `output/integrated` 标准化字段。
- [`src/utils/analyze_results.py`](/d:/PythonProjects/Employ26/src/utils/analyze_results.py) 与 `skill_combination.py`、`time_trend_analysis.py` 都在做旧版关键词/NLP 结果统计，存在明显重叠。
- [`src/visualization/wordcloud_generator.py`](/d:/PythonProjects/Employ26/src/visualization/wordcloud_generator.py) 同样依赖 `output/nlp_processed`，在“旧版关键词可视化”这一层与旧链路分析脚本属于同一时期产物。

### 3. 当前最需要注意的技术债

- `parse_salary` 在 `salary_analysis.py`、`occupation_salary_analysis.py`、`generate_standardized_tables.py` 中重复实现，后续可考虑抽到公共工具模块。
- 旧链路脚本里的技能集合多为硬编码，无法自动复用 `skill_extraction` 目录下的新词典成果。
- 如果后续要继续维护 `salary_analysis.py`、`skill_combination.py`、`time_trend_analysis.py`，更稳妥的方向是迁移到 `output/integrated` 口径，而不是继续叠加旧逻辑。
