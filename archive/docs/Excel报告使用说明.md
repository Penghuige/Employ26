# 📊 Excel整合报告使用说明

## 🎯 功能概述

新增了**Excel整合报告生成器**，将所有分析结果整合到一个Excel文件中，方便查看和进一步分析。

---

## 📁 生成的Excel报告结构

Excel文件包含**9个Sheet**，按主题分类：

### 1. 概览 (Summary)
- 数据总量统计
- 薪资基础指标（平均值、中位数、分位数）
- 数据时间跨度
- 报告生成时间

### 2. 技能薪资分析
- 50个核心技能的薪资数据
- 包含：平均薪资、中位数、最低、最高、岗位数量、占比
- 按平均薪资降序排列

### 3. 学历薪资分析
- 7个学历层次的薪资对比
- 博士、硕士、本科、大专、高中、中专、初中

### 4. 经验薪资分析
- 8个经验层次的薪资对比
- 10年以上、8-10年、5-10年、5-7年、3-5年、1-3年、1年以下、应届

### 5. 城市薪资分析
- 广东省21个城市的薪资排行
- 深圳、广州、珠海等主要城市对比

### 6. 技能组合Top100
- 最常见的技能对组合
- 如：Java + Spring、Excel + Word等
- 包含共现次数和占比

### 7. 技能三元组Top50
- 三个技能的组合模式
- 如：Python + SQL + 数据分析
- 发现技能栈模式

### 8. 年度招聘趋势
- 2022-2025年招聘量变化
- 同比增长率

### 9. 技能需求趋势
- 50个技能在各年度的需求变化
- 包含绝对数量和占比
- 识别上升/下降趋势

---

## 🚀 使用方法

### 方法1：运行完整分析（推荐）

```bash
# 运行所有分析并生成Excel报告
python run_all_analysis.py
```

这会依次执行：
1. 薪资分析
2. 技能组合分析
3. 时间趋势分析
4. 词云生成
5. **Excel整合报告生成**

### 方法2：单独生成Excel报告

```bash
# 只生成Excel报告（需要先运行过NLP处理）
python src/analysis/generate_excel_report.py
```

---

## 📂 输出位置

```
output/reports/
├── 招聘数据分析报告_20260303_143025.xlsx  ← Excel整合报告
├── 薪资分析报告.txt
├── 技能组合分析报告.txt
├── 时间趋势分析报告.txt
├── 词云图.html
├── 技能关系网络图.html
└── 时间趋势图.html
```

文件名包含时间戳，避免覆盖历史报告。

---

## 💡 Excel报告的优势

### vs 文本报告
- ✅ 数据结构化，易于筛选和排序
- ✅ 支持Excel公式和数据透视表
- ✅ 可以直接复制到PPT或Word
- ✅ 支持导出为CSV、PDF等格式

### vs HTML可视化
- ✅ 可以进行二次计算和分析
- ✅ 适合生成自定义图表
- ✅ 方便与其他数据合并
- ✅ 支持批注和协作

---

## 🔧 扩展性设计

### 添加新的分析Sheet

当你添加新的NLP分析功能时，只需在 `generate_excel_report.py` 中添加新的生成方法：

```python
class ExcelReportGenerator:
    
    def generate_new_analysis_sheet(self):
        """生成新的分析Sheet"""
        logger.info("生成新分析...")
        
        # 你的分析逻辑
        data = []
        for item in self.df:
            # 处理数据
            data.append({
                '字段1': value1,
                '字段2': value2,
                ...
            })
        
        df_result = pd.DataFrame(data)
        return df_result
    
    def generate_report(self):
        """生成完整报告"""
        # ... 现有代码 ...
        
        sheets = {
            '概览': self.generate_summary_sheet(),
            # ... 现有Sheet ...
            '新分析': self.generate_new_analysis_sheet(),  # 添加这里
        }
        
        # ... 保存代码 ...
```

### 示例：添加行业分析Sheet

```python
def generate_industry_salary_sheet(self):
    """生成行业薪资分析Sheet"""
    logger.info("生成行业薪资分析...")
    
    industry_stats = self.df_valid.groupby('公司行业')['平均薪资'].agg([
        ('平均薪资', 'mean'),
        ('中位数薪资', 'median'),
        ('岗位数量', 'count')
    ]).reset_index()
    
    industry_stats = industry_stats.sort_values('平均薪资', ascending=False)
    industry_stats.index = industry_stats.index + 1
    
    return industry_stats
```

然后在 `generate_report()` 中添加：

```python
sheets = {
    # ... 现有Sheet ...
    '行业薪资分析': self.generate_industry_salary_sheet(),
}
```

---

## 📊 数据分析技巧

### 在Excel中进行二次分析

#### 1. 筛选高薪技能
```
在"技能薪资分析"Sheet中：
1. 选中数据区域
2. 数据 → 筛选
3. 在"平均薪资"列筛选 > 20000
```

#### 2. 创建数据透视表
```
1. 选中"技能组合Top100"数据
2. 插入 → 数据透视表
3. 行：技能1
4. 值：共现次数（求和）
```

#### 3. 生成图表
```
1. 选中"年度招聘趋势"数据
2. 插入 → 图表 → 折线图
3. 查看招聘量变化趋势
```

#### 4. 计算薪资增长率
```
在"技能薪资分析"Sheet中添加列：
=[@平均薪资] / $B$2 - 1
（计算相对于平均薪资的溢价率）
```

---

## 🎨 自定义Excel样式

如果需要美化Excel报告，可以使用 `openpyxl` 的样式功能：

```python
from openpyxl.styles import Font, PatternFill, Alignment

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name, df in sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=True)
        
        # 获取工作表
        worksheet = writer.sheets[sheet_name]
        
        # 设置标题行样式
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        
        # 自动调整列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
```

---

## ❓ 常见问题

### Q1: Excel文件打不开？
**A:** 确保安装了 `openpyxl` 库：
```bash
pip install openpyxl
```

### Q2: 生成的Excel文件在哪里？
**A:** 在 `output/reports/` 目录下，文件名包含时间戳。

### Q3: 如何只更新某个Sheet？
**A:** 目前需要重新生成整个报告。如果只想更新部分数据，可以：
1. 打开现有Excel文件
2. 运行单独的分析脚本
3. 手动复制粘贴数据

### Q4: 数据量太大，Excel打开很慢？
**A:** 可以修改代码，只导出Top N的数据：
```python
df_skill = df_skill.head(100)  # 只保留前100行
```

### Q5: 如何导出为CSV？
**A:** 在Excel中：文件 → 另存为 → CSV格式

---

## 🔄 更新日志

### v1.0 (2026-03-03)
- ✅ 初始版本
- ✅ 支持9个分析Sheet
- ✅ 自动生成时间戳文件名
- ✅ 集成到主分析流程

### 未来计划
- 🔜 添加Excel样式美化
- 🔜 支持自定义Sheet选择
- 🔜 添加图表自动生成
- 🔜 支持增量更新

---

## 📞 技术支持

如有问题，请查看：
1. `技能识别方法说明.md` - 了解技能识别原理
2. `NLP分析方案.md` - 完整的分析方案
3. 项目代码注释

---

**祝使用愉快！** 🎉








