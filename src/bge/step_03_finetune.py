import pandas as pd
import re
import os
import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models

from src.model_platform.torch_runtime import empty_cuda_cache_safe, resolve_torch_device

# 过滤 Numpy 版本警告，保持终端清洁
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 配置区域 =================
# 1. 输入文件路径
TIER1_FILE = r"src\bge\data5\Tier1_Matched_Data.csv"
DICT_FILE = r"data\中国职业大典.xlsx"
# D5 自动质检输出（用于困难样本增量训练）
HARD_LABEL_FILE = r"src\bge\output\qwen3_8b_rag_labels_latest.csv"
HARD_SAMPLE_MAX = 50000
HARD_SAMPLE_REPEAT = 2

# 2. 模型路径配置（从集中路径配置获取，支持环境变量覆盖）
from config.paths import get_project_paths
_paths = get_project_paths()
# 原始本地模型地址
LOCAL_MODEL_PATH = str(_paths.bge_model_path)
# 微调后新模型的保存地址
OUTPUT_MODEL_PATH = str(_paths.project_root / "models" / "bge-base-zh-finetuned")

# 3. 训练参数（针对 RTX 4090 24G 显存优化）
BATCH_SIZE = 32
EPOCHS = 5
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5

# 修正后的正则：解决了 bad character range 报错，增强了薪资捕获能力
NOISE_REGEX = r'(月薪|底薪|综合薪资|提成|绩效|奖金|包吃包住|包食宿|五险一金|带薪年假|周末双休|长白班|提供早午餐|月入过万|上不封顶|年底双薪|\d{3,5}-\d{3,5}元|\d+[kKwW](-\d+[kKwW])?|[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+|【.*?】|\[.*?\])'


def clean_text_for_finetune(text):
    """
    深度清洗函数：剥离所有非职业特征相关的语义噪音
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. 剔除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 2. 靶向去噪
    try:
        text = re.sub(NOISE_REGEX, ' ', text, flags=re.IGNORECASE)
    except Exception:
        text = text.replace('【', ' ').replace('】', ' ')

    # 3. 仅保留中文字符和基础标点，过滤特殊符号干扰
    text = re.sub(r'[^\w\u4e00-\u9fa5，。、；：！]', ' ', text)

    # 4. 规范化空格并截断
    return re.sub(r'\s+', ' ', text).strip()[:MAX_SEQ_LENGTH]


def build_training_data():
    print(">>> 步骤 1: 加载并清洗 Tier1 匹配数据...")
    if not os.path.exists(TIER1_FILE):
        print(f"❌ 找不到 Tier1 数据文件: {TIER1_FILE}")
        return []

    df_t1 = pd.read_csv(TIER1_FILE, encoding='utf-8-sig')
    # 统一字段名，避免隐藏空格/大小写差异导致取列失败
    df_t1.columns = [str(col).strip() for col in df_t1.columns]

    # 兼容不同来源的列名写法
    desc_col_t1 = next((c for c in ['岗位描述', 'job_description', '职位描述'] if c in df_t1.columns), None)
    code_col_t1 = next((c for c in ['tier1_matched_code', 'tier1_match_code', 'matched_code', '职业代码'] if c in df_t1.columns), None)

    if desc_col_t1 is None or code_col_t1 is None:
        print(f"❌ Tier1 数据缺少必要列。当前列名: {list(df_t1.columns)}")
        return []

    # 适配 D2 新流程：若存在质检列，仅使用 qc_is_correct=1 的高置信样本参与微调
    if 'qc_is_correct' in df_t1.columns:
        before_len = len(df_t1)
        df_t1 = df_t1[df_t1['qc_is_correct'] == 1].copy()
        print(f"   已按 qc_is_correct=1 过滤: {before_len} -> {len(df_t1)} 条")
        if df_t1.empty:
            print("❌ 过滤后无可训练样本，请检查 D2 质检输出。")
            return []

    print(">>> 步骤 2: 读取官方职业大典标准文件...")
    try:
        df_dict = pd.read_excel(DICT_FILE, engine='openpyxl')
    except Exception as e:
        print(f"❌ 读取大典文件失败: {e}")
        return []

    df_dict.fillna('', inplace=True)

    # 自动识别列名（针对 Excel 不同表头的容错逻辑）
    title_col = next((col for col in ['title', '职业名称'] if col in df_dict.columns), 'title')
    code_col = next((col for col in ['code', '职业代码'] if col in df_dict.columns), 'code')
    desc_col = next((col for col in ['desc', '职业定义'] if col in df_dict.columns), 'desc')
    task_col = next((col for col in ['tasks', '主要工作任务'] if col in df_dict.columns), 'tasks')

    dict_map = {
        str(row[code_col]): f"{row[title_col]}。定义：{row[desc_col]} 任务：{row[task_col]}"
        for _, row in df_dict.iterrows()
    }

    # 读取 D5 困难样本：tier2 且判错样本，使用 gold_code 作为增量监督信号
    hard_examples = []
    if os.path.exists(HARD_LABEL_FILE):
        try:
            hard_df = pd.read_csv(HARD_LABEL_FILE, encoding='utf-8-sig')
            hard_df.columns = [str(c).strip() for c in hard_df.columns]

            required_cols = {'stage', 'is_correct', 'gold_code', '岗位描述'}
            if required_cols.issubset(set(hard_df.columns)):
                hard_df = hard_df[
                    (hard_df['stage'] == 'tier2')
                    & (hard_df['is_correct'] == 0)
                    & (hard_df['gold_code'].notna())
                ].copy()
                hard_df['gold_code'] = hard_df['gold_code'].astype(str).str.strip()
                hard_df = hard_df[hard_df['gold_code'].isin(dict_map.keys())]

                if len(hard_df) > HARD_SAMPLE_MAX:
                    hard_df = hard_df.sample(n=HARD_SAMPLE_MAX, random_state=42)

                hard_df['cleaned_desc'] = hard_df['岗位描述'].apply(clean_text_for_finetune)
                hard_df = hard_df[hard_df['cleaned_desc'].str.len() > 10]

                for _, row in hard_df.iterrows():
                    code = row['gold_code']
                    hard_examples.append(InputExample(texts=[row['cleaned_desc'], dict_map[code]]))

                # 困难样本加权重复，提升边界判别能力
                hard_examples = hard_examples * max(1, HARD_SAMPLE_REPEAT)
                print(f"   已加载 D5 困难样本: {len(hard_examples)} 条（含重复加权）")
            else:
                print("[WARN] D5 标签文件缺少困难样本所需列，跳过增量训练样本。")
        except Exception as e:
            print(f"[WARN] 读取 D5 困难样本失败，已跳过: {e}")
    else:
        print(f"[INFO] 未发现 D5 标签文件，跳过困难样本增量: {HARD_LABEL_FILE}")

    # 1. 清洗基础样本描述
    tqdm.pandas(desc="清洗进度")
    df_t1['cleaned_desc'] = df_t1[desc_col_t1].progress_apply(clean_text_for_finetune)
    df_t1 = df_t1[df_t1['cleaned_desc'].str.len() > 10].copy()

    # 2. 类别均衡化（Down-sampling）
    print(">>> 步骤 3: 执行类别均衡化降采样...")
    # 采用“先打乱再每类取前100条”的方式降采样，避免 groupby.apply 在不同 pandas 版本的列行为差异
    df_sampled = (
        df_t1.sample(frac=1.0, random_state=42)
        .groupby(code_col_t1, group_keys=False)
        .head(100)
        .copy()
    )

    # 3. 构造基础正样本对
    train_examples = []
    for _, row in df_sampled.iterrows():
        code = str(row.get(code_col_t1, ''))
        if code in dict_map:
            train_examples.append(InputExample(texts=[row['cleaned_desc'], dict_map[code]]))

    # 4. 补全长尾缺失职业
    print(">>> 步骤 4: 补全缺失职业的长尾分布...")
    matched_codes = set(df_sampled[code_col_t1].astype(str).unique())
    all_codes = set(df_dict[code_col].astype(str).unique())
    unmatched_codes = all_codes - matched_codes

    for code in unmatched_codes:
        row_dict = df_dict[df_dict[code_col].astype(str) == code].iloc[0]
        pseudo_anchor = str(row_dict[title_col])
        pseudo_positive = dict_map[code]
        train_examples.append(InputExample(texts=[pseudo_anchor, pseudo_positive]))

    # 5. 拼接困难样本增量数据
    if hard_examples:
        train_examples.extend(hard_examples)

    print(f"✅ 微调集构建完成：总计 {len(train_examples)} 条样本。")
    return train_examples


def train_finetuned_model():
    train_samples = build_training_data()
    if not train_samples:
        return

    # 清理显存碎片
    empty_cuda_cache_safe()

    device = resolve_torch_device()
    abs_local_path = os.path.abspath(LOCAL_MODEL_PATH)

    print(f"\n>>> 步骤 5: 从本地路径加载模型组件...")
    if not os.path.exists(abs_local_path):
        print(f"❌ 物理路径不存在: {abs_local_path}")
        return

    # 手动模块化加载，彻底绕过 HuggingFace Repo 验证逻辑
    try:
        word_embedding_model = models.Transformer(abs_local_path, max_seq_length=MAX_SEQ_LENGTH)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        print("✅ 本地模型加载成功。")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 启动训练
    print(f"开始在 {device.upper()} 上执行微调...")
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=int(len(train_dataloader) * 0.1),
            optimizer_params={'lr': LEARNING_RATE},
            output_path=OUTPUT_MODEL_PATH,
            show_progress_bar=True
        )
        print(f"\n🎉 领域自适应微调完成！新模型已保存至: {OUTPUT_MODEL_PATH}")
    except NameError as e:
        print(f"❌ 训练中断 (NameError): {e}")
        print("💡 建议：请确保已安装 'datasets' 库 (pip install datasets) 并重启此脚本。")
    except Exception as e:
        print(f"❌ 训练过程发生未知错误: {e}")


if __name__ == "__main__":
    train_finetuned_model()
