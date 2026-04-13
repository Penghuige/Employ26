from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG 全局配置。

    说明：
    1) 默认模型路径按你的本地环境预设；
    2) 产物文件统一放在 src/rag/artifacts；
    3) 可通过 CLI 参数覆写。
    """

    # -------------------------
    # 1. 模型与知识库路径
    # -------------------------
    embedding_model_path: str = r"D:\model\bge-base-zh-finetuned"
    generator_model_path: str = r"D:\model\Qwen3-8B"
    kb_excel_path: str = r"data\中国职业大典.xlsx"

    # -------------------------
    # 2. 索引产物路径
    # -------------------------
    index_path: str = r"src\rag\artifacts\occupation_index.faiss"
    task_index_path: str = r"src\rag\artifacts\occupation_task_index.faiss"
    metadata_path: str = r"src\rag\artifacts\occupation_metadata.json"

    # -------------------------
    # 3. 推理参数
    # -------------------------
    embedding_batch_size: int = 128
    top_k: int = 8
    max_new_tokens: int = 320
    do_sample: bool = False
    temperature: float = 0.0

    # -------------------------
    # 4. 知识库字段候选
    # -------------------------
    title_candidates: tuple = ("title", "职业名称", "name")
    code_candidates: tuple = ("code", "职业代码", "id")
    desc_candidates: tuple = ("desc", "职业定义", "definition")
    task_candidates: tuple = ("tasks", "主要工作任务", "task")

    # -------------------------
    # 5. 分块策略
    # -------------------------
    # "merged": 每个职业生成一个 task chunk（默认，平衡速度与覆盖）
    # "item":   每个 task_item 生成独立 chunk（更细粒度，适合长任务列表）
    task_chunk_mode: str = "merged"
