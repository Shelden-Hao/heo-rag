import os
import sys
import json

# 把项目根目录加到 sys.path 最前面，让后续的 from app.config import settings 这类项目内部导入能找到模块。
# 如果不加这一行，直接用 python app/evaluation/evaluator.py 运行时会报模块找不到的错误（因为 Python 不会自动把项目根加入搜索路径）
# TODO: 后续统一封装路径工具
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# setdefault 仅在环境变量未设置时才写入值。这样可以确保 .env 或外部设置的值不会被覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 清华镜像
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows 上 HuggingFace 缓存符号链接会报 warning，关掉它
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 离线模式
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # 离线模式

# Windows 终端默认编码是 GBK（cp936），而 print() 输出的中文是 UTF-8
# 直接输出会报 UnicodeEncodeError 或乱码。reconfigure(encoding='utf-8') 强制 stdout/stderr 使用 UTF-8 编码输出中文
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

load_dotenv()


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        filename = doc.metadata.get("filename", "未知")
        parts.append(f"【来源：{filename}】\n{doc.page_content}")
    return "\n\n".join(parts)


def run_evaluation(dataset_path: str = None, top_k: int = 3, batch_size: int = 5):
    from app.config import settings
    from app.storage.db import init_db
    from app.embedding.local_embed import get_embeddings
    from app.retrieval.hybrid import hybrid_retrieve
    from app.generation.llm import get_llm
    from app.generation.prompt_templates import RAG_PROMPT
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"加载数据集，共 {len(dataset)} 条数据源。", flush=True)
    init_db()
    print("加载 embeddings 模型...", flush=True)
    get_embeddings()
    print("embeddings 模型已加载。", flush=True)

    rag_llm = get_llm()
    print("LLM 加载完成。", flush=True)

    from ragas import evaluate
    from datasets import Dataset as HFDataset
    from openai import OpenAI as OpenAIclient
    from ragas.llms import llm_factory
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    deepseek_client = OpenAIclient(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    ragas_llm = llm_factory("deepseek-chat", provider="openai", client=deepseek_client)
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    print("评估 LLM 和 embeddings 已初始化。", flush=True)

    # 记录模型原始问答
    records = []
    for i, item in enumerate(dataset):
        question = item["question"]
        print(f"\n[{i + 1}/{len(dataset)}] {question}", flush=True)

        docs = hybrid_retrieve(question, settings.retrieval_top_k)[:top_k]
        context_str = _format_docs(docs)
        print(f"  检索到 {len(docs)} 个文档片段，共 {len(context_str)} 字", flush=True)

        chain = (
                {
                    "context": RunnableLambda(lambda q, ctx=context_str: ctx),
                    "question": RunnablePassthrough(),
                }
                | RAG_PROMPT
                | rag_llm
                | StrOutputParser()
        )

        print("  生成回答...", flush=True)
        answer = chain.invoke(question)
        print(f"  回答：{answer[:120]}...", flush=True)

        records.append({
            "question": question,
            "contexts": [doc.page_content for doc in docs],
            "answer": answer,
            "ground_truth": item["ground_truth"],
        })

    eval_dataset = HFDataset.from_list(records)
    print(f"\n开始 ragas 评估，共 {len(eval_dataset)} 条样本...", flush=True)

    # 调用 ragas 评估
    result = evaluate(
        dataset=eval_dataset,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        batch_size=batch_size,
        raise_exceptions=True,
    )

    print("\n" + "=" * 60, flush=True)
    print("RAGAS 评估结果", flush=True)
    print("=" * 60, flush=True)
    for metric_name, score in result._repr_dict.items():  # result._repr_dict 是 ragas 算好的各指标平均分（比如 {"faithfulness": 0.8917, ...}）
        print(f"  {metric_name}: {score:.4f}", flush=True)

    # overall 存平均分，per_question 之后填充逐题得分
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    results_data = {
        "overall": {k: float(v) for k, v in result._repr_dict.items()},
        "per_question": [],
    }
    # 把 to_pandas() 返回的表格逐行拆开，从 records 里取出对应的题目和标准答案，加上该行各指标的得分，组成每道题的记录。
    df = result.to_pandas()
    for i, row in df.iterrows():
        q_data = {"question": records[i]["question"], "ground_truth": records[i]["ground_truth"]}
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                q_data[col] = round(val, 4)
        results_data["per_question"].append(q_data)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {results_path}", flush=True)

    return result


if __name__ == "__main__":
    run_evaluation()
