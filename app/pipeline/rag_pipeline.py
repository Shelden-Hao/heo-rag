import os
import sys

# 支持直接运行：将项目根目录加入 sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 必须在导入 sentence-transformers 之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from app.config import settings
from app.storage.db import get_connection, init_db
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import rerank
from app.generation.llm import get_llm
from app.generation.prompt_templates import SYSTEM_PROMPT, RAG_PROMPT


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        filename = doc.metadata.get("filename", "未知")
        parts.append(f"【来源：{filename}】\n{doc.page_content}")
    return "\n\n".join(parts)


class HybridRetriever:
    def __init__(self, k: int = 5):
        self.k = k

    def invoke(self, query: str):
        return hybrid_retrieve(query, self.k)


def ask(query: str, use_rerank: bool = True) -> dict:
    retriever = HybridRetriever(k=settings.retrieval_top_k)
    llm = get_llm()

    rag_chain = (
            {
                "context": RunnableLambda(retriever.invoke) | _format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    raw_docs = retriever.invoke(query)
    if use_rerank and raw_docs:
        docs = rerank(query, raw_docs, top_k=3)
    else:
        docs = raw_docs[:3]

    sources = [
        {
            "content": doc.page_content[:200],  # 引用预览
            "filename": doc.metadata.get("filename", ""),
        }
        for doc in docs
    ]

    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations (query, answer, sources) VALUES (?, ?, ?)",
        (query, answer, json.dumps(sources, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    init_db()
    print("DB initialized. Loading embedding model...")

    from app.embedding.local_embed import get_embeddings

    get_embeddings()
    print("Model loaded.\n")

    result = ask("秀秀训练营中如何讲解js事件循环？", use_rerank=False)
    print(f"Q: {result['query']}")
    print(f"A: {result['answer']}")
