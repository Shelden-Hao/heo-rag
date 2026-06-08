import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from app.config import settings
from app.storage.vector_store import get_retriever, get_vectorstore


# 分词
def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


# BM25 检索器
# BM25 是什么：经典的关键词检索算法（搜索引擎用的核心算法），原理是：
# - 查询词在文档中出现次数越多 → 得分越高
# - 文档越短，同样的出现次数 → 得分越高（短文档更精准）
# - 词在越少文档中出现 → 权重越高（类似 TF-IDF）
# 举例：查询"事件循环"
# - "事件循环其实很简单！.md" 包含"事件循环"多次 → 高分
# - "秀秀训练营简介.md" 不包含"事件循环" → 低分
class BM25Retriever:
    def __init__(self, corpus: list[Document], k: int = 5):
        self.corpus = corpus  # 保存所有文档
        self.k = k  # 返回 top_k 个结果
        tokenized = [_tokenize(doc.page_content) for doc in corpus]  # 把每个文档的内容都分词
        self.bm25 = BM25Okapi(tokenized)  # 用 BM25 算法建立倒排索引

    def invoke(self, query: str) -> list[Document]:
        scores = self.bm25.get_scores(_tokenize(query))  # 计算查询条件和每个文档的得分
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)  # 把文档按得分降序排序
        return [self.corpus[i] for i, _ in ranked[:self.k]]  # 返回 top_k 个结果


def _load_all_documents() -> list[Document]:
    vs = get_vectorstore()
    data = vs.get()  # 取出全部文档（内容 + 元数据）
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(docs, metas)
    ]


# 混合检索 - 向量检索（语义匹配） + BM25 检索（精确匹配），然后合并结果
def hybrid_retrieve(query: str, top_k: int | None = None) -> list[Document]:
    k = top_k or settings.retrieval_top_k
    weight_vec = settings.hybrid_weight_vector
    weight_key = settings.hybrid_weight_keyword

    vector_retriever = get_retriever(k * 2)  # 取 2k，后面合并时可能被 BM25 抢位
    vector_results = vector_retriever.invoke(query)  # Chroma 余弦相似度检索

    corpus = _load_all_documents()
    if not corpus:
        return vector_results[:k]

    bm25_retriever = BM25Retriever(corpus, k * 2)
    bm25_results = bm25_retriever.invoke(query)  # # BM25 关键词检索

    def _doc_key(doc: Document) -> str:
        return doc.page_content[:200]  # 取前 200 个字符作为 key 文档唯一标识

    # 这里开始去重
    seen = {}  # 存储已处理的文档
    merged = []  # 合并后的结果：[(文档, 得分), ...]

    # 第一轮：把向量检索结果放进去，每篇得分比如 = weight_vec（0.3），重要性占比 0.3
    for doc in vector_results:
        key = _doc_key(doc)
        seen[key] = len(merged)
        merged.append((doc, weight_vec))

    # 第二轮：把 BM25 结果放进去
    for doc in bm25_results:
        key = _doc_key(doc)
        if key in seen: # 如果向量检索也找到了这篇文档
            idx = seen[key]
            doc_old, w_old = merged[idx]
            merged[idx] = (doc_old, w_old + weight_key) # 得分叠加：0.3 + 0.7 = 1.0
        else:  # BM25 独家找到的
            seen[key] = len(merged)
            merged.append((doc, weight_key))

    # 按得分降序排列，取 top_k
    merged.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in merged[:k]]
