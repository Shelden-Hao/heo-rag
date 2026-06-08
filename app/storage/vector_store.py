import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.config import settings
from app.embedding.local_embed import get_embeddings


_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        os.makedirs(settings.chroma_dir, exist_ok=True)
        _vectorstore = Chroma(
            collection_name="rag_chunks",
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_dir,
        )
    return _vectorstore


def add_documents(documents: list[Document]):
    vs = get_vectorstore()
    vs.add_documents(documents)


def search(query: str, top_k: int = 5) -> list[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query, k=top_k)


def search_with_score(query: str, top_k: int = 5) -> list[tuple[Document, float]]:
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=top_k)


def get_retriever(top_k: int | None = None):
    vs = get_vectorstore()
    k = top_k or settings.retrieval_top_k
    return vs.as_retriever(search_kwargs={"k": k})
