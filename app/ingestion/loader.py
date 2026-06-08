import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from app.config import settings


def load_documents(directory: str | None = None) -> list[Document]:
    root = directory or settings.documents_dir
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Documents directory not found: {root}")

    all_docs = []
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext in (".md", ".markdown"):
                loader = TextLoader(fpath, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyMuPDFLoader(fpath)
            else:
                continue
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = fpath
                doc.metadata["filename"] = fname
            all_docs.extend(docs)
            print(f"  Loaded: {fname} ({len(docs)} pages)")
        except Exception as e:
            print(f"  Warning: Failed to load {fname}: {e}")

    return all_docs
