from app.config import settings
from app.ingestion.loader import load_documents
from app.chunking.splitter import split_documents
from app.storage.vector_store import add_documents


def import_documents(directory: str | None = None) -> dict:
    docs = load_documents(directory)
    print(f"Loaded {len(docs)} raw documents")

    if not docs:
        return {"raw_documents": 0, "chunks": 0}

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    for chunk in chunks:
        if "source" in chunk.metadata:
            chunk.metadata["filename"] = chunk.metadata["source"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]

    add_documents(chunks)
    print(f"Stored {len(chunks)} chunks into Chroma")

    return {
        "raw_documents": len(docs),
        "chunks": len(chunks),
    }
