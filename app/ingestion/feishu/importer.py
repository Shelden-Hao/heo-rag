from langchain_core.documents import Document
from app.ingestion.feishu.client import feishu_client
from app.ingestion.feishu.converter import blocks_to_document
from app.chunking.splitter import split_documents
from app.storage.vector_store import add_documents


def import_feishu_document(document_id: str) -> dict:
    print(f"Fetching document: {document_id}")
    blocks = feishu_client.get_document_blocks(document_id)

    title = "feishu_doc"
    for block in blocks:
        if block.get("block_type") == 2:  # page block
            text_body = block.get("page", {})
            if text_body and "elements" in text_body:
                elements = text_body["elements"]
                for elem in elements:
                    if "text_run" in elem:
                        title = elem["text_run"].get("content", title)
                        break
            break

    doc = blocks_to_document(blocks, title, document_id)
    print(f"  Title: {title}, Content length: {len(doc.page_content)}")

    chunks = split_documents([doc])
    print(f"  Split into {len(chunks)} chunks")

    add_documents(chunks)
    print(f"  Stored into Chroma")

    return {
        "raw_documents": 1,
        "chunks": len(chunks),
        "title": title,
    }


def import_feishu_folder(folder_token: str) -> dict:
    print(f"Listing documents in folder: {folder_token}")
    docs_meta = feishu_client.list_documents(folder_token)
    print(f"Found {len(docs_meta)} docx documents")

    all_docs: list[Document] = []
    for meta in docs_meta:
        doc_id = meta["token"]
        title = meta["name"]
        print(f"  Fetching: {title} ({doc_id})")
        try:
            blocks = feishu_client.get_document_blocks(doc_id)
            doc = blocks_to_document(blocks, title, doc_id)
            all_docs.append(doc)
        except Exception as e:
            print(f"  Warning: Failed to fetch {title}: {e}")

    if not all_docs:
        return {"raw_documents": 0, "chunks": 0}

    print(f"Loaded {len(all_docs)} raw documents from Feishu")

    chunks = split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")

    add_documents(chunks)
    print(f"Stored {len(chunks)} chunks into Chroma")

    return {
        "raw_documents": len(all_docs),
        "chunks": len(chunks),
    }
