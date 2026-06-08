import sys
import uvicorn
from app.config import settings
from app.storage.db import init_db


def main():
    init_db()
    print(f"Database initialized at: {settings.db_path}")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "import":
            from app.pipeline.ingestion_pipeline import import_documents
            target = sys.argv[2] if len(sys.argv) > 2 else settings.documents_dir
            result = import_documents(target)
            print(f"Imported: {result['raw_documents']} raw docs -> {result['chunks']} chunks")
            return
        elif cmd == "import-feishu":
            if len(sys.argv) < 3:
                print("Usage: python main.py import-feishu <folder_token>")
                print("       python main.py import-feishu <document_id>")
                return
            from app.ingestion.feishu.importer import import_feishu_folder, import_feishu_document
            target = sys.argv[2]
            if len(target) > 30:
                result = import_feishu_document(target)
            else:
                result = import_feishu_folder(target)
            print(f"Imported from Feishu: {result['raw_documents']} raw docs -> {result['chunks']} chunks")
            return
        elif cmd == "ask":
            if len(sys.argv) < 3:
                print("Usage: python main.py ask <query>")
                return
            from app.pipeline.rag_pipeline import ask
            result = ask(" ".join(sys.argv[2:]))
            print(f"\nQ: {result['query']}")
            print(f"A: {result['answer']}")
            return

    uvicorn.run("app.api.routes:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
