import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.pipeline.rag_pipeline import ask
from app.pipeline.ingestion_pipeline import import_documents
from app.config import settings

app = FastAPI(title="秀秀训练营 RAG API", version="0.2.0")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_rerank: bool = True


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]


@app.on_event("startup")
def startup():
    from app.storage.db import init_db
    init_db()
    from app.embedding.local_embed import get_embeddings
    get_embeddings()
    print("Embedding model loaded.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    try:
        return ask(req.query, req.use_rerank)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/import/file")
def import_file(filepath: str):
    try:
        from app.pipeline.ingestion_pipeline import import_documents
        result = import_documents(filepath.rsplit("\\", 1)[0].rsplit("/", 1)[0])
        return {"message": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/import/directory")
def import_dir(directory: Optional[str] = None):
    try:
        result = import_documents(directory or settings.documents_dir)
        return {"message": f"Imported {result['chunks']} chunks", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/import/feishu")
def import_feishu(folder_token: str):
    try:
        from app.ingestion.feishu.importer import import_feishu_folder
        result = import_feishu_folder(folder_token)
        return {"message": f"Imported from Feishu: {result['chunks']} chunks", **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
