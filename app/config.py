import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

if os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING"):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class Settings(BaseSettings):
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    feishu_app_id: str = ""
    feishu_app_secret: str = ""

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    documents_dir: str = "./data/documents"
    db_path: str = "./data/rag.db"
    chroma_dir: str = "./data/chroma"

    retrieval_top_k: int = 5
    hybrid_weight_vector: float = 0.6
    hybrid_weight_keyword: float = 0.4
    chunk_size: int = 512
    chunk_overlap: int = 128

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
