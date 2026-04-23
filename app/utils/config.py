import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

LLM_MODEL = os.getenv("LLM_MODEL", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf")  # "hf" | "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QWEN2_VL_MODEL = os.getenv("QWEN2_VL_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
QWEN3_EMBED_MODEL = os.getenv("QWEN3_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/b2_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)