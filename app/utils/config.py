import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-2-9b-it")
DIALOGUE_LLM_MODEL = os.getenv("DIALOGUE_LLM_MODEL", LLM_MODEL)
SLOT_EXTRACTOR_BACKEND = os.getenv("SLOT_EXTRACTOR_BACKEND", "prompt").strip().lower()
SLOT_EXTRACTOR_MODEL = os.getenv("SLOT_EXTRACTOR_MODEL", DIALOGUE_LLM_MODEL)
SLOT_EXTRACTOR_ADAPTER_PATH = os.getenv("SLOT_EXTRACTOR_ADAPTER_PATH", "").strip()
QWEN2_VL_MODEL = os.getenv("QWEN2_VL_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
QWEN3_EMBED_MODEL = os.getenv("QWEN3_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/b2_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
