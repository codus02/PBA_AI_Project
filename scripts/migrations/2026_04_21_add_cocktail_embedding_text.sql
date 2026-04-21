BEGIN;

-- cocktails.embedding_text: Qwen3-Embedding 인코딩 원문 텍스트 (벡터 재생성용)
ALTER TABLE cocktails
    ADD COLUMN IF NOT EXISTS embedding_text TEXT;

COMMIT;
