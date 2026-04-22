"""칵테일 임베딩 인덱싱 — cocktails.embedding_text → Qwen3-Embedding → pgvector.

각 칵테일 행의 embedding_text(큐레이션 원문)을 그대로 1024-dim 벡터로 인코딩해서
cocktails.embedding 컬럼에 저장한다. HNSW 인덱스도 같이 생성/갱신한다.

선행 조건:
  - 2026_04_21_add_cocktail_embedding_text.sql 실행됨
  - load_cocktail_embedding_text.py 로 embedding_text 채워짐

실행:
    python -m scripts.build_cocktail_embeddings              # 전체 재인덱싱
    python -m scripts.build_cocktail_embeddings --only-missing  # embedding이 NULL인 것만
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, engine
from app.db.models import Cocktail
from app.utils.model_loader import embed_texts


def _ensure_hnsw_index() -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cocktails_embedding_hnsw "
            "ON cocktails USING hnsw (embedding vector_cosine_ops)"
        ))


def main(only_missing: bool = False, batch_size: int = 16) -> None:
    db: Session = SessionLocal()
    try:
        q = db.query(Cocktail).filter(Cocktail.is_active.is_(True))
        if only_missing:
            q = q.filter(Cocktail.embedding.is_(None))
        cocktails = q.order_by(Cocktail.cocktail_id).all()
        if not cocktails:
            print("처리할 칵테일 없음.")
            return

        targets: list[Cocktail] = []
        skipped: list[int] = []
        for c in cocktails:
            if c.embedding_text and c.embedding_text.strip():
                targets.append(c)
            else:
                skipped.append(c.cocktail_id)

        if skipped:
            print(f"[!] embedding_text 비어있는 {len(skipped)}건 스킵: {skipped[:10]}...")
            print("    → scripts/migrations/load_cocktail_embedding_text.py 먼저 실행 필요")

        if not targets:
            print("인코딩할 대상 없음.")
            return

        texts_list = [c.embedding_text for c in targets]
        print(f"[1/2] Qwen3-Embedding 인코딩 ({len(targets)} 칵테일, batch={batch_size})")
        print(f"  샘플 (id={targets[0].cocktail_id}):\n----\n{texts_list[0][:400]}\n----")

        t0 = time.perf_counter()
        vecs = embed_texts(texts_list, batch_size=batch_size, max_length=512)
        dt = time.perf_counter() - t0
        print(f"  완료: {vecs.shape} in {dt:.1f}s ({dt/len(targets):.2f}s/cocktail)")

        for c, v in zip(targets, vecs):
            c.embedding = v.tolist()
        db.commit()
        print(f"[2/2] DB 업데이트 완료: {len(targets)}건")

        print("[+] HNSW 인덱스 보장")
        _ensure_hnsw_index()
        print("완료.")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-missing", action="store_true",
                    help="embedding이 NULL인 칵테일만 처리")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    main(only_missing=args.only_missing, batch_size=args.batch_size)
