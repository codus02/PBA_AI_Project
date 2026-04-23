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

from app.db.crud import get_all_recipes_with_ingredients
from app.db.database import SessionLocal, engine
from app.db.models import Cocktail
from app.utils.model_loader import embed_texts

TASTE_LABELS = {
    "sweet_level": "단맛",
    "sour_level": "신맛",
    "bitter_level": "쓴맛",
    "body_level": "바디감",
    "freshness_level": "청량감",
    "creamy_level": "크리미함",
}

AROMA_LABELS = {
    "minty_score": "민트향",
    "fruity_score": "과일향",
    "citrus_score": "시트러스향",
    "herbal_score": "허브향",
    "coffee_score": "커피향",
    "woody_score": "우디향",
    "floral_score": "꽃향",
}


def _strength_phrase(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 3.5:
        return "강한 도수"
    if value >= 2.0:
        return "중간 도수"
    if value > 0:
        return "가벼운 도수"
    return "무알콜"


def _level_phrase(value: float) -> str:
    if value >= 3.5:
        return "강함"
    if value >= 2.5:
        return "뚜렷함"
    if value >= 1.5:
        return "중간"
    return "은은함"


def _aggregate_aroma(recipe_items: list[tuple]) -> tuple[dict[str, float], dict[str, str]]:
    agg: dict[str, float] = {}
    max_ingredient_name: dict[str, str] = {}
    for _recipe, ingredient in recipe_items or []:
        ing_name = getattr(ingredient, "ingredients_name", "") or ""
        for col in AROMA_LABELS:
            raw = getattr(ingredient, col, None)
            if raw is None:
                continue
            value = float(raw)
            if value > agg.get(col, 0.0):
                agg[col] = value
                max_ingredient_name[col] = ing_name
    return agg, max_ingredient_name


def _build_embedding_text(cocktail: Cocktail, recipe_items: list[tuple]) -> str:
    base = (cocktail.embedding_text or "").strip()
    extras: list[str] = []

    extras.append(f"카테고리: {cocktail.category}")
    if cocktail.mood_tag:
        extras.append(f"무드 태그: {str(cocktail.mood_tag).replace('|', ', ')}")
    strength_val = float(cocktail.alcohol_score) if cocktail.alcohol_score is not None else None
    strength_phrase = _strength_phrase(strength_val)
    if strength_phrase:
        extras.append(f"도수 프로필: {strength_phrase}" + (f" ({strength_val:.1f}/5)" if strength_val is not None else ""))
    extras.append("논알콜 여부: 무알콜" if cocktail.is_non_alcoholic else "논알콜 여부: 알코올 포함")

    taste_scores: list[str] = []
    taste_highlights: list[str] = []
    for col, label in TASTE_LABELS.items():
        raw = getattr(cocktail, col, None)
        if raw is None:
            continue
        value = float(raw)
        taste_scores.append(f"{label} {value:.1f}")
        if value >= 2.5:
            taste_highlights.append(f"{label} {_level_phrase(value)}")
    if taste_scores:
        extras.append("맛 점수: " + ", ".join(taste_scores))
    if taste_highlights:
        extras.append("대표 맛: " + ", ".join(taste_highlights))

    aroma_scores, aroma_sources = _aggregate_aroma(recipe_items)
    aroma_parts: list[str] = []
    aroma_highlights: list[str] = []
    for col, label in AROMA_LABELS.items():
        value = aroma_scores.get(col, 0.0)
        if value < 2.0:
            continue
        src = aroma_sources.get(col)
        aroma_parts.append(f"{label} {value:.1f}" + (f" ({src})" if src else ""))
        if value >= 2.5:
            aroma_highlights.append(f"{label} {_level_phrase(value)}")
    if aroma_parts:
        extras.append("향 점수(재료 기준): " + ", ".join(aroma_parts))
    if aroma_highlights:
        extras.append("대표 향: " + ", ".join(aroma_highlights))

    structured = "\n".join(extras)
    if not base:
        return structured
    return f"{base}\n\n[구조화 프로필]\n{structured}"


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
        all_ri = get_all_recipes_with_ingredients(db)
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

        texts_list = [_build_embedding_text(c, all_ri.get(c.cocktail_id, [])) for c in targets]
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
