"""칵테일 임베딩 인덱싱 — Qwen3-Embedding-0.6B → pgvector.

각 칵테일의 description + 맛 레벨 + 카테고리/무드 + 주요 재료 텍스트를 합쳐
1024-dim 벡터로 인코딩 후 cocktails.embedding 컬럼에 저장한다.
HNSW 인덱스도 같이 생성/갱신한다.

실행:
    python -m scripts.build_cocktail_embeddings              # 전체 재인덱싱
    python -m scripts.build_cocktail_embeddings --only-missing  # 임베딩 없는 것만
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
from app.db.models import Cocktail, Ingredient, Recipe
from app.utils.model_loader import embed_texts


# 4 이상이면 강한 축, 2 이상이면 약하게 언급
def _level_phrase(label: str, value: float | None) -> str | None:
    if value is None:
        return None
    v = float(value)
    if v >= 4.0:
        return f"{label} 강함"
    if v >= 2.5:
        return f"{label} 중간"
    if v >= 1.0:
        return f"{label} 약함"
    return None


def _compose_text(c: Cocktail, ingredients: list[Ingredient]) -> str:
    taste_bits = [
        _level_phrase("단맛", c.sweet_level),
        _level_phrase("신맛", c.sour_level),
        _level_phrase("쓴맛", c.bitter_level),
        _level_phrase("바디감", c.body_level),
        _level_phrase("청량감", c.freshness_level),
        _level_phrase("크리미", c.creamy_level),
        _level_phrase("매콤", c.spicy_level),
        _level_phrase("고소", c.nutty_level),
    ]
    taste_line = ", ".join(b for b in taste_bits if b) or "특징적인 맛 축 없음"

    bases = [ing.ingredients_name for ing in ingredients if ing.ingredient_type == "BASE"]
    mixers = [ing.ingredients_name for ing in ingredients if ing.ingredient_type in ("MIXER", "SYRUP", "JUICE", "TOPPING")]
    base_line = ", ".join(bases) if bases else "베이스 정보 없음"
    mixer_line = ", ".join(mixers[:6]) if mixers else "보조 재료 없음"

    sensory_notes = sorted({(ing.sensory_note or "").strip() for ing in ingredients if ing.sensory_note})
    sensory_line = "; ".join(s for s in sensory_notes if s and "balanced" not in s.lower())[:300]

    mood = (c.mood_tag or "").replace("|", ", ")
    alc = "무알콜" if c.is_non_alcoholic else "알콜"

    parts = [
        f"{c.name_kr} ({c.name_en or ''}) — {c.category}, {alc}",
        f"무드: {mood}" if mood else "",
        c.description or "",
        f"맛 프로파일: {taste_line}",
        f"베이스: {base_line}",
        f"주요 보조 재료: {mixer_line}",
    ]
    if sensory_line:
        parts.append(f"재료 감각 키워드: {sensory_line}")
    return "\n".join(p for p in parts if p)


def _fetch_recipe_ingredients(db: Session) -> dict[int, list[Ingredient]]:
    rows = (
        db.query(Recipe, Ingredient)
        .join(Ingredient, Recipe.ingredient_id == Ingredient.ingredient_id)
        .all()
    )
    out: dict[int, list[Ingredient]] = {}
    for r, ing in rows:
        out.setdefault(r.cocktail_id, []).append(ing)
    return out


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

        print(f"[1/3] 레시피·재료 조회 ({len(cocktails)} 칵테일)")
        ri_map = _fetch_recipe_ingredients(db)

        print("[2/3] 임베딩 텍스트 합성")
        texts: list[str] = []
        for c in cocktails:
            texts.append(_compose_text(c, ri_map.get(c.cocktail_id, [])))
        print(f"  샘플 텍스트 (id={cocktails[0].cocktail_id}):\n----\n{texts[0][:400]}\n----")

        print(f"[3/3] Qwen3-Embedding 인코딩 (batch={batch_size})")
        t0 = time.perf_counter()
        vecs = embed_texts(texts, batch_size=batch_size, max_length=512)
        dt = time.perf_counter() - t0
        print(f"  인코딩 완료: {vecs.shape} in {dt:.1f}s ({dt/len(texts):.2f}s/cocktail)")

        for c, v in zip(cocktails, vecs):
            c.embedding = v.tolist()
        db.commit()
        print(f"  DB 업데이트 완료: {len(cocktails)}건")

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
