"""data/eval/recommendation_eval_v2_500.csv의 gold_allowed_categories를
DB cocktails.category 값 기준으로 재작성.

기존: 영문 다중 태그 (Highball/Fresh/Low-ABV 등) — DB 한글 카테고리와 불일치
신규: gold_expected_cocktails의 각 칵테일의 실제 DB category 집합

원본은 .bak 로 백업.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db.database import SessionLocal
from app.db.models import Cocktail

CSV = Path("data/eval/recommendation_eval_v2_500.csv")


def main():
    backup = CSV.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy(CSV, backup)
        print(f"백업: {backup}")

    db = SessionLocal()
    try:
        name_to_cat = {c.name_kr: c.category for c in db.query(Cocktail).all()}
    finally:
        db.close()

    df = pd.read_csv(CSV)

    new_col = []
    unresolved = set()
    for v in df["gold_expected_cocktails"]:
        try:
            names = json.loads(v) if isinstance(v, str) and v.strip() else []
        except Exception:
            names = []
        cats = []
        for n in names:
            cat = name_to_cat.get(n)
            if cat is None:
                unresolved.add(n)
                continue
            if cat not in cats:
                cats.append(cat)
        new_col.append(json.dumps(cats, ensure_ascii=False))

    df["gold_allowed_categories"] = new_col
    df.to_csv(CSV, index=False)

    print(f"rewrote {CSV} ({len(df)} rows)")
    if unresolved:
        print(f"⚠️  DB에 없는 칵테일 {len(unresolved)}개: {sorted(unresolved)}")
    print("\n[gold_allowed_categories 분포 top 15]")
    print(df["gold_allowed_categories"].value_counts().head(15))


if __name__ == "__main__":
    main()
