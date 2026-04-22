"""cocktails.embedding_text 컬럼을 CSV로 채워넣기.

선행 조건:
  - 2026_04_21_add_cocktail_embedding_text.sql 실행되어 embedding_text 컬럼 존재.

CSV 형식 (헤더):
  cocktail_id, embedding_text

실행:
  python -m scripts.migrations.load_cocktail_embedding_text \
      --csv data/cocktail_embedding_text.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db.database import SessionLocal


DEFAULT_CSV = Path("data/cocktail_embedding_text.csv")


def main(csv_path: Path, dry_run: bool = False) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = {"cocktail_id", "embedding_text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV에 필수 컬럼 누락: {missing} (found: {list(df.columns)})")

    df = df[["cocktail_id", "embedding_text"]].dropna(subset=["cocktail_id"])
    df["cocktail_id"] = df["cocktail_id"].astype(int)
    df["embedding_text"] = df["embedding_text"].fillna("").astype(str)

    print(f"[1/2] CSV 로드 완료: {len(df)}건 ({csv_path})")
    print(f"  샘플 (id={df.iloc[0].cocktail_id}):\n    {df.iloc[0].embedding_text[:200]}...")

    if dry_run:
        print("[dry-run] DB 업데이트 생략")
        return

    db = SessionLocal()
    try:
        updated = 0
        missing_ids: list[int] = []
        for row in df.itertuples(index=False):
            res = db.execute(
                text(
                    "UPDATE cocktails SET embedding_text = :txt "
                    "WHERE cocktail_id = :cid"
                ),
                {"txt": row.embedding_text, "cid": int(row.cocktail_id)},
            )
            if res.rowcount:
                updated += 1
            else:
                missing_ids.append(int(row.cocktail_id))
        db.commit()
        print(f"[2/2] UPDATE 완료: {updated}건")
        if missing_ids:
            print(f"  ! cocktail_id {len(missing_ids)}건 DB에 없음: {missing_ids[:10]}...")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(csv_path=args.csv, dry_run=args.dry_run)
