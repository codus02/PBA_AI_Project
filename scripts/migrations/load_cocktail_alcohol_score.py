"""cocktails.alcohol_score 컬럼을 CSV로 채워넣기.

선행 조건:
  - 2026_04_21_add_cocktail_alcohol_score.sql 실행되어 alcohol_score 컬럼 존재.

CSV 형식 (헤더 필수):
  cocktail_id, alcohol_score
  ※ alcohol_score 값은 **raw ABV %** (예: 28). 로더 내부에서
     0~5 스케일로 변환된 뒤 DB에 저장됨.

변환식: alcohol_score(DB) = min(abv_percent / 8.0, 5.0)
  (40% ABV 가 5.0 상한. STRENGTH_RANGE의 light/medium/strong 버킷과 맞춤.)

실행:
  python -m scripts.migrations.load_cocktail_alcohol_score \
      --csv data/cocktail_alcohol_score.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db.database import SessionLocal


DEFAULT_CSV = Path("data/cocktail_alcohol_score.csv")
ABV_MAX = 40.0  # 40% ABV → score 5.0


def abv_to_score(abv_percent: float) -> float:
    """raw ABV% → 0~5 점수. min(abv/8, 5.0)."""
    return round(min(max(abv_percent, 0.0) / 8.0, 5.0), 1)


def main(csv_path: Path, dry_run: bool = False) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = {"cocktail_id", "alcohol_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV에 필수 컬럼 누락: {missing} (found: {list(df.columns)})")

    df = df.dropna(subset=["cocktail_id", "alcohol_score"])
    df["cocktail_id"] = df["cocktail_id"].astype(int)
    df["alcohol_score"] = df["alcohol_score"].astype(float)

    # 범위 체크: raw ABV 는 0~100
    out_of_range = df[(df["alcohol_score"] < 0.0) | (df["alcohol_score"] > 100.0)]
    if len(out_of_range):
        raise SystemExit(
            f"alcohol_score(raw ABV) 값이 0~100 범위 밖: {len(out_of_range)}건\n"
            f"{out_of_range.head(10)}"
        )

    df["score_converted"] = df["alcohol_score"].map(abv_to_score)

    print(f"[1/2] CSV 로드: {len(df)}건 ({csv_path})")
    print(f"  raw ABV 분포: min={df.alcohol_score.min()}, "
          f"max={df.alcohol_score.max()}, mean={df.alcohol_score.mean():.2f}")
    print(f"  변환 score 분포: min={df.score_converted.min()}, "
          f"max={df.score_converted.max()}, mean={df.score_converted.mean():.2f}")
    print(f"  샘플 변환 (id=1): ABV {df.iloc[0].alcohol_score}% → score {df.iloc[0].score_converted}")

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
                    "UPDATE cocktails SET alcohol_score = :v "
                    "WHERE cocktail_id = :cid"
                ),
                {"v": float(row.score_converted), "cid": int(row.cocktail_id)},
            )
            if res.rowcount:
                updated += 1
            else:
                missing_ids.append(int(row.cocktail_id))
        db.commit()
        print(f"[2/2] UPDATE 완료: {updated}건")
        if missing_ids:
            print(f"  ! cocktail_id {len(missing_ids)}건 DB에 없음: {missing_ids[:10]}...")

        total_row = db.execute(text("SELECT COUNT(*) FROM cocktails")).scalar()
        filled_row = db.execute(
            text("SELECT COUNT(*) FROM cocktails WHERE alcohol_score IS NOT NULL")
        ).scalar()
        print(f"  커버리지: {filled_row}/{total_row} "
              f"({filled_row/max(total_row,1)*100:.1f}%)")
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(csv_path=args.csv, dry_run=args.dry_run)
