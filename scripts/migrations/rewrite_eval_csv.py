"""data/eval/slot_extraction_eval_v2_500.csv를 신규 DB 스키마에 맞게 재작성.

변경사항:
  - 컬럼명 DB 일치: taste_profile_json / aroma_profile_json / disliked_bases_json / favorite_drinks_json
  - gold_finish_preference 컬럼 삭제 (스키마에서 제거됨)
  - gold_party_purpose 값 리매핑:
      casual_hangout → hangout
      social         → hangout
      after_work     → business
      party          → celebration
      special_day    → celebration
  - gold_current_mood: 기존 값은 전부 vibe 라벨(mood_tag용)이므로 모두 비움.
      이후 user_text에서 명시적 기분 표현만 good/soso/bad로 백필.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CSV = Path("data/eval/slot_extraction_eval_v2_500.csv")

PARTY_MAP = {
    "casual_hangout": "hangout",
    "social": "hangout",
    "after_work": "business",
    "party": "celebration",
    "special_day": "celebration",
    "date": "date",
    "celebration": "celebration",
    "business": "business",
    "solo": "solo",
    "hangout": "hangout",
}

MOOD_GOOD = ["기분 좋", "기분좋", "신나", "신난", "들떠", "설레", "행복", "최고", "기뻐", "기쁘"]
MOOD_BAD = ["우울", "기분 안 좋", "기분안좋", "기분 별로", "꿀꿀", "짜증", "지쳐", "피곤", "힘들", "스트레스", "울적"]
MOOD_SOSO = ["그냥 그래", "그냥그래", "보통이야", "보통정도", "그럭저럭", "나쁘진 않", "무난"]


def derive_mood(text: str) -> str:
    t = text or ""
    for kw in MOOD_BAD:
        if kw in t:
            return "bad"
    for kw in MOOD_GOOD:
        if kw in t:
            return "good"
    for kw in MOOD_SOSO:
        if kw in t:
            return "soso"
    return ""


def main():
    df = pd.read_csv(CSV)

    df = df.drop(columns=["gold_finish_preference"], errors="ignore")

    df = df.rename(
        columns={
            "gold_taste_profile": "gold_taste_profile_json",
            "gold_aroma_profile": "gold_aroma_profile_json",
            "gold_disliked_bases": "gold_disliked_bases_json",
            "gold_favorite_drinks": "gold_favorite_drinks_json",
        }
    )

    df["gold_party_purpose"] = df["gold_party_purpose"].map(
        lambda v: PARTY_MAP.get(v, "") if isinstance(v, str) and v else ""
    )
    df["gold_party_purpose"] = df["gold_party_purpose"].replace("", pd.NA)

    df["gold_current_mood"] = df["user_text"].map(derive_mood)
    df["gold_current_mood"] = df["gold_current_mood"].replace("", pd.NA)

    cols = [
        "case_id",
        "user_text",
        "gold_current_mood",
        "gold_party_purpose",
        "gold_taste_profile_json",
        "gold_aroma_profile_json",
        "gold_strength_preference",
        "gold_disliked_bases_json",
        "gold_favorite_drinks_json",
    ]
    df = df[cols]

    df.to_csv(CSV, index=False)

    print(f"rewrote {CSV} ({len(df)} rows)")
    print("\n[gold_party_purpose counts]")
    print(df["gold_party_purpose"].value_counts(dropna=False))
    print("\n[gold_current_mood counts]")
    print(df["gold_current_mood"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
