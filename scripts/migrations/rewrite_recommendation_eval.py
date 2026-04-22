"""data/eval/recommendation_eval_v2_500.csv를 신규 영문 enum 스키마에 맞게 재작성.

옛 스키마(한국어 값) → 신 스키마(영문 enum) 매핑.

변환:
  - party_purpose: 친구모임→hangout, 생일파티/축하자리→celebration, 데이트→date, 혼술→solo
  - current_mood : 신남→good, 차분함→soso, 우울함/스트레스→bad
  - strength_preference: 약함→light, 중간→medium, 강함→strong
  - preferred/disliked_tastes (영문 enum: sweet/sour/bitter/body/creamy/freshness):
      단맛/달콤함 → sweet
      상큼함/신맛 → sour
      쓴맛        → bitter
      청량함       → freshness
      부드러움/고소함 → creamy
      과일향       → taste에서 drop (aroma쪽으로 이동)
  - preferred/disliked_aromas (영문 enum: minty/fruity/citrus/herbal/coffee/woody/floral):
      민트향→minty, 과일향→fruity, 시트러스향→citrus, 허브향→herbal,
      커피향→coffee, 우디향→woody, 플로럴향→floral
  - disliked_bases (영문 enum: whiskey/gin/rum/vodka/tequila):
      위스키→whiskey, 진→gin, 럼→rum, 데킬라→tequila, 보드카→vodka

taste 컬럼에 섞여 있던 '과일향'은 같은 행의 aroma 컬럼에 merge.
space_mood_tags, gold_expected_cocktails, gold_allowed_categories 는 자유 문자열 → 그대로 유지.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CSV = Path("data/eval/recommendation_eval_v2_500.csv")


PARTY_MAP = {
    "친구모임": "hangout",
    "생일파티": "celebration",
    "축하자리": "celebration",
    "데이트": "date",
    "혼술": "solo",
}

MOOD_MAP = {
    "신남": "good",
    "차분함": "soso",
    "우울함": "bad",
    "스트레스": "bad",
}

STRENGTH_MAP = {
    "약함": "light",
    "중간": "medium",
    "강함": "strong",
}

TASTE_MAP = {
    "단맛": "sweet",
    "달콤함": "sweet",
    "상큼함": "sour",
    "신맛": "sour",
    "쓴맛": "bitter",
    "청량함": "freshness",
    "부드러움": "creamy",
    "고소함": "creamy",
}

AROMA_MAP = {
    "민트향": "minty",
    "과일향": "fruity",
    "시트러스향": "citrus",
    "허브향": "herbal",
    "커피향": "coffee",
    "우디향": "woody",
    "플로럴향": "floral",
}

# taste 컬럼에 잘못 들어간 aroma 어휘 — aroma 쪽으로 이동
TASTE_TO_AROMA_BLEED = {"과일향"}

BASE_MAP = {
    "위스키": "whiskey",
    "진": "gin",
    "럼": "rum",
    "데킬라": "tequila",
    "보드카": "vodka",
}


def _parse_list(val) -> list:
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        v = json.loads(val) if isinstance(val, str) else val
        return list(v) if isinstance(v, list) else []
    except Exception:
        return []


def _dump_list(values: list) -> str:
    return json.dumps(values, ensure_ascii=False)


def _map_scalar(val, mapping: dict[str, str]):
    if pd.isna(val):
        return val
    s = str(val).strip()
    return mapping.get(s, s)


def _map_list(values: list, mapping: dict[str, str]) -> list:
    out = []
    seen = set()
    for v in values:
        k = str(v).strip()
        mapped = mapping.get(k, None)
        if mapped is None or mapped in seen:
            continue
        out.append(mapped)
        seen.add(mapped)
    return out


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    print(f"[1/2] 로드: {n} rows")

    # scalar 매핑
    df["party_purpose"] = df["party_purpose"].map(lambda v: _map_scalar(v, PARTY_MAP))
    df["current_mood"] = df["current_mood"].map(lambda v: _map_scalar(v, MOOD_MAP))
    df["strength_preference"] = df["strength_preference"].map(
        lambda v: _map_scalar(v, STRENGTH_MAP)
    )

    # list 매핑 (taste / aroma / bases)
    new_pref_tastes, new_disl_tastes = [], []
    new_pref_aromas, new_disl_aromas = [], []
    new_bases = []

    for _, row in df.iterrows():
        raw_pt = _parse_list(row.get("preferred_tastes"))
        raw_dt = _parse_list(row.get("disliked_tastes"))
        raw_pa = _parse_list(row.get("preferred_aromas"))
        raw_da = _parse_list(row.get("disliked_aromas"))
        raw_bs = _parse_list(row.get("disliked_bases"))

        # taste에 섞여있던 aroma 어휘 분리
        pt_taste = [t for t in raw_pt if t not in TASTE_TO_AROMA_BLEED]
        pt_aroma_bleed = [t for t in raw_pt if t in TASTE_TO_AROMA_BLEED]
        dt_taste = [t for t in raw_dt if t not in TASTE_TO_AROMA_BLEED]
        dt_aroma_bleed = [t for t in raw_dt if t in TASTE_TO_AROMA_BLEED]

        new_pref_tastes.append(_dump_list(_map_list(pt_taste, TASTE_MAP)))
        new_disl_tastes.append(_dump_list(_map_list(dt_taste, TASTE_MAP)))
        new_pref_aromas.append(
            _dump_list(_map_list(raw_pa + pt_aroma_bleed, AROMA_MAP))
        )
        new_disl_aromas.append(
            _dump_list(_map_list(raw_da + dt_aroma_bleed, AROMA_MAP))
        )
        new_bases.append(_dump_list(_map_list(raw_bs, BASE_MAP)))

    df["preferred_tastes"] = new_pref_tastes
    df["disliked_tastes"] = new_disl_tastes
    df["preferred_aromas"] = new_pref_aromas
    df["disliked_aromas"] = new_disl_aromas
    df["disliked_bases"] = new_bases

    df.to_csv(CSV, index=False)
    print(f"[2/2] 저장: {CSV}  (backup: {CSV}.bak)")

    # sanity 체크
    print("\n-- sanity check --")
    for col in ("party_purpose", "current_mood", "strength_preference"):
        print(f"  {col}: {dict(df[col].value_counts(dropna=False))}")
    print(f"  preferred_tastes sample: {df['preferred_tastes'].iloc[0]}")
    print(f"  preferred_aromas sample: {df['preferred_aromas'].iloc[0]}")
    print(f"  disliked_bases samples : {df['disliked_bases'].iloc[:5].tolist()}")


if __name__ == "__main__":
    main()
