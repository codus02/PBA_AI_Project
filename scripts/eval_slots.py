# scripts/eval_slots.py
"""
슬롯 추출 평가 스크립트
gold 데이터(영어) → 우리 출력 형식(한국어)으로 정규화 후 비교
"""
import sys, json
sys.path.insert(0, ".")

import pandas as pd
from app.agents.preference_agent import run_dialogue

# ── gold → 우리 형식 매핑 ─────────────────────────────────────────

PARTY_PURPOSE_MAP = {
    "birthday":       "생일 파티",
    "date":           "데이트",
    "casual_hangout": "친목 모임",
    "social":         "친목 모임",
    "after_work":     "회식",
    "business":       "회식",
    "work":           "회식",
    "party":          "파티",
    "celebration":    "파티",
    "special_day":    "기념일",
    "anniversary":    "기념일",
    "solo":           "혼술",
    "alone":          "혼술",
}

CURRENT_MOOD_MAP = {
    "social":    "즐거움",
    "lively":    "신남",
    "relaxing":  "편안함",
    "dark":      "우울함",
    "cool":      "편안함",
    "casual":    "편안함",
    "spacious":  "편안함",
    "brunch":    "편안함",
    "classic":   "편안함",
    "private":   "편안함",
    "happy":     "기쁨",
    "excited":   "신남",
    "cozy":      "편안함",
    "calm":      "편안함",
    "tired":     "피곤함",
    "sad":       "슬픔",
}

STRENGTH_MAP = {
    "light":  "약함",
    "medium": "중간",
    "strong": "강함",
    "low":    "약함",
    "high":   "강함",
}

FINISH_MAP = {
    "clean":  "깔끔하게",
    "crisp":  "깔끔하게",
    "light":  "깔끔하게",
    "smooth": "여운 있게",
    "heavy":  "여운 있게",
    "long":   "여운 있게",
    "rich":   "여운 있게",
}

# taste_profile: JSON {"sour":"low","sweet":"high",...} → 한국어 리스트
TASTE_KEY_MAP = {
    "sweet":     "단맛",
    "bitter":    "쓴맛",
    "sour":      "신맛",
    "fresh":     "청량함",
    "freshness": "청량함",
    "body":      "바디감",
    "creamy":    "크리미함",
    "spicy":     "스파이시",
    "nutty":     "고소함",
}

# aroma_profile: JSON {"fruity":"high","minty":"low",...} → 한국어 리스트
AROMA_KEY_MAP = {
    "minty":    "민트향",
    "mint":     "민트향",
    "fruity":   "과일향",
    "fruit":    "과일향",
    "citrus":   "시트러스향",
    "herbal":   "허브향",
    "herb":     "허브향",
    "coffee":   "커피향",
    "woody":    "우디향",
    "wood":     "우디향",
    "floral":   "꽃향",
    "flower":   "꽃향",
}

# disliked_bases: ["whiskey","rum",...] → 한국어
BASE_MAP = {
    "whiskey": "위스키",
    "rum":     "럼",
    "vodka":   "보드카",
    "gin":     "진",
    "tequila": "데킬라",
    "sake":    "사케",
    "beer":    "맥주",
    "wine":    "와인",
    "soju":    "소주",
}


def _norm_single(val, mapping):
    if pd.isna(val) or str(val).strip() == "":
        return None
    return mapping.get(str(val).strip().lower())


def _parse_taste_profile(val) -> list:
    """{"sweet":"high","sour":"low"} → "high"인 것만 한국어 리스트로"""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        d = json.loads(val)
        result = []
        for k, v in d.items():
            kr = TASTE_KEY_MAP.get(k.lower())
            if kr and str(v).lower() in ("high", "medium"):  # low는 비선호로 간주
                result.append(kr)
        return result
    except:
        return []


def _parse_aroma_profile(val) -> list:
    """{"fruity":"high","minty":"low"} → "high"/"medium"인 것만 한국어 리스트로"""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        d = json.loads(val)
        result = []
        for k, v in d.items():
            kr = AROMA_KEY_MAP.get(k.lower())
            if kr and str(v).lower() in ("high", "medium"):
                result.append(kr)
        return result
    except:
        return []


def _parse_disliked_bases(val) -> list:
    """["whiskey","rum"] → ["위스키","럼"]"""
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        lst = json.loads(val)
        return [BASE_MAP[x.lower()] for x in lst if x.lower() in BASE_MAP]
    except:
        return []


def _list_overlap(pred_list, gold_list) -> bool:
    """1개 이상 겹치면 True"""
    if not pred_list or not gold_list:
        return False
    pred_set = set(str(x) for x in pred_list)
    gold_set = set(str(x) for x in gold_list)
    return bool(pred_set & gold_set)


def eval_slots():
    df = pd.read_csv("data/eval/slot_extraction_eval_v2_500.csv")

    results = {
        "party_purpose":      {"correct": 0, "total": 0},
        "current_mood":       {"correct": 0, "total": 0},
        "strength_preference":{"correct": 0, "total": 0},
        "finish_preference":  {"correct": 0, "total": 0},
        "preferred_tastes":   {"correct": 0, "total": 0},
        "preferred_aromas":   {"correct": 0, "total": 0},
        "disliked_bases":     {"correct": 0, "total": 0},
    }

    for _, row in df.iterrows():
        # 슬롯 추출 실행
        result = run_dialogue(history=[], slots={}, user_msg=row["user_text"])
        ext = result.get("extracted_slots", {})

        # ── party_purpose ──────────────────────────────────────────
        gold = _norm_single(row.get("gold_party_purpose"), PARTY_PURPOSE_MAP)
        if gold:
            results["party_purpose"]["total"] += 1
            if ext.get("party_purpose") == gold:
                results["party_purpose"]["correct"] += 1

        # ── current_mood ───────────────────────────────────────────
        gold = _norm_single(row.get("gold_current_mood"), CURRENT_MOOD_MAP)
        if gold:
            results["current_mood"]["total"] += 1
            if ext.get("current_mood") == gold:
                results["current_mood"]["correct"] += 1

        # ── strength_preference ────────────────────────────────────
        gold = _norm_single(row.get("gold_strength_preference"), STRENGTH_MAP)
        if gold:
            results["strength_preference"]["total"] += 1
            if ext.get("strength_preference") == gold:
                results["strength_preference"]["correct"] += 1

        # ── finish_preference ──────────────────────────────────────
        gold = _norm_single(row.get("gold_finish_preference"), FINISH_MAP)
        if gold:
            results["finish_preference"]["total"] += 1
            if ext.get("finish_preference") == gold:
                results["finish_preference"]["correct"] += 1

        # ── preferred_tastes (1개 이상 겹치면 정답) ─────────────────
        gold_tastes = _parse_taste_profile(row.get("gold_taste_profile"))
        if gold_tastes:
            results["preferred_tastes"]["total"] += 1
            pred_tastes = ext.get("preferred_tastes") or []
            if _list_overlap(pred_tastes, gold_tastes):
                results["preferred_tastes"]["correct"] += 1

        # ── preferred_aromas (1개 이상 겹치면 정답) ─────────────────
        gold_aromas = _parse_aroma_profile(row.get("gold_aroma_profile"))
        if gold_aromas:
            results["preferred_aromas"]["total"] += 1
            pred_aromas = ext.get("preferred_aromas") or []
            if _list_overlap(pred_aromas, gold_aromas):
                results["preferred_aromas"]["correct"] += 1

        # ── disliked_bases (1개 이상 겹치면 정답) ───────────────────
        gold_bases = _parse_disliked_bases(row.get("gold_disliked_bases"))
        if gold_bases:
            results["disliked_bases"]["total"] += 1
            pred_bases = ext.get("disliked_bases") or []
            if _list_overlap(pred_bases, gold_bases):
                results["disliked_bases"]["correct"] += 1

    print("\n[슬롯 추출 평가]")
    total_c, total_t = 0, 0
    for field, r in results.items():
        if r["total"] == 0:
            continue
        acc = r["correct"] / r["total"] * 100
        print(f"  {field:25s}: {r['correct']:3d}/{r['total']:3d} = {acc:.1f}%")
        total_c += r["correct"]
        total_t += r["total"]

    overall = total_c / total_t * 100 if total_t else 0
    print(f"\n  전체 평균 정확도: {overall:.1f}%")
    return overall


if __name__ == "__main__":
    eval_slots()
