"""
eval CSV(500행) → QLoRA 학습용 JSONL 변환

현재 스키마 기준:
  포함: party_purpose(매핑), taste_profile, aroma_profile,
        strength_preference, disliked_bases, favorite_drinks
  제외: current_mood (CSV값이 공간 분위기 태그 — 코드 스키마와 의미 불일치)
        finish_preference (현재 스키마에 없는 슬롯)

출력: data/finetune/train.jsonl, data/finetune/val.jsonl (90:10 split)
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

# ── 현재 스키마 기준 party_purpose 매핑 ──────────────────────────────────────
_PURPOSE_MAP = {
    "celebration": "celebration",
    "date": "date",
    "business": "business",
    "solo": "solo",
    "hangout": "hangout",
    "casual_hangout": "hangout",
    "social": "hangout",
    "special_day": "celebration",
    "party": "hangout",
    "after_work": "hangout",
}

# ── 시스템 프롬프트 (preference_agent._EXTRACT_SYSTEM_PROMPT 와 동일) ─────────
SYSTEM_PROMPT = """너는 한국어 칵테일 취향 슬롯 추출기다. 이번 발화에 명시된 정보만 추출해 JSON 한 객체만 출력. 설명 금지.

[스키마] {"extracted_slots":{...}}
- current_mood: "good"|"soso"|"bad"|null
- party_purpose: "celebration"|"date"|"business"|"solo"|"hangout"|null
- taste_profile: {"sweet"|"sour"|"bitter"|"body"|"creamy"|"freshness": "zero"|"low"|"medium"|"high"}
- aroma_profile: {"woody"|"minty"|"fruity"|"citrus"|"floral"|"coffee"|"herbal": "zero"|"low"|"medium"|"high"}
- strength_preference: "zero"|"light"|"medium"|"strong"|null
- disliked_bases: ["whiskey"|"gin"|"rum"|"vodka"|"tequila"]
- favorite_drinks: [문자열]

[원칙]
1. 이번 발화에 없는 축 추가 금지. 질문/"모르겠다/딱히" → {"extracted_slots":{}}.
2. 축 분류 (혼동 주의): freshness/creamy/body → taste_profile. citrus/fruity → aroma_profile.
   모든 강도는 taste_profile/aroma_profile 안에 nesting. top-level에 직접 넣으면 드롭.
   taste/aroma 값은 "strong" 금지 — strength_preference 전용. "강하게" → "high".
3. 선호 극성: "싫어/빼줘/질색" → zero. 나머지 언급은 강도(low/medium/high).
4. CORRECTION ("X한 적 없어/X 말 안 했어") → 해당 축만 null. "추천해줘/모르겠다" 는 → {}.
5. CONFIRM ("응/ㅇㅇ/맞아"): 봇이 구체 강도값 제안 시 그 값 추출. 양자택일 후 "응" → {}.
6. current_mood = 사용자 본인 기분("기분 좋아/우울")만. 공간 분위기 금지.
7. 한 발화에 여러 축 → 모두 추출.

[강도] 확/엄청/강하게→high, 적당히/중간→medium, 살짝/약간/은은→low, 싫어/빼줘→zero
[키워드] taste: 단→sweet, 신→sour, 쓴→bitter, 바디/묵직→body, 크리미/우유/밀크→creamy, 청량/상큼/시원→freshness
         aroma: 우디/나무→woody, 민트→minty, 과일/프루티/파인애플/망고→fruity, 시트러스/레몬/자몽/라임→citrus, 꽃→floral, 커피→coffee, 허브→herbal
         purpose: 혼자/혼술→solo, 회식/거래처→business, 생일/기념/축하→celebration, 데이트/썸→date, 친구/모임→hangout
         mood: 좋아/신나→good, 우울/힘들→bad | strength: 무알콜→zero, 약하게→light, 보통→medium, 강하게→strong

예시:
"상큼 짱 좋아, 과일향도, 묵직 중간"→{"extracted_slots":{"taste_profile":{"freshness":"high","body":"medium"},"aroma_profile":{"fruity":"high"}}}
"위스키 싫어"→{"extracted_slots":{"disliked_bases":["whiskey"]}}
"단맛 좋다고 안 했는데?"→{"extracted_slots":{"taste_profile":{"sweet":null}}}
"응" (봇: "청량high·크리미low로?")→{"extracted_slots":{"taste_profile":{"freshness":"high","creamy":"low"}}}
"친구 생일파티야"→{"extracted_slots":{"party_purpose":"celebration"}}""".strip()


def _parse_json_field(raw: str) -> dict | list | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _build_gold_slots(row: dict) -> dict:
    slots: dict = {}

    # party_purpose — 매핑 적용
    purpose_raw = (row.get("gold_party_purpose") or "").strip()
    if purpose_raw:
        mapped = _PURPOSE_MAP.get(purpose_raw)
        if mapped:
            slots["party_purpose"] = mapped

    # taste_profile
    taste = _parse_json_field(row.get("gold_taste_profile", ""))
    if taste:
        slots["taste_profile"] = taste

    # aroma_profile
    aroma = _parse_json_field(row.get("gold_aroma_profile", ""))
    if aroma:
        slots["aroma_profile"] = aroma

    # strength_preference
    strength = (row.get("gold_strength_preference") or "").strip()
    if strength:
        slots["strength_preference"] = strength

    # disliked_bases
    dislikes = _parse_json_field(row.get("gold_disliked_bases", ""))
    if dislikes:
        slots["disliked_bases"] = dislikes

    # favorite_drinks
    favs = _parse_json_field(row.get("gold_favorite_drinks", ""))
    if favs:
        slots["favorite_drinks"] = favs

    return slots


def build_messages(user_text: str, gold_slots: dict) -> list[dict]:
    """SFTTrainer messages 형식 (system / user / assistant 3-turn)."""
    assistant_output = json.dumps({"extracted_slots": gold_slots}, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_output},
    ]


def main():
    src = Path(__file__).parent.parent / "data" / "eval" / "slot_extraction_eval_v2_500.csv"
    out_dir = Path(__file__).parent.parent / "data" / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    random.seed(42)
    random.shuffle(rows)

    split = int(len(rows) * 0.9)
    train_rows, val_rows = rows[:split], rows[split:]

    skipped = 0
    stats = {"train": 0, "val": 0}

    for split_name, split_rows in [("train", train_rows), ("val", val_rows)]:
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in split_rows:
                user_text = (row.get("user_text") or "").strip()
                if not user_text:
                    skipped += 1
                    continue
                gold = _build_gold_slots(row)
                msgs = build_messages(user_text, gold)
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                stats[split_name] += 1

    print(f"train: {stats['train']}개  val: {stats['val']}개  skipped: {skipped}개")
    print(f"출력 → {out_dir}/train.jsonl, val.jsonl")

    # 샘플 확인
    with open(out_dir / "train.jsonl", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("\n[샘플 assistant 출력]")
    print(sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
