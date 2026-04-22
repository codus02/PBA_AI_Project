"""대화 품질 자동 평가 — 모델(Qwen/EXAONE 등) 간 비교용 (LLM-agnostic).

스크립트된 시나리오의 고정 유저 발화를 analyze_user_turn 에 순차 주입하여
매 턴 reply/extracted_slots/action/user_intent 를 기록하고 자동 지표를 계산한다.

자동 지표:
- korean_only_rate : reply 에 한자/영어/일본어 섞이지 않은 비율
- json_parse_rate  : source=llm (정상 JSON 파싱) 비율
- intent_match     : user_intent 가 시나리오 gold 와 일치한 비율
- extract_match    : extracted_slots 가 gold 와 일치한 턴 비율 (subset 매칭)
- repeat_rate      : 이번 reply 의 질문이 직전 LLM 질문과 유사한 비율
- avg_recommend_turn : RECOMMEND 로 넘어간 평균 턴 수 (넘어간 시나리오만)

사용:
  python scripts/eval_conversation.py --tag qwen
  python scripts/eval_conversation.py --tag exaone --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 모델 바꿀 때는 import 전에 env 덮어써야 LLM_MODEL 이 읽힘
def _maybe_override_model(model_id: str | None) -> None:
    if model_id:
        os.environ["LLM_MODEL"] = model_id


SCENARIOS: list[dict] = [
    {
        "id": "first_celebration_sweet",
        "initial": {
            "familiarity_tag": "처음",
            "strength_tag": "중간",
            "taste_tags_json": ["단맛", "청량함"],
            "aroma_tags_json": ["과일향"],
        },
        "turns": [
            {"user": "친구 생일파티야",
             "gold_intent": "SLOT",
             "gold_extract": {"party_purpose": "celebration"}},
            {"user": "칵테일이 따뜻하다는게 뭔 소리야?",
             "gold_intent": "QUESTION",
             "gold_extract": {}},
            {"user": "단맛은 확 달달한 게 좋아",
             "gold_intent": "SLOT",
             "gold_extract": {"taste_profile": {"sweet": "high"}}},
            {"user": "알아서 골라줘",
             "gold_intent": "STOP"},
        ],
    },
    {
        "id": "first_unknown_fallback",
        "initial": {
            "familiarity_tag": "처음",
            "strength_tag": "약함",
            "taste_tags_json": ["신맛"],
            "aroma_tags_json": ["시트러스향"],
        },
        "turns": [
            {"user": "혼자 집에서 마시려고", "gold_intent": "SLOT",
             "gold_extract": {"party_purpose": "solo"}},
            {"user": "잘 모르겠는데", "gold_intent": "UNKNOWN"},
            {"user": "평소엔 레몬에이드 자주 마셔", "gold_intent": "SLOT"},
        ],
    },
    {
        "id": "regular_direct",
        "initial": {
            "familiarity_tag": "자주",
            "strength_tag": "강함",
            "taste_tags_json": ["쓴맛"],
            "aroma_tags_json": ["우디향"],
        },
        "turns": [
            {"user": "오늘 거래처 미팅 끝나고 한잔", "gold_intent": "SLOT",
             "gold_extract": {"party_purpose": "business"}},
            {"user": "진 베이스로 드라이하게", "gold_intent": "SLOT"},
            {"user": "추천해줘", "gold_intent": "STOP"},
        ],
    },
    {
        "id": "correction_mood",
        "initial": {
            "familiarity_tag": "가끔",
            "strength_tag": "중간",
            "taste_tags_json": ["단맛"],
            "aroma_tags_json": ["꽃향"],
        },
        "turns": [
            {"user": "그냥 하루 마무리로 한 잔", "gold_intent": "SLOT"},
            # LLM 이 mood=good 으로 넘겨짚을 가능성 테스트
            {"user": "나 기분 좋다는 말 안 했는데", "gold_intent": "CORRECTION",
             "gold_extract": {"current_mood": None}},
            {"user": "피곤해서 부드러운 게 땡겨", "gold_intent": "SLOT"},
        ],
    },
    {
        "id": "intensity_pending",
        "initial": {
            "familiarity_tag": "가끔",
            "strength_tag": "중간",
            "taste_tags_json": ["단맛", "신맛"],
            "aroma_tags_json": ["민트향", "우디향"],
        },
        "turns": [
            {"user": "친구들이랑 집들이", "gold_intent": "SLOT"},
            # 초기 medium 강도에 대해 LLM 이 확/은은 질문 해야함
            {"user": "신맛은 은은하게, 단맛은 확 가는 게 좋아", "gold_intent": "SLOT",
             "gold_extract": {"taste_profile": {"sour": "low", "sweet": "high"}}},
            {"user": "이제 추천해줘", "gold_intent": "STOP"},
        ],
    },
]


_HANJA_RE = re.compile(r"[\u3400-\u9FFF]")  # CJK Unified Ideographs
_ENG_RE = re.compile(r"[A-Za-z]{3,}")  # 3+ alpha in a row (이름 외 누수)
_JAPANESE_RE = re.compile(r"[\u3040-\u30FF]")


def _is_korean_only(text: str) -> bool:
    if not text:
        return True
    if _HANJA_RE.search(text):
        return False
    if _JAPANESE_RE.search(text):
        return False
    # 이름 외 영단어 3글자 이상 연속 — 예외 허용 리스트
    allow = {"Mint", "Julep", "Gin", "Rum", "Tonic"}
    for m in _ENG_RE.finditer(text):
        if m.group(0) not in allow:
            return False
    return True


def _subset_match(gold: dict, got: dict) -> bool:
    """gold 의 모든 키/값이 got 에 있으면 True. dict 값은 subset, None 은 정확 일치."""
    for k, v in (gold or {}).items():
        if k not in got:
            return False
        gv = got[k]
        if v is None:
            if gv is not None:
                return False
        elif isinstance(v, dict):
            if not isinstance(gv, dict):
                return False
            for sk, sv in v.items():
                if gv.get(sk) != sv:
                    return False
        elif isinstance(v, list):
            if not isinstance(gv, list):
                return False
            if not set(v).issubset(set(gv)):
                return False
        else:
            if gv != v:
                return False
    return True


def _question_similarity(prev: str, curr: str) -> float:
    """간단 토큰 자카드 유사도."""
    if not prev or not curr:
        return 0.0
    a = set(re.findall(r"[가-힣A-Za-z]+", prev))
    b = set(re.findall(r"[가-힣A-Za-z]+", curr))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_scenario(scenario: dict) -> dict:
    from app.agents.preference_agent import (
        analyze_user_turn,
        merge_slots,
        _seed_slots_from_initial_tags,
    )

    tag_row = SimpleNamespace(**scenario["initial"])
    slots = _seed_slots_from_initial_tags(tag_row)
    familiarity = tag_row.familiarity_tag

    history: list[dict] = []
    turn_logs: list[dict] = []
    recommend_turn: int | None = None
    prev_llm_reply = ""

    for i, step in enumerate(scenario["turns"], 1):
        user_msg = step["user"]
        result = analyze_user_turn(
            history=history,
            slots=slots,
            user_msg=user_msg,
            familiarity=familiarity,
            user_turn_count=i,
        )
        extracted = result["extracted_slots"]
        slots = merge_slots(slots, extracted)
        history.append({"speaker_role": "USER", "utterance_text": user_msg})
        reply = result["reply"]
        history.append({"speaker_role": "LLM", "utterance_text": reply})

        sim = _question_similarity(prev_llm_reply, reply)
        korean_only = _is_korean_only(reply)
        intent_ok = (
            ("gold_intent" not in step)
            or result.get("user_intent") == step["gold_intent"]
        )
        extract_ok = _subset_match(step.get("gold_extract") or {}, extracted)

        turn_logs.append({
            "turn": i,
            "user": user_msg,
            "reply": reply,
            "action": result["action"],
            "user_intent": result.get("user_intent"),
            "extracted": extracted,
            "slots_after": dict(slots),
            "source": result["source"],
            "korean_only": korean_only,
            "intent_match": intent_ok,
            "extract_match": extract_ok,
            "repeat_sim": round(sim, 3),
            "gold_intent": step.get("gold_intent"),
            "gold_extract": step.get("gold_extract"),
        })

        if result["action"] == "RECOMMEND" and recommend_turn is None:
            recommend_turn = i

        prev_llm_reply = reply

    return {
        "id": scenario["id"],
        "turns": turn_logs,
        "recommend_turn": recommend_turn,
        "final_slots": slots,
    }


def aggregate(runs: list[dict]) -> dict:
    all_turns = [t for r in runs for t in r["turns"]]
    n = max(len(all_turns), 1)
    ko = sum(1 for t in all_turns if t["korean_only"])
    js = sum(1 for t in all_turns if t["source"] == "llm")
    im = sum(1 for t in all_turns if t["intent_match"])
    em = sum(1 for t in all_turns if t["extract_match"])
    rp = sum(1 for t in all_turns if t["repeat_sim"] >= 0.6)
    rec_turns = [r["recommend_turn"] for r in runs if r["recommend_turn"] is not None]
    return {
        "n_turns": n,
        "korean_only_rate": round(ko / n * 100, 1),
        "json_parse_rate": round(js / n * 100, 1),
        "intent_match_rate": round(im / n * 100, 1),
        "extract_match_rate": round(em / n * 100, 1),
        "repeat_rate": round(rp / n * 100, 1),
        "avg_recommend_turn": round(sum(rec_turns) / len(rec_turns), 2) if rec_turns else None,
        "scenarios_reached_recommend": f"{len(rec_turns)}/{len(runs)}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="결과 파일 접미사 (예: qwen, exaone)")
    ap.add_argument("--model", default=None, help="HF model id (생략 시 env LLM_MODEL 사용)")
    ap.add_argument("--out-dir", default="eval_results/conversation")
    args = ap.parse_args()

    _maybe_override_model(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[대화 품질 평가] model={os.getenv('LLM_MODEL')}  tag={args.tag}")
    print(f"시나리오 {len(SCENARIOS)}개")

    runs: list[dict] = []
    for sc in SCENARIOS:
        print(f"  → {sc['id']}")
        runs.append(run_scenario(sc))

    metrics = aggregate(runs)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = {
        "tag": args.tag,
        "model": os.getenv("LLM_MODEL"),
        "timestamp": stamp,
        "metrics": metrics,
        "runs": runs,
    }
    path = out_dir / f"{args.tag}_{stamp}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    txt_path = out_dir / f"{args.tag}_{stamp}.txt"
    txt_path.write_text(_format_summary_txt(out))

    print(f"\n저장: {path}")
    print(f"      {txt_path}")
    print("\n[지표]")
    for k, v in metrics.items():
        print(f"  {k:28s} : {v}")


def _format_summary_txt(out: dict) -> str:
    lines: list[str] = []
    lines.append(f"[대화 품질 평가] tag={out['tag']}  model={out['model']}  {out['timestamp']}")
    lines.append("")
    lines.append("[지표]")
    for k, v in out["metrics"].items():
        lines.append(f"  {k:28s} : {v}")
    lines.append("")
    for r in out["runs"]:
        lines.append(f"=== {r['id']}  (recommend_turn={r['recommend_turn']}) ===")
        for t in r["turns"]:
            flags = []
            if not t["korean_only"]:
                flags.append("非한국어")
            if not t["intent_match"]:
                flags.append("intent✗")
            if not t["extract_match"]:
                flags.append("extract✗")
            if t["repeat_sim"] >= 0.6:
                flags.append("반복")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"  T{t['turn']} {t['user_intent']}(gold={t['gold_intent']}) "
                f"action={t['action']}{flag_str}"
            )
            lines.append(f"    유저: {t['user']}")
            lines.append(f"    reply: {t['reply']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
