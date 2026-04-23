"""
슬롯 추출 모델 성능 평가 스크립트

파인튜닝 전(Ollama/Gemma3:4b)과 후(QLoRA 어댑터)를 동일한 val set 으로 비교.

실행:
    # e2e CSV로 Ollama 평가 후 저장
    python scripts/eval_slot_extractor.py --backend ollama --eval_csv data/eval/e2e_recommendation_eval_v1_500.csv --save

    # e2e CSV로 어댑터 평가 후 저장
    python scripts/eval_slot_extractor.py --backend adapter --eval_csv data/eval/e2e_recommendation_eval_v1_500.csv --save

    # 비교 출력
    python scripts/eval_slot_extractor.py --compare

    # 빠른 테스트 (50개만)
    python scripts/eval_slot_extractor.py --backend ollama --eval_csv data/eval/e2e_recommendation_eval_v1_500.csv --limit 50 --save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

VAL_FILE = ROOT / "data" / "finetune" / "val.jsonl"
RESULTS_DIR = ROOT / "data" / "finetune" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 평가할 필드 ────────────────────────────────────────────────────────────────
SCALAR_FIELDS = ["party_purpose", "strength_preference", "current_mood"]
DICT_FIELDS   = ["taste_profile", "aroma_profile"]
LIST_FIELDS   = ["disliked_bases", "favorite_drinks"]


# ─────────────────────────────────────────────────────────────────────────────
# 추론 함수
# ─────────────────────────────────────────────────────────────────────────────

def _run_ollama(system_prompt: str, user_text: str) -> str:
    import requests
    from app.utils.config import LLM_MODEL, OLLAMA_BASE_URL
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "options": {"num_predict": 200, "temperature": 0.0},
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _run_adapter(system_prompt: str, user_text: str) -> str:
    from app.utils.slot_extractor import adapter_extract_slots
    return adapter_extract_slots(user_text, system_prompt=system_prompt, max_new_tokens=200)


# ─────────────────────────────────────────────────────────────────────────────
# 파싱
# ─────────────────────────────────────────────────────────────────────────────

def _parse_slots(raw: str) -> dict:
    """raw JSON 문자열 → extracted_slots dict."""
    raw = raw.strip()
    # JSON 블록 추출
    start = raw.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start : i + 1])
                    return obj.get("extracted_slots", obj)
                except json.JSONDecodeError:
                    return {}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────────────────────────────────────

def _dict_overlap(pred: dict, gold: dict) -> tuple[int, int, int]:
    """(tp, fp, fn) — 키+값 모두 일치해야 tp. 값이 중첩 dict여도 안전."""
    def items_set(d: dict) -> set:
        return {(k, json.dumps(v, sort_keys=True, ensure_ascii=False)) for k, v in d.items()}
    pred_items = items_set(pred)
    gold_items = items_set(gold)
    tp = len(pred_items & gold_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)
    return tp, fp, fn


def _list_overlap(pred: list, gold: list) -> tuple[int, int, int]:
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    return tp, len(ps) - tp, len(gs) - tp


def evaluate_sample(pred_slots: dict, gold_slots: dict) -> dict:
    """샘플 1개 평가 → 필드별 세부 결과 dict."""
    result = {}

    # 스칼라 필드
    for f in SCALAR_FIELDS:
        p = pred_slots.get(f)
        g = gold_slots.get(f)
        result[f] = {"pred": p, "gold": g, "correct": p == g}

    # 딕트 필드 (taste_profile, aroma_profile)
    for f in DICT_FIELDS:
        p = pred_slots.get(f) or {}
        g = gold_slots.get(f) or {}
        tp, fp, fn = _dict_overlap(p, g)
        result[f] = {"pred": p, "gold": g, "tp": tp, "fp": fp, "fn": fn,
                     "exact": p == g}

    # 리스트 필드
    for f in LIST_FIELDS:
        p = pred_slots.get(f) or []
        g = gold_slots.get(f) or []
        tp, fp, fn = _list_overlap(p, g)
        result[f] = {"pred": p, "gold": g, "tp": tp, "fp": fp, "fn": fn,
                     "exact": sorted(p) == sorted(g)}

    return result


def aggregate(sample_results: list[dict]) -> dict:
    """전체 샘플 결과 → 집계 지표."""
    n = len(sample_results)
    agg: dict = {}

    for f in SCALAR_FIELDS:
        hits = sum(1 for r in sample_results if r[f]["correct"])
        # gold 가 None 인 샘플(슬롯 없는 발화)은 분모에서 제외
        denom = sum(1 for r in sample_results if r[f]["gold"] is not None)
        agg[f] = {"acc": hits / denom if denom else None, "hits": hits, "denom": denom}

    for f in DICT_FIELDS + LIST_FIELDS:
        tp = sum(r[f]["tp"] for r in sample_results)
        fp = sum(r[f]["fp"] for r in sample_results)
        fn = sum(r[f]["fn"] for r in sample_results)
        exact = sum(1 for r in sample_results if r[f]["exact"])
        prec = tp / (tp + fp) if tp + fp else None
        rec  = tp / (tp + fn) if tp + fn else None
        f1   = 2 * prec * rec / (prec + rec) if prec and rec else None
        agg[f] = {"exact_match": exact / n, "precision": prec,
                  "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CSV → record 변환
# ─────────────────────────────────────────────────────────────────────────────

def load_records_from_csv(csv_path: str, limit: int | None = None) -> list[dict]:
    """e2e_recommendation_eval_v1_500.csv → {user_text, gold_slots} 리스트."""
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    records = []
    for row in rows:
        user_text = (row.get("user_text") or "").strip()
        if not user_text:
            continue

        gold: dict = {}
        # 스칼라
        for f in ["current_mood", "party_purpose", "strength_preference"]:
            v = (row.get(f"gold_{f}") or "").strip()
            if v:
                gold[f] = v
        # 딕트
        for f in ["taste_profile", "aroma_profile"]:
            raw = (row.get(f"gold_{f}") or "").strip()
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {}
            if parsed:
                gold[f] = parsed
        # 리스트
        for f in ["disliked_bases", "favorite_drinks"]:
            raw = (row.get(f"gold_{f}") or "").strip()
            try:
                parsed = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                parsed = []
            if parsed:
                gold[f] = parsed

        records.append({"user_text": user_text, "gold_slots": gold})
    return records


def load_records_from_jsonl(path: str, limit: int | None = None) -> list[dict]:
    """기존 val.jsonl → {user_text, gold_slots} 리스트."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    records = []
    for rec in rows:
        msgs = rec["messages"]
        user_text  = msgs[1]["content"]
        gold_slots = json.loads(msgs[2]["content"]).get("extracted_slots", {})
        records.append({"user_text": user_text, "gold_slots": gold_slots})
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(backend: str, eval_csv: str | None = None, limit: int | None = None) -> dict:
    from app.agents.preference_agent import _EXTRACT_SYSTEM_PROMPT

    if eval_csv:
        records = load_records_from_csv(eval_csv, limit=limit)
    else:
        records = load_records_from_jsonl(str(VAL_FILE), limit=limit)

    print(f"[{backend}] {len(records)}개 평가 시작...")

    sample_results = []
    errors = 0
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        user_text  = rec["user_text"]
        gold_slots = rec["gold_slots"]

        try:
            if backend == "ollama":
                raw = _run_ollama(_EXTRACT_SYSTEM_PROMPT, user_text)
            else:
                raw = _run_adapter(_EXTRACT_SYSTEM_PROMPT, user_text)
            pred_slots = _parse_slots(raw)
        except Exception as e:
            print(f"  [{i}] 오류: {e}")
            pred_slots = {}
            errors += 1

        sample_results.append(evaluate_sample(pred_slots, gold_slots))

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(records)} 완료 ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"완료: {elapsed:.1f}s, 오류: {errors}개")

    return {
        "backend": backend,
        "n": len(records),
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
        "aggregate": aggregate(sample_results),
        "samples": sample_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 비교 출력
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None:
        return "  —   "
    return f"{v*100:5.1f}%"


def print_comparison(before: dict, after: dict):
    ba = before["aggregate"]
    aa = after["aggregate"]

    print("\n" + "=" * 62)
    print(f"  슬롯 추출 성능 비교  (val {before['n']}개)")
    print("=" * 62)
    print(f"{'필드':<22}{'지표':<12}{'파인튜닝 전':>10}{'파인튜닝 후':>10}{'변화':>10}")
    print("-" * 62)

    def diff(a, b):
        if a is None or b is None:
            return "   —"
        d = (b - a) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    for f in SCALAR_FIELDS:
        b_acc = ba[f]["acc"]
        a_acc = aa[f]["acc"]
        print(f"{f:<22}{'정확도':<12}{_pct(b_acc)}{_pct(a_acc):>10}{diff(b_acc, a_acc):>10}")

    print()
    for f in DICT_FIELDS + LIST_FIELDS:
        b, a = ba[f], aa[f]
        print(f"{f:<22}{'exact_match':<12}{_pct(b['exact_match'])}{_pct(a['exact_match']):>10}{diff(b['exact_match'], a['exact_match']):>10}")
        print(f"{'':22}{'F1':<12}{_pct(b['f1'])}{_pct(a['f1']):>10}{diff(b['f1'], a['f1']):>10}")

    print("-" * 62)
    print(f"{'추론 시간':<22}{'초/val set':<12}{before['elapsed_s']:>9.1f}s{after['elapsed_s']:>9.1f}s")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "adapter"], default="ollama")
    p.add_argument("--eval_csv", type=str, default=None, help="평가할 CSV 경로 (e2e CSV)")
    p.add_argument("--limit", type=int, default=None, help="평가할 샘플 수 제한")
    p.add_argument("--save", action="store_true", help="결과를 JSON 파일로 저장")
    p.add_argument("--compare", action="store_true", help="저장된 두 결과를 비교 출력")
    args = p.parse_args()

    if args.compare:
        before_path = RESULTS_DIR / "result_ollama.json"
        after_path  = RESULTS_DIR / "result_adapter.json"
        if not before_path.exists() or not after_path.exists():
            print("비교하려면 먼저 두 백엔드 모두 --save 로 실행해야 합니다.")
            print(f"  없는 파일: {[f for f in [before_path, after_path] if not f.exists()]}")
            sys.exit(1)
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after  = json.loads(after_path.read_text(encoding="utf-8"))
        print_comparison(before, after)
        return

    eval_csv = str(ROOT / args.eval_csv) if args.eval_csv else None
    result = run_eval(args.backend, eval_csv=eval_csv, limit=args.limit)

    if args.save:
        out = RESULTS_DIR / f"result_{args.backend}.json"
        save_obj = {k: v for k, v in result.items() if k != "samples"}
        out.write_text(json.dumps(save_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장 → {out}")

    # 단독 실행 시 집계 출력
    agg = result["aggregate"]
    print("\n[집계 결과]")
    for f in SCALAR_FIELDS:
        print(f"  {f}: 정확도 {_pct(agg[f]['acc'])}  ({agg[f]['hits']}/{agg[f]['denom']})")
    for f in DICT_FIELDS + LIST_FIELDS:
        print(f"  {f}: exact={_pct(agg[f]['exact_match'])}  F1={_pct(agg[f]['f1'])}")


if __name__ == "__main__":
    main()
