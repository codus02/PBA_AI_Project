"""슬롯 추출 평가 — 신규 스키마 (영문 enum + intensity dict).

data/eval/slot_extraction_eval_v2_500.csv의 gold 라벨과 analyze_user_turn 출력 비교.

메트릭:
  - 스칼라 enum (current_mood, party_purpose, strength, finish): exact match
  - intensity dict (taste_profile, aroma_profile):
      * key_presence: gold 키가 pred에 있는지 (F1)
      * exact_kv: 키 일치 && 값 일치 (F1)
  - list enum (disliked_bases): set F1
  - favorite_drinks: 존재 여부(binary) — free-form이라 문자열 정확 비교는 무의미
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import analyze_user_turn

CSV_PATH = Path("data/eval/slot_extraction_eval_v2_500.csv")


# ============================================================
# gold 파싱 유틸
# ============================================================

def _s(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s or None


def _parse_json_list(val) -> list:
    s = _s(val)
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _parse_json_dict(val) -> dict:
    s = _s(val)
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


# ============================================================
# 메트릭
# ============================================================

class Accum:
    __slots__ = ("correct", "total")

    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, ok: bool):
        self.total += 1
        if ok:
            self.correct += 1

    def pct(self) -> float:
        return (self.correct / self.total * 100) if self.total else 0.0


class PRF:
    __slots__ = ("tp", "fp", "fn")

    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, pred: set, gold: set):
        self.tp += len(pred & gold)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)

    def f1(self) -> tuple[float, float, float]:
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        return p * 100, r * 100, f * 100


# ============================================================
# 평가 루프
# ============================================================

def run_eval(limit: int | None = None, verbose: bool = False):
    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.head(limit)

    enum_acc = {k: Accum() for k in ["current_mood", "party_purpose", "strength_preference", "finish_preference"]}

    taste_key_prf = PRF()
    taste_kv_prf = PRF()
    aroma_key_prf = PRF()
    aroma_kv_prf = PRF()
    bases_prf = PRF()
    favs_bin = Accum()  # gold에 값이 있을 때 pred도 비어있지 않으면 correct

    t0 = time.perf_counter()
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        result = analyze_user_turn(history=[], slots={}, user_msg=row.user_text)
        pred = result["extracted_slots"]

        # scalar enums
        for csv_key, slot_key in [
            ("gold_current_mood", "current_mood"),
            ("gold_party_purpose", "party_purpose"),
            ("gold_strength_preference", "strength_preference"),
            ("gold_finish_preference", "finish_preference"),
        ]:
            g = _s(getattr(row, csv_key))
            if g is None:
                continue  # gold 비어 있으면 평가 대상 아님
            p = pred.get(slot_key)
            enum_acc[slot_key].add(p == g)

        # taste_profile
        g_taste = _parse_json_dict(row.gold_taste_profile)
        p_taste = pred.get("taste_profile") or {}
        if g_taste or p_taste:
            taste_key_prf.add(set(p_taste.keys()), set(g_taste.keys()))
            taste_kv_prf.add(
                {(k, v) for k, v in p_taste.items()},
                {(k, v) for k, v in g_taste.items()},
            )

        # aroma_profile
        g_aroma = _parse_json_dict(row.gold_aroma_profile)
        p_aroma = pred.get("aroma_profile") or {}
        if g_aroma or p_aroma:
            aroma_key_prf.add(set(p_aroma.keys()), set(g_aroma.keys()))
            aroma_kv_prf.add(
                {(k, v) for k, v in p_aroma.items()},
                {(k, v) for k, v in g_aroma.items()},
            )

        # disliked_bases
        g_bases = set(_parse_json_list(row.gold_disliked_bases))
        p_bases = set(pred.get("disliked_bases") or [])
        if g_bases or p_bases:
            bases_prf.add(p_bases, g_bases)

        # favorite_drinks: binary presence
        g_favs = _parse_json_list(row.gold_favorite_drinks)
        # gold_favorite_drinks is sometimes free string, not JSON — handle both
        if not g_favs:
            raw = _s(row.gold_favorite_drinks)
            g_favs = [raw] if raw else []
        if g_favs:
            p_favs = pred.get("favorite_drinks") or []
            favs_bin.add(bool(p_favs))

        if verbose and i <= 10:
            print(f"[{i}/{n}] user={row.user_text[:60]}...")
            print(f"  pred: {json.dumps(pred, ensure_ascii=False)}")
        elif i % 25 == 0:
            dt = time.perf_counter() - t0
            print(f"  progress {i}/{n}  elapsed={dt:.1f}s")

    total_time = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"SLOT EXTRACTION EVAL — {n} cases ({total_time:.1f}s, {total_time/max(n,1):.2f}s/case)")
    print("=" * 60)

    print("\n[Scalar enums — exact match]")
    for k, acc in enum_acc.items():
        print(f"  {k:22s}: {acc.correct:4d}/{acc.total:4d} = {acc.pct():5.1f}%")

    def _pr(label, prf: PRF):
        p, r, f = prf.f1()
        print(f"  {label:32s}: P={p:5.1f}  R={r:5.1f}  F1={f:5.1f}  (tp={prf.tp} fp={prf.fp} fn={prf.fn})")

    print("\n[Intensity dicts]")
    _pr("taste_profile KEY presence", taste_key_prf)
    _pr("taste_profile KEY+VALUE exact", taste_kv_prf)
    _pr("aroma_profile KEY presence", aroma_key_prf)
    _pr("aroma_profile KEY+VALUE exact", aroma_kv_prf)

    print("\n[List enum]")
    _pr("disliked_bases", bases_prf)

    print("\n[Favorite drinks — binary presence]")
    print(f"  favorite_drinks detected : {favs_bin.correct:4d}/{favs_bin.total:4d} = {favs_bin.pct():5.1f}%")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="처음 N건만 평가")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    run_eval(limit=args.limit, verbose=args.verbose)
