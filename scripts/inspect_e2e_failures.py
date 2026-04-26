"""per-item CSV 에서 실패 패턴 분석.

eval_e2e_recommendation.py 가 저장한 per-item CSV 를 읽어서:
  - labeled miss (has_exact_gold=1, hit_3=0) 케이스의 gold vs top3 비교
  - gold 가 retrieval 에서 아예 누락됐는지 / 뒤쪽 순위로 밀렸는지 구분 (source 컬럼)
  - category miss 도 별도로 덤프

사용:
  python scripts/inspect_e2e_failures.py eval_results/cases/e2e/gemma_e2e_oracle_smoke20_XXXX.csv
  python scripts/inspect_e2e_failures.py eval_results/cases/e2e/gemma_e2e_oracle_smoke20_XXXX.csv --only cat_miss
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _parse_list(s):
    if pd.isna(s) or not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _parse_dict(s):
    if pd.isna(s) or not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _fmt_slots(row) -> str:
    parts = []
    for k in ("current_mood_gold", "party_purpose_gold", "strength_preference_gold"):
        v = row.get(k)
        if pd.notna(v) and v:
            parts.append(f"{k.replace('_gold','')}={v}")
    t = _parse_dict(row.get("taste_gold"))
    a = _parse_dict(row.get("aroma_gold"))
    if t:
        parts.append(f"taste={t}")
    if a:
        parts.append(f"aroma={a}")
    b = _parse_list(row.get("bases_gold"))
    if b:
        parts.append(f"bases!={b}")
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument(
        "--only",
        choices=["labeled_miss", "hit1_miss", "cat_miss", "no_candidate", "all"],
        default="all",
        help="hit1_miss = 정답이 top3 에 있지만 #1 이 아닌 케이스 (랭킹 순위 문제)",
    )
    ap.add_argument("--max", type=int, default=None, help="케이스 최대 출력 개수")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[err] not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    n = len(df)

    # 분류
    def _i(v):
        try:
            return int(v)
        except Exception:
            return None

    df["has_exact_gold_i"] = df["has_exact_gold"].apply(_i)
    df["hit_1_i"] = df["hit_1"].apply(_i)
    df["hit_3_i"] = df["hit_3"].apply(_i)
    df["cat_hit_3_i"] = df["cat_hit_3"].apply(_i)
    df["no_candidate_i"] = df["no_candidate"].apply(_i)

    labeled = df[df["has_exact_gold_i"] == 1]
    labeled_miss = labeled[labeled["hit_3_i"] == 0]
    # Hit@1 only miss = 정답이 top3 안에는 있지만 #1 이 아닌 케이스 (순위 문제만 격리).
    hit1_miss = labeled[(labeled["hit_3_i"] == 1) & (labeled["hit_1_i"] == 0)]
    cat_miss = df[(df["cat_hit_3_i"] == 0) & df["cat_hit_3"].notna() & (df["cat_hit_3"] != "")]
    no_cand = df[df["no_candidate_i"] == 1]

    print("=" * 70)
    print(f"per-item CSV: {args.csv}")
    print(f"total={n}  labeled={len(labeled)}  labeled_miss={len(labeled_miss)}  "
          f"hit1_miss={len(hit1_miss)}  cat_miss={len(cat_miss)}  no_candidate={len(no_cand)}")
    print("=" * 70)

    def _dump(title, rows, max_n=None):
        if rows.empty:
            return
        print(f"\n### {title}  (n={len(rows)})")
        for i, (_, r) in enumerate(rows.iterrows()):
            if max_n and i >= max_n:
                print(f"  ... +{len(rows)-max_n} more")
                break
            case_id = r.get("case_id", "?")
            mode = r.get("mode", "?")
            gold_ck = _parse_list(r.get("gold_cocktails"))
            gold_cat = _parse_list(r.get("gold_categories"))
            top_names = _parse_list(r.get("top3_names"))
            top_cats = _parse_list(r.get("top3_categories"))
            top_src = _parse_list(r.get("top3_sources"))
            score_names = _parse_list(r.get("score_top3_names"))
            retrieved_names = _parse_list(r.get("retrieved_names"))
            survivor_names = _parse_list(r.get("survivor_names"))
            user_text = str(r.get("user_text", ""))[:80]

            print(f"\n  [case {case_id}] mode={mode}")
            print(f"    user_text : {user_text}")
            print(f"    gold_slots: {_fmt_slots(r)}")
            print(f"    gold_ck   : {gold_ck}")
            print(f"    gold_cat  : {gold_cat}")
            print(f"    retrieved?: gold={r.get('gold_in_retrieved', '')}  "
                  f"survivor={r.get('gold_in_survivors', '')}  "
                  f"score_top3={r.get('gold_in_score_top3', '')}  "
                  f"final_ranker={r.get('final_ranker', '')}")
            print(f"    retrieved : {retrieved_names[:5]}{'...' if len(retrieved_names) > 5 else ''}")
            print(f"    survivors : {survivor_names[:5]}{'...' if len(survivor_names) > 5 else ''}")
            print(f"    score_top3: {score_names}")
            print(f"    top3_name : {top_names}")
            print(f"    top3_cat  : {top_cats}")
            print(f"    top3_src  : {top_src}")

            # gold 가 retrieval 에서 빠졌는지 vs top3 에는 들어왔지만 1위가 아닌지
            in_top3 = any(n in gold_ck for n in top_names)
            if gold_ck:
                if in_top3:
                    rank = next((i+1 for i, n in enumerate(top_names) if n in gold_ck), None)
                    print(f"    => gold in top3 (rank={rank}) — rerank 순위 문제")
                else:
                    print(f"    => gold NOT in top3 — retrieval 에서 누락 or 하드필터 드랍")

    if args.only in ("labeled_miss", "all"):
        _dump("Labeled miss (hit_3=0) — 정답이 top3 에 아예 없음", labeled_miss, args.max)
    if args.only in ("hit1_miss", "all"):
        _dump("Hit@1 only miss (top3 안에 있지만 #1 아님) — 순위 문제", hit1_miss, args.max)
    if args.only in ("cat_miss", "all"):
        _dump("Category miss (cat_hit_3=0)", cat_miss, args.max)
    if args.only in ("no_candidate", "all"):
        _dump("No candidate (하드필터로 전멸)", no_cand, args.max)


if __name__ == "__main__":
    main()
