"""두 모델의 대화 품질 JSON 결과를 side-by-side 비교.

사용:
  python scripts/compare_models.py eval_results/conversation/qwen_*.json eval_results/conversation/exaone_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _print_metrics_table(a: dict, b: dict) -> None:
    ma, mb = a["metrics"], b["metrics"]
    keys = list(ma.keys())
    print(f"\n{'지표':30s} {a['tag']:>14s} {b['tag']:>14s}")
    print("-" * 62)
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        print(f"{k:30s} {str(va):>14s} {str(vb):>14s}")


def _print_scenario_diff(a: dict, b: dict) -> None:
    runs_a = {r["id"]: r for r in a["runs"]}
    runs_b = {r["id"]: r for r in b["runs"]}
    for sid in runs_a:
        if sid not in runs_b:
            continue
        print(f"\n=== {sid} ===")
        ta, tb = runs_a[sid]["turns"], runs_b[sid]["turns"]
        for i in range(max(len(ta), len(tb))):
            if i < len(ta):
                t = ta[i]
                print(f"  [{a['tag']} T{t['turn']}] 유저: {t['user']}")
                print(f"     intent={t['user_intent']}(gold={t['gold_intent']}) "
                      f"ko={t['korean_only']} ext_ok={t['extract_match']}")
                print(f"     reply: {t['reply'][:120]}")
            if i < len(tb):
                t = tb[i]
                print(f"  [{b['tag']} T{t['turn']}] 유저: {t['user']}")
                print(f"     intent={t['user_intent']}(gold={t['gold_intent']}) "
                      f"ko={t['korean_only']} ext_ok={t['extract_match']}")
                print(f"     reply: {t['reply'][:120]}")


def main():
    if len(sys.argv) != 3:
        print("사용: python scripts/compare_models.py <a.json> <b.json>")
        sys.exit(1)
    a = _load(sys.argv[1])
    b = _load(sys.argv[2])
    print(f"모델 A: {a.get('model')}  (tag={a['tag']})")
    print(f"모델 B: {b.get('model')}  (tag={b['tag']})")
    _print_metrics_table(a, b)
    _print_scenario_diff(a, b)


if __name__ == "__main__":
    main()
