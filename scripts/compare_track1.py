"""Track 1 비교 — 베이스라인 (raw 2B prompt) vs 어댑터 (2B+LoRA)
사용: python /tmp/compare_track1.py <baseline.json> <adapter.json>
"""
import sys
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def slot_metrics(data):
    """summary JSON 에서 슬롯 메트릭 뽑기"""
    raw = data.get("raw_summary", "") or ""
    out = {}
    for line in raw.split("\n"):
        line = line.strip()
        if "current_mood" in line and "=" in line:
            out["mood_em"] = float(line.split("=")[-1].strip().rstrip("%"))
        elif "party_purpose" in line and "=" in line and "exact match" not in line.lower():
            out["party_em"] = float(line.split("=")[-1].strip().rstrip("%"))
        elif "strength_preference" in line and "=" in line:
            out["strength_em"] = float(line.split("=")[-1].strip().rstrip("%"))
        elif "taste_profile KEY+VALUE" in line and "F1=" in line:
            out["taste_kv_f1"] = float(line.split("F1=")[1].split()[0])
        elif "aroma_profile KEY+VALUE" in line and "F1=" in line:
            out["aroma_kv_f1"] = float(line.split("F1=")[1].split()[0])
        elif "disliked_bases" in line and "F1=" in line:
            out["disliked_f1"] = float(line.split("F1=")[1].split()[0])
        elif "Hit@1" in line and "=" in line:
            out["hit1"] = float(line.split("=")[-1].strip().rstrip("%"))
        elif "Hit@3" in line and "(labeled)" in line:
            out["hit3"] = float(line.split("=")[-1].strip().rstrip("%"))
        elif "Cat" in line and "labeled" in line:
            out["cat_lab"] = float(line.split("=")[-1].strip().rstrip("%"))
    return out

if __name__ == "__main__":
    base = slot_metrics(load(sys.argv[1]))
    adp  = slot_metrics(load(sys.argv[2]))

    rows = [
        ("current_mood EM",   "mood_em"),
        ("party_purpose EM",  "party_em"),
        ("strength EM",       "strength_em"),
        ("taste KV F1",       "taste_kv_f1"),
        ("aroma KV F1",       "aroma_kv_f1"),
        ("disliked_bases F1", "disliked_f1"),
        ("Hit@1 (e2e)",       "hit1"),
        ("Hit@3 (e2e)",       "hit3"),
        ("Cat Hit@3 (lab)",   "cat_lab"),
    ]
    print(f"{'metric':<25} {'baseline':>10} {'adapter':>10} {'Δ (pp)':>10}")
    print("-" * 60)
    for label, k in rows:
        b = base.get(k, "-")
        a = adp.get(k, "-")
        d = (a - b) if isinstance(b, float) and isinstance(a, float) else "-"
        d_str = f"{d:+.1f}" if isinstance(d, float) else "-"
        b_str = f"{b:.1f}" if isinstance(b, float) else "-"
        a_str = f"{a:.1f}" if isinstance(a, float) else "-"
        print(f"{label:<25} {b_str:>10} {a_str:>10} {d_str:>10}")
