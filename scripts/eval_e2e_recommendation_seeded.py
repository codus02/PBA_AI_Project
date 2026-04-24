"""Seeded E2E 평가 — initial tags + user_text refinement → 추천.

현재 plain e2e eval은 `history=[]`, `slots={}` 로 시작해 실서비스보다 더 빡세게 측정한다.
이 스크립트는 실제 initial tag seed 를 먼저 넣고, 이번 턴 대화에서 refinement 한 뒤
최종 슬롯 상태로 추천 성능을 측정한다.

중요:
- 기본 모드는 CSV 안의 실제 initial tag 컬럼을 seed 로 사용한다.
- eval CSV 에 태그 컬럼이 없는데도 seed 를 넣고 싶다면, 분석용으로만
  `--seed-source gold_pseudo` 를 명시해야 한다.
- `gold_pseudo` 는 정답 누수가 있으므로 공식 성능 숫자에 쓰면 안 된다.

출력:
  cases:    eval_results/cases/e2e/{model}_e2e_{tag}_{stamp}.csv
  summary:  eval_results/summary/e2e/{model}_e2e_{tag}_{stamp}.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import (
    INTENSITY_PENDING,
    _seed_slots_from_initial_tags,
    analyze_user_turn,
    merge_slots,
)
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids
from app.db.database import SessionLocal
from scripts._eval_save import save_eval_result, short_model_name
from scripts.eval_e2e_recommendation import (
    CSV_PATH,
    Accum,
    MockVector,
    PRF,
    _llm_top3,
    _parse_json_dict,
    _parse_json_list,
    _s,
)


def _parse_tag_list(val) -> list[str]:
    s = _s(val)
    if not s:
        return []
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return [item.strip() for item in s.split(",") if item.strip()]


def _gold_to_seed_slots(row) -> dict:
    """gold 슬롯 → pseudo initial tag seed (분석용 전용)."""
    seed: dict = {}

    strength = _s(getattr(row, "gold_strength_preference", None))
    if strength:
        seed["strength_preference"] = strength

    taste_seed: dict[str, str] = {}
    for key, intensity in _parse_json_dict(getattr(row, "gold_taste_profile", None)).items():
        if intensity in ("low", "medium", "high"):
            taste_seed[key] = INTENSITY_PENDING
    if taste_seed:
        seed["taste_profile"] = taste_seed

    aroma_seed: dict[str, str] = {}
    for key, intensity in _parse_json_dict(getattr(row, "gold_aroma_profile", None)).items():
        if intensity in ("low", "medium", "high"):
            aroma_seed[key] = INTENSITY_PENDING
    if aroma_seed:
        seed["aroma_profile"] = aroma_seed

    return seed


def _row_to_column_seed_slots(row) -> dict:
    """실제 initial tag 컬럼에서 service seed 를 복원."""
    tag_row = SimpleNamespace(
        strength_tag=(
            _s(getattr(row, "init_strength_tag", None))
            or _s(getattr(row, "strength_tag", None))
        ),
        taste_tags_json=(
            _parse_tag_list(getattr(row, "init_taste_tags_json", None))
            or _parse_tag_list(getattr(row, "taste_tags_json", None))
        ),
        aroma_tags_json=(
            _parse_tag_list(getattr(row, "init_aroma_tags_json", None))
            or _parse_tag_list(getattr(row, "aroma_tags_json", None))
        ),
    )
    return _seed_slots_from_initial_tags(tag_row)


def _resolve_seed_strategy(df: pd.DataFrame, seed_source: str) -> str:
    if seed_source == "gold_pseudo":
        return seed_source

    tag_cols = {
        "init_strength_tag",
        "strength_tag",
        "init_taste_tags_json",
        "taste_tags_json",
        "init_aroma_tags_json",
        "aroma_tags_json",
    }
    if not (tag_cols & set(df.columns)):
        raise SystemExit(
            "seeded eval requires actual initial tag columns, but none were found in the CSV.\n"
            "Expected one of: init_strength_tag/strength_tag, "
            "init_taste_tags_json/taste_tags_json, init_aroma_tags_json/aroma_tags_json.\n"
            "If you really want a leakage-heavy ablation, rerun with --seed-source gold_pseudo."
        )
    return "columns"


def _seeded_profile(seed_slots: dict, extracted: dict) -> dict:
    merged = merge_slots(seed_slots, extracted or {})
    return {"merged_slots": merged, "vector": MockVector(), "space": None}


def run_eval(limit: int | None = None, tag: str | None = None, seed_source: str = "columns"):
    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.head(limit)
    resolved_seed_source = _resolve_seed_strategy(df, seed_source)
    n = len(df)

    enum_acc = {k: Accum() for k in ["current_mood", "party_purpose", "strength_preference"]}
    taste_key_prf = PRF()
    taste_kv_prf = PRF()
    aroma_key_prf = PRF()
    aroma_kv_prf = PRF()
    bases_prf = PRF()

    hit_k1 = 0
    hit_k3 = 0
    labeled_total = 0
    cat_hit = 0
    cat_total = 0
    rec_total = 0
    no_candidate = 0
    cat_hit_labeled = 0
    cat_total_labeled = 0
    unevaluable_skipped = 0
    unevaluable_reasons: dict[str, int] = {}

    per_item: list[dict] = []
    db: Session = SessionLocal()
    t0 = time.perf_counter()

    try:
        all_ri = get_all_recipes_with_ingredients(db)
        available_ids = get_available_ingredient_ids(db)

        for i, row in enumerate(df.itertuples(index=False), 1):
            case_id = getattr(row, "case_id", i)
            user_text = getattr(row, "user_text", "")

            gold_cocktails = _parse_json_list(row.gold_expected_cocktails)
            gold_categories = _parse_json_list(row.gold_allowed_categories)
            has_exact_gold = bool(gold_cocktails)
            has_cat_gold = bool(gold_categories)

            if not has_exact_gold and not has_cat_gold:
                unevaluable_skipped += 1
                reason = _s(getattr(row, "unlabelable_reason", None)) or "unknown"
                unevaluable_reasons[reason] = unevaluable_reasons.get(reason, 0) + 1
                continue

            rec: dict = {"case_id": case_id, "user_text": user_text, "mode": "seeded_e2e"}

            g_mood = _s(getattr(row, "gold_current_mood", None))
            g_party = _s(getattr(row, "gold_party_purpose", None))
            g_strength = _s(getattr(row, "gold_strength_preference", None))
            g_taste = _parse_json_dict(getattr(row, "gold_taste_profile", None))
            g_aroma = _parse_json_dict(getattr(row, "gold_aroma_profile", None))
            g_bases_list = _parse_json_list(getattr(row, "gold_disliked_bases", None))

            rec["current_mood_gold"] = g_mood
            rec["party_purpose_gold"] = g_party
            rec["strength_preference_gold"] = g_strength
            rec["taste_gold"] = json.dumps(g_taste, ensure_ascii=False)
            rec["aroma_gold"] = json.dumps(g_aroma, ensure_ascii=False)
            rec["bases_gold"] = json.dumps(sorted(g_bases_list), ensure_ascii=False)
            rec["unlabelable_reason"] = _s(getattr(row, "unlabelable_reason", None)) or ""

            if resolved_seed_source == "gold_pseudo":
                seed_slots = _gold_to_seed_slots(row)
            else:
                seed_slots = _row_to_column_seed_slots(row)
            rec["seed_slots"] = json.dumps(seed_slots, ensure_ascii=False)
            rec["seed_source"] = resolved_seed_source

            result = analyze_user_turn(
                history=[],
                slots=seed_slots,
                user_msg=user_text,
                generate_next_question=False,
            )
            pred_raw = result.get("extracted_slots") or {}
            pred_merged = merge_slots(seed_slots, pred_raw)

            rec["current_mood_pred_raw"] = pred_raw.get("current_mood")
            rec["party_purpose_pred_raw"] = pred_raw.get("party_purpose")
            rec["strength_preference_pred_raw"] = pred_raw.get("strength_preference")
            rec["taste_pred_raw"] = json.dumps(pred_raw.get("taste_profile") or {}, ensure_ascii=False)
            rec["aroma_pred_raw"] = json.dumps(pred_raw.get("aroma_profile") or {}, ensure_ascii=False)
            rec["bases_pred_raw"] = json.dumps(sorted(pred_raw.get("disliked_bases") or []), ensure_ascii=False)

            for slot_key, g in [
                ("current_mood", g_mood),
                ("party_purpose", g_party),
                ("strength_preference", g_strength),
            ]:
                p = pred_merged.get(slot_key)
                rec[f"{slot_key}_pred"] = p
                if g is None:
                    rec[f"{slot_key}_hit"] = ""
                    continue
                ok = (p == g)
                rec[f"{slot_key}_hit"] = int(ok)
                enum_acc[slot_key].add(ok)

            p_taste = pred_merged.get("taste_profile") or {}
            rec["taste_pred"] = json.dumps(p_taste, ensure_ascii=False)
            if g_taste or p_taste:
                taste_key_prf.add(set(p_taste.keys()), set(g_taste.keys()))
                taste_kv_prf.add(
                    {(k, v) for k, v in p_taste.items()},
                    {(k, v) for k, v in g_taste.items()},
                )

            p_aroma = pred_merged.get("aroma_profile") or {}
            rec["aroma_pred"] = json.dumps(p_aroma, ensure_ascii=False)
            if g_aroma or p_aroma:
                aroma_key_prf.add(set(p_aroma.keys()), set(g_aroma.keys()))
                aroma_kv_prf.add(
                    {(k, v) for k, v in p_aroma.items()},
                    {(k, v) for k, v in g_aroma.items()},
                )

            g_bases = set(g_bases_list)
            p_bases = set(pred_merged.get("disliked_bases") or [])
            rec["bases_pred"] = json.dumps(sorted(p_bases), ensure_ascii=False)
            if g_bases or p_bases:
                bases_prf.add(p_bases, g_bases)

            profile = _seeded_profile(seed_slots, pred_raw)
            diag: dict = {}
            top3 = _llm_top3(
                db,
                profile,
                all_ri,
                available_ids,
                diagnostics=diag,
                gold_cocktails=gold_cocktails,
                gold_categories=gold_categories,
            )
            rec.update(diag)

            rec_total += 1
            if has_exact_gold:
                labeled_total += 1
            if has_cat_gold:
                cat_total += 1
            if has_exact_gold and has_cat_gold:
                cat_total_labeled += 1

            if not top3:
                no_candidate += 1
                rec["top3_names"] = ""
                rec["top3_categories"] = ""
                rec["top3_sources"] = ""
                rec["hit_1"] = 0 if has_exact_gold else ""
                rec["hit_3"] = 0 if has_exact_gold else ""
                rec["cat_hit_3"] = 0 if has_cat_gold else ""
                rec["no_candidate"] = 1
            else:
                h1 = int(has_exact_gold and top3[0]["name_kr"] in gold_cocktails)
                h3 = int(has_exact_gold and any(t["name_kr"] in gold_cocktails for t in top3))
                ch = int(has_cat_gold and any(t["category"] in gold_categories for t in top3))
                if has_exact_gold:
                    hit_k1 += h1
                    hit_k3 += h3
                if has_cat_gold:
                    cat_hit += ch
                    if has_exact_gold:
                        cat_hit_labeled += ch

                rec["top3_names"] = json.dumps([t["name_kr"] for t in top3], ensure_ascii=False)
                rec["top3_categories"] = json.dumps([t["category"] for t in top3], ensure_ascii=False)
                rec["top3_sources"] = json.dumps([t.get("source", "") for t in top3], ensure_ascii=False)
                rec["hit_1"] = h1 if has_exact_gold else ""
                rec["hit_3"] = h3 if has_exact_gold else ""
                rec["cat_hit_3"] = ch if has_cat_gold else ""
                rec["no_candidate"] = 0

            rec["has_exact_gold"] = int(has_exact_gold)
            rec["gold_cocktails"] = json.dumps(gold_cocktails, ensure_ascii=False)
            rec["gold_categories"] = json.dumps(gold_categories, ensure_ascii=False)

            per_item.append(rec)

            if i % 25 == 0:
                dt = time.perf_counter() - t0
                print(f"  progress {i}/{n}  elapsed={dt:.1f}s")

    finally:
        db.close()

    elapsed = time.perf_counter() - t0
    rec_denom = max(rec_total, 1)
    label_denom = max(labeled_total, 1)
    cat_denom = max(cat_total, 1)

    lines: list[str] = []

    def _log(msg: str = ""):
        print(msg)
        lines.append(msg)

    if tag:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = os.getenv("LLM_MODEL", "")
        per_dir = Path("eval_results/cases/e2e")
        per_dir.mkdir(parents=True, exist_ok=True)
        per_csv = per_dir / f"{short_model_name(model_name)}_e2e_{tag}_{stamp}.csv"
        pd.DataFrame(per_item).to_csv(per_csv, index=False)
        _log(f"[per-item] {len(per_item)} cases → {per_csv}  (model={model_name})")

    _log("\n" + "=" * 60)
    _log(
        f"E2E EVAL — mode=seeded_e2e  total_rows={n}  evaluable={rec_total}  "
        f"unevaluable_skipped={unevaluable_skipped}  ({elapsed:.1f}s, {elapsed/max(n,1):.2f}s/case)"
    )
    if unevaluable_reasons:
        reasons_fmt = ", ".join(
            f"{k}={v}" for k, v in sorted(unevaluable_reasons.items(), key=lambda x: -x[1])
        )
        _log(f"  unevaluable breakdown: {reasons_fmt}")
    _log("=" * 60)
    if resolved_seed_source == "gold_pseudo":
        _log("  seed strategy: gold-derived pseudo initial tags (analysis only; leakage-heavy)")
    else:
        _log("  seed strategy: actual initial-tag columns from the eval CSV")

    _log("\n[Slot State After Seed + Refinement]")
    _log("  scalar enums — exact match")
    for k, acc in enum_acc.items():
        _log(f"    {k:22s}: {acc.correct:4d}/{acc.total:4d} = {acc.pct():5.1f}%")

    def _pr(label, prf: PRF):
        p, r, f = prf.f1()
        _log(f"    {label:32s}: P={p:5.1f}  R={r:5.1f}  F1={f:5.1f}  (tp={prf.tp} fp={prf.fp} fn={prf.fn})")

    _log("  intensity dicts")
    _pr("taste_profile KEY presence", taste_key_prf)
    _pr("taste_profile KEY+VALUE exact", taste_kv_prf)
    _pr("aroma_profile KEY presence", aroma_key_prf)
    _pr("aroma_profile KEY+VALUE exact", aroma_kv_prf)
    _log("  list enum")
    _pr("disliked_bases", bases_prf)

    _log("\n[Recommendation (seeded slots + extracted refinement → top-3 vs gold)]")
    _log(f"  exact gold 있는 케이스      : {labeled_total}/{rec_total} = {labeled_total/rec_denom*100:.1f}%")
    _log(f"  Hit@1  (labeled)            : {hit_k1}/{labeled_total} = {hit_k1/label_denom*100:.1f}%")
    _log(f"  Hit@3  (labeled)            : {hit_k3}/{labeled_total} = {hit_k3/label_denom*100:.1f}%")
    _log(f"  카테고리 Hit@3 (all cases)  : {cat_hit}/{cat_total} = {cat_hit/cat_denom*100:.1f}%")
    _log(f"  카테고리 Hit@3 (labeled)    : {cat_hit_labeled}/{cat_total_labeled} = {cat_hit_labeled/max(cat_total_labeled,1)*100:.1f}%")
    _log(f"  후보 없음                   : {no_candidate}/{rec_total} = {no_candidate/rec_denom*100:.1f}%")

    _, _, taste_key_f1 = taste_key_prf.f1()
    _, _, taste_kv_f1 = taste_kv_prf.f1()
    _, _, aroma_key_f1 = aroma_key_prf.f1()
    _, _, aroma_kv_f1 = aroma_kv_prf.f1()
    _, _, bases_f1 = bases_prf.f1()
    scalar_pcts = [acc.pct() for acc in enum_acc.values() if acc.total]
    scalar_avg = sum(scalar_pcts) / len(scalar_pcts) if scalar_pcts else 0.0

    return {
        "mode": "seeded_e2e",
        "seed_strategy": resolved_seed_source,
        "n": n,
        "rec_n": rec_total,
        "unevaluable_skipped": unevaluable_skipped,
        "unevaluable_reasons": unevaluable_reasons,
        "elapsed_sec": elapsed,
        "scalar_avg": scalar_avg,
        "scalar_per_slot": {k: acc.pct() for k, acc in enum_acc.items()},
        "taste_key_f1": taste_key_f1,
        "taste_kv_f1": taste_kv_f1,
        "aroma_key_f1": aroma_key_f1,
        "aroma_kv_f1": aroma_kv_f1,
        "bases_f1": bases_f1,
        "exact_gold_coverage": labeled_total / rec_denom,
        "hit@1_labeled": hit_k1 / label_denom,
        "hit@3_labeled": hit_k3 / label_denom,
        "cat_hit@3_all": cat_hit / cat_denom,
        "cat_hit@3_labeled": cat_hit_labeled / max(cat_total_labeled, 1),
        "no_candidate_rate": no_candidate / rec_denom,
        "labeled_n": labeled_total,
        "cat_n": cat_total,
        "_summary_lines": lines,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="저장 라벨. 지정 시 eval_results/summary/e2e/{model}_e2e_{tag}_{stamp}.{json,txt} 저장.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="결과 메타 라벨 전용 — 실제 LLM 은 .env 의 LLM_MODEL 로 정해짐.",
    )
    ap.add_argument(
        "--seed-source",
        type=str,
        default="columns",
        choices=("columns", "gold_pseudo"),
        help="seed source. columns=실제 initial tag 컬럼, gold_pseudo=분석용 정답 유도 seed.",
    )
    args = ap.parse_args()

    env_model = os.getenv("LLM_MODEL", "")
    print(f"[env] LLM_MODEL={env_model or '(unset)'}")
    if args.model and env_model and args.model.lower() not in env_model.lower():
        print(
            f"[WARN] --model='{args.model}' 가 env LLM_MODEL='{env_model}' 와 일치하지 않음. "
            f"결과 라벨과 실제 실행 모델이 다를 수 있음. 계속하려면 5초 대기..."
        )
        time.sleep(5)

    result = run_eval(limit=args.limit, tag=args.tag, seed_source=args.seed_source)
    if args.tag:
        summary_lines = result.pop("_summary_lines", [])
        json_path, txt_path = save_eval_result(
            kind="e2e",
            tag=args.tag,
            limit=args.limit,
            payload=result,
            summary_lines=summary_lines,
            model=args.model,
        )
        print(f"\n[saved] {json_path}")
        print(f"[saved] {txt_path}")
