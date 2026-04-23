"""E2E 평가 — user_text → LLM 슬롯 추출 → 추천 → gold 와 비교.

입력:  data/eval/e2e_recommendation_eval_v2_2_500.csv
        (user_text + gold_slots + gold_expected_cocktails + gold_allowed_categories
         + unlabelable_reason)

모드:
  --mode e2e    (default) user_text → LLM 슬롯 추출 → 추천 → gold 비교
  --mode oracle           gold_slots 직접 주입 → 추천 → gold 비교 (슬롯 추출 skip)

파이프라인 (한 케이스 당, e2e):
  1) analyze_user_turn(user_text) → extracted_slots
  2) 슬롯 메트릭 계산 (extracted vs gold)
  3) extracted_slots 로 profile 조립 (space=None, MockVector)
  4) synthesize_query → retrieve_candidates → 하드필터 → rerank_with_llm → fallback score
  5) 추천 메트릭 계산 (top-3 vs gold_cocktails / gold_categories)

oracle 은 1-2 단계 생략. 추천 로직 자체의 천장 측정.

unevaluable (gold=[] AND cat=[]) 케이스는 추천 denominator 에서 자동 제외.

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
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.preference_agent import analyze_user_turn
from app.agents.orchestration_agent import (
    _expand_retrieved_candidates,
    _has_disliked_base,
    _has_zero_taste_conflict,
    _is_unstockable,
    RERANK_REASON_POOL_N,
    rerank_with_llm,
    retrieve_candidates,
    score_cocktail,
    synthesize_query,
)
from app.db.crud import get_all_recipes_with_ingredients, get_available_ingredient_ids
from app.db.database import SessionLocal
from scripts._eval_save import save_eval_result, short_model_name

CSV_PATH = Path("data/eval/e2e_recommendation_eval_v2_2_500.csv")
RAG_RETRIEVE_N = 20
ENABLE_RERANK_PREVIEW = os.getenv("EVAL_RERANK_PREVIEW", "").lower() in ("1", "true", "yes")


# ============================================================
# 파싱 유틸
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
# MockVector (space=None 이므로 vector 는 중립값 사용)
# ============================================================

class MockVector:
    sweetness_score = Decimal("3.0")
    bitterness_score = Decimal("3.0")
    sourness_score = Decimal("3.0")
    freshness_score = Decimal("3.0")
    body_score = Decimal("3.0")
    herbal_score = Decimal("2.0")
    citrus_score = Decimal("2.0")
    alcohol_score = Decimal("3.0")


# ============================================================
# 슬롯 메트릭
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
# 추천 (기존 eval_recommendation._llm_top3 와 동일 구조)
# ============================================================

def _llm_top3(
    db: Session,
    profile: dict,
    all_ri: dict,
    available_ids,
    diagnostics: dict | None = None,
    gold_cocktails: list | None = None,
    gold_categories: list | None = None,
) -> list[dict]:
    merged = profile["merged_slots"]
    gold_set = set(gold_cocktails or [])
    cat_set = set(gold_categories or [])

    def _names(cocktails) -> str:
        return json.dumps([c.name_kr for c in cocktails], ensure_ascii=False)

    def _categories(cocktails) -> str:
        return json.dumps([c.category for c in cocktails], ensure_ascii=False)

    def _gold_hit(cocktails):
        if not gold_set:
            return ""
        return int(any(c.name_kr in gold_set for c in cocktails))

    def _cat_hit(cocktails):
        if not cat_set:
            return ""
        return int(any(c.category in cat_set for c in cocktails))

    query_text = synthesize_query(profile)
    retrieved = retrieve_candidates(
        db=db,
        query_text=query_text,
        top_k=RAG_RETRIEVE_N,
        exclude_ids=None,
        strength_preference=merged.get("strength_preference"),
    )
    retrieved = _expand_retrieved_candidates(
        db,
        retrieved,
        merged,
        all_ri,
        exclude_ids=None,
    )
    retrieved_cocktails = [c for c, _ in retrieved]
    if diagnostics is not None:
        diagnostics["query_text"] = query_text
        diagnostics["retrieved_count"] = len(retrieved_cocktails)
        diagnostics["retrieved_names"] = _names(retrieved_cocktails)
        diagnostics["retrieved_categories"] = _categories(retrieved_cocktails)
        diagnostics["gold_in_retrieved"] = _gold_hit(retrieved_cocktails)
        diagnostics["cat_in_retrieved"] = _cat_hit(retrieved_cocktails)

    survivors: list[tuple] = []
    for cocktail, dist in retrieved:
        ri = all_ri.get(cocktail.cocktail_id, [])
        if _has_disliked_base(merged, ri):
            continue
        if _is_unstockable(ri, available_ids):
            continue
        if _has_zero_taste_conflict(merged, cocktail):
            continue
        survivors.append((cocktail, dist))

    if not survivors:
        if diagnostics is not None:
            diagnostics["survivor_count"] = 0
            diagnostics["survivor_names"] = "[]"
            diagnostics["survivor_categories"] = "[]"
            diagnostics["gold_in_survivors"] = _gold_hit([])
            diagnostics["cat_in_survivors"] = _cat_hit([])
            diagnostics["score_top3_names"] = "[]"
            diagnostics["score_top3_categories"] = "[]"
            diagnostics["score_top3_scores"] = "[]"
            diagnostics["gold_in_score_top3"] = _gold_hit([])
            diagnostics["cat_in_score_top3"] = _cat_hit([])
            diagnostics["rerank_top3_names"] = ""
            diagnostics["rerank_top3_categories"] = ""
            diagnostics["final_ranker"] = "no_survivors"
        return []

    survivor_cocktails = [c for c, _ in survivors]
    if diagnostics is not None:
        diagnostics["survivor_count"] = len(survivor_cocktails)
        diagnostics["survivor_names"] = _names(survivor_cocktails)
        diagnostics["survivor_categories"] = _categories(survivor_cocktails)
        diagnostics["gold_in_survivors"] = _gold_hit(survivor_cocktails)
        diagnostics["cat_in_survivors"] = _cat_hit(survivor_cocktails)

    scored = []
    for c, _dist in survivors:
        ri = all_ri.get(c.cocktail_id, [])
        s = score_cocktail(c, profile, ri)
        scored.append({
            "cocktail_id": c.cocktail_id,
            "name_kr": c.name_kr,
            "category": c.category,
            "score": s,
            "source": "score_primary",
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    score_top3 = scored[:3]
    if diagnostics is not None:
        diagnostics["score_top3_names"] = json.dumps([t["name_kr"] for t in score_top3], ensure_ascii=False)
        diagnostics["score_top3_categories"] = json.dumps([t["category"] for t in score_top3], ensure_ascii=False)
        diagnostics["score_top3_scores"] = json.dumps([t["score"] for t in score_top3], ensure_ascii=False)
        diagnostics["gold_in_score_top3"] = "" if not gold_set else int(any(t["name_kr"] in gold_set for t in score_top3))
        diagnostics["cat_in_score_top3"] = "" if not cat_set else int(any(t["category"] in cat_set for t in score_top3))

    if ENABLE_RERANK_PREVIEW:
        rerank_pool_ids = [item["cocktail_id"] for item in scored[:max(3, RERANK_REASON_POOL_N)]]
        id_to_cocktail = {c.cocktail_id: c for c in survivor_cocktails}
        rerank_pool = [id_to_cocktail[cid] for cid in rerank_pool_ids if cid in id_to_cocktail]
        reranked = rerank_with_llm(
            profile,
            rerank_pool,
            k=len(rerank_pool),
            recipe_ingredients=all_ri,
        )

        if reranked and diagnostics is not None:
            top3 = []
            for item in reranked[:3]:
                c = id_to_cocktail.get(item["cocktail_id"])
                if c is None:
                    continue
                top3.append({
                    "name_kr": c.name_kr,
                    "category": c.category,
                    "source": "rag_llm_preview",
                })
            diagnostics["rerank_top3_names"] = json.dumps([t["name_kr"] for t in top3], ensure_ascii=False)
            diagnostics["rerank_top3_categories"] = json.dumps([t["category"] for t in top3], ensure_ascii=False)
        elif diagnostics is not None:
            diagnostics["rerank_top3_names"] = ""
            diagnostics["rerank_top3_categories"] = ""
    elif diagnostics is not None:
        diagnostics["rerank_top3_names"] = ""
        diagnostics["rerank_top3_categories"] = ""

    if diagnostics is not None:
        diagnostics["final_ranker"] = "score_primary"
    for row in score_top3:
        row["source"] = "score_primary"
    return score_top3


# ============================================================
# profile 조립
# ============================================================

def _extracted_to_profile(pred: dict) -> dict:
    merged = {
        "party_purpose": pred.get("party_purpose"),
        "current_mood": pred.get("current_mood"),
        "taste_profile": pred.get("taste_profile") or {},
        "aroma_profile": pred.get("aroma_profile") or {},
        "strength_preference": pred.get("strength_preference"),
        "disliked_bases": pred.get("disliked_bases") or [],
        "favorite_drinks": pred.get("favorite_drinks") or [],
    }
    return {"merged_slots": merged, "vector": MockVector(), "space": None}


def _gold_to_profile(row) -> dict:
    """Oracle mode: CSV gold_* 를 그대로 profile 로. 슬롯 추출 우회."""
    merged = {
        "party_purpose": _s(getattr(row, "gold_party_purpose", None)),
        "current_mood": _s(getattr(row, "gold_current_mood", None)),
        "taste_profile": _parse_json_dict(getattr(row, "gold_taste_profile", None)),
        "aroma_profile": _parse_json_dict(getattr(row, "gold_aroma_profile", None)),
        "strength_preference": _s(getattr(row, "gold_strength_preference", None)),
        "disliked_bases": _parse_json_list(getattr(row, "gold_disliked_bases", None)),
        "favorite_drinks": _parse_json_list(getattr(row, "gold_favorite_drinks", None)),
    }
    return {"merged_slots": merged, "vector": MockVector(), "space": None}


# ============================================================
# 메인 루프
# ============================================================

def run_eval(limit: int | None = None, tag: str | None = None, mode: str = "e2e"):
    if mode not in ("e2e", "oracle"):
        raise ValueError(f"mode must be 'e2e' or 'oracle', got {mode}")

    df = pd.read_csv(CSV_PATH)
    if limit:
        df = df.head(limit)
    n = len(df)

    # 슬롯 메트릭 (e2e 만 사용)
    enum_acc = {k: Accum() for k in ["current_mood", "party_purpose", "strength_preference"]}
    taste_key_prf = PRF()
    taste_kv_prf = PRF()
    aroma_key_prf = PRF()
    aroma_kv_prf = PRF()
    bases_prf = PRF()

    # 추천 메트릭 — evaluable 케이스만 분모에 포함 (gold=[] AND cat=[] 제외)
    hit_k1 = 0
    hit_k3 = 0
    labeled_total = 0          # gold_expected_cocktails 있는 케이스
    cat_hit = 0
    cat_total = 0              # gold_allowed_categories 있는 케이스
    rec_total = 0              # evaluable = (labeled OR cat-only), 추천 denominator
    no_candidate = 0
    cat_hit_labeled = 0
    cat_total_labeled = 0
    unevaluable_skipped = 0    # gold=[] AND cat=[] → denominator 에서 제외
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

            # unevaluable skip — gold 도 category 도 없으면 추천 metric 측정 불가
            if not has_exact_gold and not has_cat_gold:
                unevaluable_skipped += 1
                reason = _s(getattr(row, "unlabelable_reason", None)) or "unknown"
                unevaluable_reasons[reason] = unevaluable_reasons.get(reason, 0) + 1
                continue

            rec: dict = {"case_id": case_id, "user_text": user_text, "mode": mode}

            # --- gold slot 컬럼은 oracle/e2e 공통 (인스펙션 용) ---
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

            # --- 1~2) 슬롯 추출 + 슬롯 메트릭 (e2e 만) ---
            pred: dict = {}
            if mode == "e2e":
                result = analyze_user_turn(
                    history=[], slots={}, user_msg=user_text, generate_next_question=False,
                )
                pred = result.get("extracted_slots") or {}

                for slot_key, g in [
                    ("current_mood", g_mood),
                    ("party_purpose", g_party),
                    ("strength_preference", g_strength),
                ]:
                    p = pred.get(slot_key)
                    rec[f"{slot_key}_pred"] = p
                    if g is None:
                        rec[f"{slot_key}_hit"] = ""
                        continue
                    ok = (p == g)
                    rec[f"{slot_key}_hit"] = int(ok)
                    enum_acc[slot_key].add(ok)

                p_taste = pred.get("taste_profile") or {}
                rec["taste_pred"] = json.dumps(p_taste, ensure_ascii=False)
                if g_taste or p_taste:
                    taste_key_prf.add(set(p_taste.keys()), set(g_taste.keys()))
                    taste_kv_prf.add(
                        {(k, v) for k, v in p_taste.items()},
                        {(k, v) for k, v in g_taste.items()},
                    )

                p_aroma = pred.get("aroma_profile") or {}
                rec["aroma_pred"] = json.dumps(p_aroma, ensure_ascii=False)
                if g_aroma or p_aroma:
                    aroma_key_prf.add(set(p_aroma.keys()), set(g_aroma.keys()))
                    aroma_kv_prf.add(
                        {(k, v) for k, v in p_aroma.items()},
                        {(k, v) for k, v in g_aroma.items()},
                    )

                g_bases = set(g_bases_list)
                p_bases = set(pred.get("disliked_bases") or [])
                rec["bases_pred"] = json.dumps(sorted(p_bases), ensure_ascii=False)
                if g_bases or p_bases:
                    bases_prf.add(p_bases, g_bases)

            # --- 3) 추천 ---
            profile = _extracted_to_profile(pred) if mode == "e2e" else _gold_to_profile(row)
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

    # per-item 저장
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
    _log(f"E2E EVAL — mode={mode}  total_rows={n}  evaluable={rec_total}  "
         f"unevaluable_skipped={unevaluable_skipped}  ({elapsed:.1f}s, {elapsed/max(n,1):.2f}s/case)")
    if unevaluable_reasons:
        reasons_fmt = ", ".join(
            f"{k}={v}" for k, v in sorted(unevaluable_reasons.items(), key=lambda x: -x[1])
        )
        _log(f"  unevaluable breakdown: {reasons_fmt}")
    _log("=" * 60)

    # 슬롯 요약 (e2e only)
    if mode == "e2e":
        _log("\n[Slot Extraction]")
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
    else:
        _log("\n[Slot Extraction] skipped (oracle mode — gold 슬롯 직접 주입)")

    # 추천 요약 — evaluable 분모 기준
    rec_header = "gold slots → top-3" if mode == "oracle" else "extracted slots → top-3"
    _log(f"\n[Recommendation ({rec_header} vs gold)]")
    _log(f"  exact gold 있는 케이스      : {labeled_total}/{rec_total} = {labeled_total/rec_denom*100:.1f}%")
    _log(f"  Hit@1  (labeled)            : {hit_k1}/{labeled_total} = {hit_k1/label_denom*100:.1f}%")
    _log(f"  Hit@3  (labeled)            : {hit_k3}/{labeled_total} = {hit_k3/label_denom*100:.1f}%")
    _log(f"  카테고리 Hit@3 (all cases)  : {cat_hit}/{cat_total} = {cat_hit/cat_denom*100:.1f}%")
    _log(f"  카테고리 Hit@3 (labeled)    : {cat_hit_labeled}/{cat_total_labeled} = {cat_hit_labeled/max(cat_total_labeled,1)*100:.1f}%")
    _log(f"  후보 없음                   : {no_candidate}/{rec_total} = {no_candidate/rec_denom*100:.1f}%")

    # payload
    _, _, taste_key_f1 = taste_key_prf.f1()
    _, _, taste_kv_f1 = taste_kv_prf.f1()
    _, _, aroma_key_f1 = aroma_key_prf.f1()
    _, _, aroma_kv_f1 = aroma_kv_prf.f1()
    _, _, bases_f1 = bases_prf.f1()
    scalar_pcts = [acc.pct() for acc in enum_acc.values() if acc.total]
    scalar_avg = sum(scalar_pcts) / len(scalar_pcts) if scalar_pcts else 0.0

    return {
        "mode": mode,
        "n": n,
        "rec_n": rec_total,
        "unevaluable_skipped": unevaluable_skipped,
        "unevaluable_reasons": unevaluable_reasons,
        "elapsed_sec": elapsed,
        # slot metrics (oracle 에선 0 으로 남지만 payload 구조는 유지)
        "scalar_avg": scalar_avg,
        "scalar_per_slot": {k: acc.pct() for k, acc in enum_acc.items()},
        "taste_key_f1": taste_key_f1,
        "taste_kv_f1": taste_kv_f1,
        "aroma_key_f1": aroma_key_f1,
        "aroma_kv_f1": aroma_kv_f1,
        "bases_f1": bases_f1,
        # rec metrics
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
    ap.add_argument("--tag", type=str, default=None,
                    help="저장 라벨. 지정 시 eval_results/summary/e2e/{model}_e2e_{tag}_{stamp}.{json,txt} 저장.")
    ap.add_argument("--model", type=str, default=None,
                    help="결과 메타 라벨 전용 — 실제 LLM 은 .env 의 LLM_MODEL 로 정해짐. "
                         "이 값은 결과 파일명/메타에만 기록됨.")
    ap.add_argument("--mode", choices=["e2e", "oracle"], default="e2e",
                    help="e2e=user_text→LLM→추천 / oracle=gold 슬롯 직접 주입→추천 (추천 로직 천장)")
    args = ap.parse_args()

    # --model 은 라벨 전용. 실제 모델과 라벨이 어긋나면 치명적이므로 경고.
    env_model = os.getenv("LLM_MODEL", "")
    print(f"[env] LLM_MODEL={env_model or '(unset)'}")
    if args.model and env_model and args.model.lower() not in env_model.lower():
        print(
            f"[WARN] --model='{args.model}' 가 env LLM_MODEL='{env_model}' 와 일치하지 않음. "
            f"결과 라벨과 실제 실행 모델이 다를 수 있음. 계속하려면 5초 대기..."
        )
        time.sleep(5)

    result = run_eval(limit=args.limit, tag=args.tag, mode=args.mode)
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
