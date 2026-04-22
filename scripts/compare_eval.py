"""Qwen vs EXAONE 평가 결과 시각화.

eval_results/{quantitative,conversation}/ 에 저장된 최신 tag별 json을 읽어
그룹 막대 차트 2장을 PNG로 저장.

실행:
    python -m scripts.compare_eval                         # 최신 자동 픽업
    python -m scripts.compare_eval --qwen <path> --exaone <path>  # 수동 지정
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


QUANT_DIR = Path("eval_results/quantitative")
CONV_DIR = Path("eval_results/conversation")
OUT_DIR = Path("eval_results")


def _pick_latest(dir_path: Path, tag: str) -> Path:
    candidates = sorted(dir_path.glob(f"{tag}_*.json"))
    if not candidates:
        raise SystemExit(f"[!] {dir_path}/{tag}_*.json 없음")
    return candidates[-1]


def _setup_font():
    for name in ["NanumGothic", "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic"]:
        try:
            mpl.font_manager.findfont(name, fallback_to_default=False)
            mpl.rcParams["font.family"] = name
            break
        except Exception:
            continue
    mpl.rcParams["axes.unicode_minus"] = False


def _grouped_bar(ax, metrics: list[str], qwen_vals: list[float],
                 exaone_vals: list[float], ylabel: str, title: str,
                 ylim: tuple[float, float] | None = None):
    x = np.arange(len(metrics))
    width = 0.38
    b1 = ax.bar(x - width / 2, qwen_vals, width, label="Qwen3-8B", color="#6A8EAE")
    b2 = ax.bar(x + width / 2, exaone_vals, width, label="EXAONE-3.5-7.8B", color="#D98E73")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if ylim:
        ax.set_ylim(*ylim)


def plot_quantitative(qwen: dict, exaone: dict, stamp: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (1) Feedback + slot scalar + favs detection + bases F1
    m1 = ["Feedback 정확도", "슬롯 scalar 평균", "favorite 검출률", "disliked_bases F1"]
    q1 = [qwen["feedback_acc_pct"], qwen["slots"]["scalar_avg"],
          qwen["slots"]["favs_detected_pct"], qwen["slots"]["bases_f1"]]
    e1 = [exaone["feedback_acc_pct"], exaone["slots"]["scalar_avg"],
          exaone["slots"]["favs_detected_pct"], exaone["slots"]["bases_f1"]]
    _grouped_bar(axes[0], m1, q1, e1, "%", "피드백 / 슬롯 기본", ylim=(0, 105))

    # (2) KV F1 — taste/aroma + per-slot scalar
    m2 = ["taste KV F1", "aroma KV F1", "mood", "party_purpose", "strength"]
    q2 = [qwen["slots"]["taste_kv_f1"], qwen["slots"]["aroma_kv_f1"],
          qwen["slots"]["scalar_per_slot"].get("current_mood", 0),
          qwen["slots"]["scalar_per_slot"].get("party_purpose", 0),
          qwen["slots"]["scalar_per_slot"].get("strength_preference", 0)]
    e2 = [exaone["slots"]["taste_kv_f1"], exaone["slots"]["aroma_kv_f1"],
          exaone["slots"]["scalar_per_slot"].get("current_mood", 0),
          exaone["slots"]["scalar_per_slot"].get("party_purpose", 0),
          exaone["slots"]["scalar_per_slot"].get("strength_preference", 0)]
    _grouped_bar(axes[1], m2, q2, e2, "%", "슬롯 세부 (KV F1 + per-slot scalar)", ylim=(0, 105))

    # (3) Recommendation
    m3 = ["Hit@1", "Hit@3", "Category Hit@3"]
    q3 = [qwen["recommendation"]["hit@1"] * 100, qwen["recommendation"]["hit@3"] * 100,
          qwen["recommendation"]["cat_hit@3"] * 100]
    e3 = [exaone["recommendation"]["hit@1"] * 100, exaone["recommendation"]["hit@3"] * 100,
          exaone["recommendation"]["cat_hit@3"] * 100]
    _grouped_bar(axes[2], m3, q3, e3, "%", "추천 매칭", ylim=(0, 105))

    fig.suptitle(
        f"정량 평가: Qwen3-8B vs EXAONE-3.5-7.8B "
        f"(qwen n={qwen['slots']['n']} / exaone n={exaone['slots']['n']})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / f"compare_{stamp}_quantitative.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_conversation(qwen: dict, exaone: dict, stamp: str) -> Path:
    qm = qwen["metrics"]
    em = exaone["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (1) rate metrics
    m1 = ["Korean only", "JSON parse", "Intent match", "Extract match", "Repeat"]
    q1 = [qm["korean_only_rate"], qm["json_parse_rate"], qm["intent_match_rate"],
          qm["extract_match_rate"], qm["repeat_rate"]]
    e1 = [em["korean_only_rate"], em["json_parse_rate"], em["intent_match_rate"],
          em["extract_match_rate"], em["repeat_rate"]]
    _grouped_bar(axes[0], m1, q1, e1, "%", "대화 품질 지표", ylim=(0, 105))

    # (2) avg recommend turn (단위 다름 → 별도)
    m2 = ["avg recommend turn"]
    q2 = [qm.get("avg_recommend_turn", 0)]
    e2 = [em.get("avg_recommend_turn", 0)]
    _grouped_bar(axes[1], m2, q2, e2, "turn", "추천 도달 턴 수 (낮을수록 좋음)")

    q_reach = qm.get("scenarios_reached_recommend", "-")
    e_reach = em.get("scenarios_reached_recommend", "-")
    fig.suptitle(
        f"대화 평가: Qwen3-8B vs EXAONE-3.5-7.8B "
        f"(n_turns qwen={qm['n_turns']}/exaone={em['n_turns']}, "
        f"scenarios_reached qwen={q_reach} / exaone={e_reach})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / f"compare_{stamp}_conversation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(qwen_quant: Path | None, exaone_quant: Path | None,
         qwen_conv: Path | None, exaone_conv: Path | None):
    _setup_font()

    qwen_quant = qwen_quant or _pick_latest(QUANT_DIR, "qwen")
    exaone_quant = exaone_quant or _pick_latest(QUANT_DIR, "exaone")
    qwen_conv = qwen_conv or _pick_latest(CONV_DIR, "qwen")
    exaone_conv = exaone_conv or _pick_latest(CONV_DIR, "exaone")

    print(f"[quant] qwen   : {qwen_quant}")
    print(f"[quant] exaone : {exaone_quant}")
    print(f"[conv ] qwen   : {qwen_conv}")
    print(f"[conv ] exaone : {exaone_conv}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    q1 = json.loads(qwen_quant.read_text())
    e1 = json.loads(exaone_quant.read_text())
    p1 = plot_quantitative(q1, e1, stamp)
    print(f"저장: {p1}")

    q2 = json.loads(qwen_conv.read_text())
    e2 = json.loads(exaone_conv.read_text())
    p2 = plot_conversation(q2, e2, stamp)
    print(f"저장: {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-quant", type=Path, default=None)
    ap.add_argument("--exaone-quant", type=Path, default=None)
    ap.add_argument("--qwen-conv", type=Path, default=None)
    ap.add_argument("--exaone-conv", type=Path, default=None)
    args = ap.parse_args()
    main(args.qwen_quant, args.exaone_quant, args.qwen_conv, args.exaone_conv)
