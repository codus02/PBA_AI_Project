# 추천 ranking 개선 실험 (2026-04-25)

## 배경
- demo 3일 전. recommendation team 단독 작업.
- 출발점: dialogue 단순화 + slot extraction 개선 후 oracle Hit@1 61.3% (commit c274aad).
- 문제 인식: **Hit@1 ↔ Hit@3 갭이 ~19p** — "정답이 top3 안에는 있는데 #1 못 차지하는" 순위 문제가 큼.

## 진행 (가설 단위)

| | 변경 | Hit@1 | Hit@3 | Cat (labeled) | 갭 |
|---|---|---|---|---|---|
| 베이스라인 | - | 61.3% | ? | ? | ? |
| **H1+H2** | vector 가중치 절반 + taste/aroma intensity 매칭 1.7x 강화 | 63.1% | 82.5% | 94.2% | 19.4p |
| **H3B** | medium 보상 (8/7→4) 절감 — 잡음 줄여 high 시그니처 부각 | 63.9% (+0.8p) | 82.5% | 94.2% | 18.6p |
| **H4** | high 보상 강화 (taste 18→22, aroma 20→25) — 시그니처 직접 부스트 | 63.9% (=) | 83.0% (+0.5p) | 94.5% | 19.1p |
| **H5** | strength dev 가중 (±15→±10, ±8→±5) — 정글버드 dominance 완화 | **65.2% (+1.3p)** | 82.5% | 93.7% | **17.3p** |

## 핵심 발견

### 1. 시그니처 vs 밸런스
gold 칵테일은 **시그니처 한 방** (coffee high, herbal high, woody high) 가졌는데,
winners 는 **medium-everything 균형형** 이라 medium 신호 누적이 high 한 방을 침몰시킴.
→ H3B + H4 로 일부 회복.

### 2. Strength 지배
`정글 버드` 같은 light/medium strength 만능 칵테일이 hit1_miss 의 13/74 차지.
strength dev 가중 ±15 가 결정타. → H5 로 +1.3p 획득.
갭이 처음으로 의미있게 좁혀짐 (-1.8p).

### 3. MockVector medium bias 발견
oracle eval 의 `MockVector` placeholder 가 **모든 axis = medium (3.0/2.0)** 으로 임의 설정 →
사용자가 표시 안 한 축에 대해 cocktail 의 medium 값을 보상해 "medium-everything 칵테일 보너스" 를 제공.

case 208 score breakdown 분석:
```
정글버드 (winner) vs 알렉산더 (gold)  diff +19.09
- vector_similarity : +9.09  ← MockVector medium 매칭
- strength          : +10.00
- taste/aroma       :   0    (둘 다 동일 점수)
```
= 정글버드 dominance 의 일부는 진짜 ranking 문제, **일부는 eval setup artifact**.

## 누적 성과 (이 세션)

- Hit@1: 61.3% → **65.2%** (+3.9p)
- Hit@3: 안정 유지 (82.5%, ±0.5p 내)
- Cat (labeled): 안정 유지 (94.2% → 93.7%)
- Hit@1↔Hit@3 갭: 19.4p → **17.3p** (-2.1p)

## 추후 개선 후보 (post-demo)

**MockVector → 슬롯 기반 vector**
- 현재: 모든 축 medium 임의값 → medium-everything bias
- 개선안: `gold_taste_profile` / `gold_aroma_profile` 의 정성 라벨 (high/medium/low) 을 numeric 으로 변환해 채우고, **슬롯에 없는 축은 `None`**.
  `score_cocktail` 에 `if user_s is None: continue` 추가.
- 기대: medium-everything bias 제거, 정글버드 dominance 추가 완화 가능.
- 베이스라인부터 재측정 필요 (~5분).

## 변경 파일

- `app/agents/orchestration_agent.py` — `score_cocktail_breakdown` weight 조정
- `scripts/inspect_e2e_failures.py` — `--only hit1_miss` 옵션 추가
- `scripts/inspect_score_breakdown.py` — 신규 (case 별 score breakdown 비교)
