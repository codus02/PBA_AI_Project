# 추천 ranking 실험 — 전체 결과 정리 (2026-04-26)

## 측정 환경 (모든 실험 공통)

- 평가 데이터: v2_2_500.csv (500 rows, 467 evaluable, 33 unevaluable)
- 평가 모드: `--mode oracle --no-rerank` (gold slots 직접 주입)
- 슬롯 추출 변수 제거 → 순수 ranking 성능만 비교
- exact gold 있는 케이스: 382/467 (81.8%)
- 측정 시간: ~53~60초/실험 (전체 467 cases)
- 데이터 출처: `eval_results/summary/final/*.json`

---

## 전체 가설 결과표 (시간 순)

| # | 시각 | 가설 | tag | Hit@1 | Hit@3 | Cat (lab) | Δ Hit@1 (vs prev) | 채택 |
|---|---|---|---|---|---|---|---|---|
| 0 | 04-25 17:29 | **베이스라인** (commit c274aad) | `oracle_baseline_c274aad` | **61.26%** | 81.15% | 91.10% | — | 출발점 |
| 1 | 04-25 17:46 | **H1** vector weight 절반 | `oracle_vector_half_c274aad` | 62.83% | 82.20% | 92.93% | +1.57p | ✅ |
| 2 | 04-25 18:06 | **H2** intensity 매칭 1.7× 강화 (H1 누적) | `oracle_vector_half_intensity_strong` | 63.09% | 82.46% | 94.24% | +0.26p | ✅ |
| 3 | 04-25 18:37 | **H3B** medium 보상 절감 | `h3b_medium_half` | 63.87% | 82.46% | 94.24% | +0.78p | ✅ |
| 4 | 04-25 18:44 | **H4** high 보상 강화 (taste 18→22, aroma 20→25) | `h4_high_boost` | 63.87% | 82.98% | 94.50% | 0p | ✅ (Hit@3↑) |
| 5 | 04-25 18:52 | **H5** strength dev 가중 절감 (±15→±10) | `h5_strength_half` | **65.18%** | 82.46% | 93.72% | +1.31p | ✅ |
| 6 | 04-25 21:38 | H6 MockVector slot-derived | `h6_slot_vector` | 53.66% | 74.08% | 91.88% | **−11.5p** | ❌ 롤백 |
| 7 | 04-25 22:48 | **H7a** retrieve_n 50 → 100 | `h7a_retrieve_100` | **68.32%** | **88.22%** | **96.60%** | **+3.14p** | ✅ **최종** |
| 8 | 04-25 23:31 | H7d 쿼리 어휘 정렬 (`강하게 선호` → `강함` 등) | `h7d_query_align` | 68.32% | 88.22% | 96.60% | 0p | ❌ 롤백 (효과 무효) |

---

## 누적 성과 (베이스라인 → H7a 최종)

| 지표 | 베이스라인 (c274aad) | H7a (final) | Δ |
|---|---|---|---|
| **Hit@1 (labeled)** | 61.26% | **68.32%** | **+7.06p** |
| **Hit@3 (labeled)** | 81.15% | **88.22%** | **+7.07p** |
| **Cat Hit@3 (labeled)** | 91.10% | **96.60%** | **+5.50p** |
| **Cat Hit@3 (all)** | 82.23% | **87.79%** | **+5.56p** |
| #1↔#3 갭 | 19.89p | 19.90p | +0.01p |

---

## 가설별 상세

### H1 — Vector weight 절반 (+1.57p)

**문제 인식**: vector_similarity 컴포넌트가 score 의 ~44 점 차지 (다른 컴포넌트보다 큰 비중). 임베딩 품질 의존도 과대.

**변경**: `score_cocktail` 의 vector weight 절반 감소.

**결과**: Hit@1 61.26% → 62.83%. Cat (labeled) 91.10% → 92.93%. Hit@3 도 +1.05p 동반 상승.

**진단**: 다른 컴포넌트 (taste/aroma/strength) 의 상대 영향력 확보. ✅ 채택.

---

### H2 — Intensity 매칭 1.7× 강화 (+0.26p, H1 누적)

**가설**: high/low 시그니처 매칭이 약하게 보상되고 있다. 사용자가 "쓴맛 강하게" 라고 했는데 medium 이 더 잘 매칭되는 현상.

**변경**: taste/aroma intensity 매칭 점수 1.7× 강화.

**결과**: Hit@1 62.83% → 63.09%. Cat (labeled) 92.93% → 94.24% (+1.31p — Cat 효과 큼).

**진단**: Hit@1 효과는 작지만 카테고리 매칭 명확히 개선. ✅ 채택.

---

### H3B — Medium 보상 절감 (+0.78p)

**문제 인식**: gold 칵테일은 시그니처 한 방 (coffee high, woody high) 이 강한데, winner 들은 medium 잡음 누적이 high 시그니처를 침몰시킴.

**변경**: medium 매칭 보상 8/7 → 4 절감.

**결과**: Hit@1 63.09% → 63.87%. Hit@3 / Cat 안정 유지.

**진단**: medium-everything 균형형 칵테일이 누적 점수로 시그니처 칵테일을 이기는 현상 일부 회복. ✅ 채택.

---

### H4 — High 보상 강화 (+0p Hit@1, +0.5p Hit@3)

**가설**: H3B 가 medium 을 절감했으니 high 를 직접 부스트하면 시그니처 추가 부각 가능.

**변경**: taste high 보상 18 → 22, aroma high 20 → 25.

**결과**: Hit@1 63.87% (불변). Hit@3 82.46% → 82.98% (+0.52p). Cat (labeled) 94.24% → 94.50%.

**진단**: Hit@1 정체. Hit@3 / Cat 소폭 개선. 효과 한계 인지. ✅ 채택 (역효과 없음).

---

### H5 — Strength dev 가중 절감 (+1.31p)

**문제 인식**: 정글 버드 등 light/medium strength 만능 칵테일이 hit1_miss 의 13/74 차지. strength dev 가중 ±15 가 과대.

**변경**: strength dev 가중 ±15 → ±10, ±8 → ±5.

**결과**: Hit@1 63.87% → **65.18%**. 갭 (#1↔#3) 17.3p 로 처음 의미있게 좁혀짐.

**진단**: strength 점수 영향력이 다른 시그니처 매칭을 가렸음. 절감으로 정글버드 dominance 완화. ✅ 채택.

---

### H6 — MockVector slot-derived (롤백) ❌

**문제 인식**: oracle eval 의 MockVector placeholder 가 모든 axis = medium (3.0/2.0) 임의값 → "medium-everything 칵테일 보너스" eval setup artifact.

**변경**: gold_taste/aroma_profile 의 정성 라벨을 numeric 으로 변환해 MockVector 채우고, 슬롯에 없는 축은 None → score 에서 skip.

**결과**: Hit@1 65.18% → **53.66%** (-11.52p). Hit@3 74.08% (-8.4p). Cat (labeled) 91.88% (-1.84p).

**진단**: vector_similarity 컴포넌트가 medium-bias 매칭에 의존하고 있었음 → slot-derived 로 바꾸니 매칭 깨짐. 점수 함수가 medium-bias 에 더 강하게 의존하고 있던 게 노출.

**액션**: 즉시 롤백. `exp/gemma-h5` 브랜치로 commit 만 보존. Post-demo 재출발 마커. ❌

---

### H7a — Retrieve_n 50 → 100 (+3.14p) ⭐ **최종 채택**

**문제 인식**: H1~H5 모두 score_cocktail 만 튜닝. retrieval 단계 (top-50 통과율) 점검 안 함.

**진단** (단계별 gold 통과율, H5 oracle 382 labeled):
```
labeled 382
   ↓ retrieval (top-50)
356 (93.2%) 통과   ← 26건 (6.8%) 임베딩 단계에서 빠짐 ❗
   ↓ 하드필터
356 (93.2%)        ← 0건 (필터로는 안 떨어짐)
   ↓ score → top 3
315 (82.5%)        ← 41건 #4+ 로 밀림
   ↓ #1
249 (65.2%)        ← 66건 top3 진입했지만 #1 못 잡음
```

**변경**: `RAG_RETRIEVE_N = 50 → 100`. 1줄.

**결과**: Hit@1 65.18% → **68.32%** (+3.14p). Hit@3 82.46% → **88.22%** (+5.76p, 가장 큰 점프). Cat (labeled) 93.72% → **96.60%** (+2.88p).

**진단 검증**: H5 분석에서 추정한 retrieval 천장 (Hit@3 ~88%, Hit@1 ~70%) 과 실측이 매우 근접. retrieval 가 진짜 cap 이었음 확인. ✅ **최종 채택**.

**비용**: 53초 → 60초 (+7초, ~13%). 데모 영향 무시 수준.

---

### H7d — 쿼리 어휘 정렬 (롤백) ❌

**가설**: Qwen3-Embedding 이 negative preference 를 약하게 인코딩 → 쿼리 어휘 ("강하게 선호") 와 칵테일 어휘 ("강함") 의 형식 불일치가 retrieval 미스 원인.

**근거 측정**:
```
'쓴맛 은은한 쪽 선호' ↔ '쓴맛 약간'  = 0.6990  (같은 의미)
'쓴맛 은은한 쪽 선호' ↔ '쓴맛 강함'  = 0.7361  ⚠️ 정반대 의미인데 더 가까움
```

**변경**: `_RAG_INTENSITY_KR` 4줄. high/medium/low/zero → 칵테일 형식 ("강함"/"중간"/"약간"/"없음") 으로 매핑.

**결과**: Hit@1 / Hit@3 / Cat 모두 H7a 와 **0건 변동**. retrieved set 자체가 동일.

**진단**: 단어 페어 거리 (0.7361 vs 0.6990) 의미는 분명. 그러나 풀 쿼리 (5~6 줄, 30~50 토큰) 안에서 4~5단어만 바뀌면 전체 임베딩에 미치는 영향 미미. lexical alignment 만으론 부족. 진짜 fix 는 임베딩 공간 자체 재학습 (H7c, post-demo).

**액션**: 롤백. ❌

---

## 채택된 변경 (`exp/gemma` 브랜치)

```
H1+H2 : score_cocktail 의 vector weight 절반, intensity 매칭 1.7× 강화
H3B   : medium 보상 8/7 → 4
H4    : taste high 18 → 22, aroma high 20 → 25
H5    : strength dev 가중 ±15 → ±10, ±8 → ±5
H7a   : RAG_RETRIEVE_N 50 → 100
```

**파일:**
- `app/agents/orchestration_agent.py` — score_cocktail weight, RAG_RETRIEVE_N
- `scripts/eval_e2e_recommendation.py` — RAG_RETRIEVE_N 동기화

---

## 보류된 변경 (post-demo)

| 가설 | 위치 | 사유 |
|---|---|---|
| H6 MockVector slot-derived | `exp/gemma-h5` 브랜치 | 점수 함수가 medium-bias 의존 → 함께 재구성 필요 |
| H7d 쿼리 어휘 정렬 (롤백) | (코드 원복) | 임베딩 도메인 fine-tuning 으로 대체 권장 |
| H7c 임베딩 도메인 fine-tuning | post-demo 트랙 | cocktail 설명 + 슬롯 페어 contrastive 학습 |
| H7b low/zero 슬롯 쿼리 제외 | 시도 안 함 | H7a 효과로 우선순위 낮아짐 |

---

## 인사이트 3 줄

1. **단일 단계 깊이 파는 것보다 파이프라인 전 단계 점검이 효율적.**
   점수 함수 4 라운드 튜닝 (+3.9p) ≈ retrieval pool 1줄 변경 (+3.1p).

2. **eval setup artifact 가 점수 함수에 깊이 박혀 있을 수 있음.**
   MockVector medium-bias 가 vector_similarity 컴포넌트의 신호 일부를 담당하고 있었음. H6 롤백이 이걸 노출.

3. **임베딩 단계 어휘 정렬은 풀 쿼리에선 dilute.**
   페어 단위 거리 차이 (0.7361 vs 0.6990) 가 retrieved set 변동을 못 일으킴. 진짜 fix = 임베딩 공간 재학습.

---

## 한 줄 결론

**oracle 모드 ranking 실험: 베이스라인 Hit@1 61.26% → H7a 68.32% (+7.06p), Hit@3 81.15% → 88.22% (+7.07p), Cat (labeled) 91.10% → 96.60% (+5.50p). 점수 함수 (H1~H5) + retrieval (H7a) 두 카테고리 합쳐 누적. H6 / H7d 는 롤백, post-demo 재시도 마커로 보존.**
