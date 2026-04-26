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

## 결과 매우 안 좋음 
============================================================
 FINAL EVAL — model=Gemma  tag=h6_slot_vector  20260425_2138
 limit=full
============================================================
[per-item] 467 cases → eval_results/cases/e2e/gemma_e2e_h6_slot_vector_20260425_2138.csv  (model=google/gemma-2-9b-it)

============================================================
E2E EVAL — mode=oracle  total_rows=500  evaluable=467  unevaluable_skipped=33  (52.9s, 0.11s/case)
  설정                      : rerank=off  retrieve_n=50  rerank_pool=12
  unevaluable breakdown: no_candidates_passed_hard_check=21, zero_high_impossible=12
============================================================

[Slot Extraction] skipped (oracle mode — gold 슬롯 직접 주입)

[Recommendation (gold slots → top-3 vs gold)]
  exact gold 있는 케이스      : 382/467 = 81.8%
  Hit@1  (labeled)            : 205/382 = 53.7%
  Hit@3  (labeled)            : 283/382 = 74.1%
  카테고리 Hit@3 (all cases)  : 392/467 = 83.9%
  카테고리 Hit@3 (labeled)    : 351/382 = 91.9%
  후보 없음                   : 0/467 = 0.0%

## --> 기존 가설 H5에서 추가 실험 진행하기로...

---

## 다음 단계 계획 — Retrieval 단계 진단

### 배경: H1~H5 의 한계
가설 1~5 까지 모두 `score_cocktail` 함수의 weight 조정에 집중.
누적 Hit@1 +3.9p (61.3% → 65.2%). 근데 H4 부터 효과 감소 시작 (+0p, +0.5p, +0p, +1.3p),
그 이상 짜내기 어려움 인지.

### 의문 제기
- "추천 성능 더 높일 수 없냐, e2e 로 돌리면 더 떨어지지 않냐"
- "임베딩이 잘못된 건 아니냐" — score 만 봤지 retrieval 단계는 안 봤음
- 데모 직전이라 retrieval/embedding 단계도 점검 필요성 발생

### 진단: 단계별 gold 통과율 (H5 oracle, 382 labeled)

```
              labeled 케이스 382
                    │
                    ▼ retrieval (코사인) → top 50
              356 (93.2%) 통과   ← 26건 (6.8%) 임베딩 단계에서 빠짐 ❗
                    │
                    ▼ 하드필터
              356 (93.2%) 통과   ← 0건 (필터로는 안 떨어짐)
                    │
                    ▼ score → top 3
              315 (82.5%) 통과   ← 41건 score 단계에서 #4+ 로 밀림
                    │
                    ▼ #1
              249 (65.2%)        ← 66건 top3 안인데 #1 못 잡음
```

= 점수 함수 튜닝 (66건 #1 회복) 만 봤는데 **retrieval 단계에서 26건이 이미 빠지는 cap** 발견.

### 가설: Negative preference 가 임베딩에서 약함

retrieval 빠진 26건 중 **17건 (65%)** 이 low/zero 슬롯 보유.
원인 추정:
- `synthesize_query` 가 low/zero 도 쿼리에 포함시키긴 함 ("쓴맛 은은한 쪽 선호")
- 그러나 일반 텍스트 임베딩 (Qwen3-Embedding-0.6B) 이 negative preference 를 약하게 인코딩
- 칵테일 description 도 "이 칵테일은 X 를 가졌다" 형식이지 "Y 가 없다" 형식이 아님
- → low/zero 표시한 사용자 쿼리가 그 칵테일들을 retrieve 못함

### 이론적 천장 추정
retrieval 100% 가정 시 (26 케이스 회복):
- Hit@3: 82.5% → 약 88%
- Hit@1: 65.2% → 약 70%
- = retrieval 한 단계 고치면 +4~6p 추가 회복 가능

### 다음 가설 후보

#### H7a (1줄 변경, 가장 안전)
`RAG_RETRIEVE_N = 50` → `100`. retrieval pool 2배.
- 비용: score 단계 후보 2배 (~1초 추가, 53초 → 54초)
- 리스크: 거의 없음
- 기대: retrieval 미스 26건 중 절반 회복 → **+2~3p Hit@1**

#### H7b (조금 큰 변경)
low/zero 슬롯을 쿼리에서 제외, hard filter + score 만으로 처리.
- 쿼리가 깨끗해져서 medium/high 신호에 retrieval 집중
- 근데 low/zero 만 있는 케이스 쿼리 비어버림 → 더 나빠질 가능성 있음
- 실험 가치 있지만 H7a 보다 리스크 큼

#### H7c (큰 작업, post-demo)
embedding 자체 fine-tuning. cocktail 설명 + 사용자 슬롯 페어 데이터로 contrastive 학습.
- 효과 가장 클 가능성
- 시간/리소스 부담 큼

### 결정 기준 (H7a)
- Hit@1 +1p 이상 & 다른 메트릭 -1p 이내 → 채택
- 효과 없거나 역효과 → 롤백, H7b 시도 또는 점수 튜닝 (Option 1: 카테고리 매칭 보너스) 으로 회귀

### 데모 시나리오와의 관계
- 현재 측정 모두 oracle + no-rerank
- 진짜 demo 성능 = e2e + rerank ON 환경. 슬롯 추출 정확도 + LLM rerank 두 변수 더 들어감.
- retrieval 보강은 두 환경 모두에 적용되는 개선이라 가치 있음

---

## H7a 결과 — Retrieve_n 100 (적용)

### 변경
- `RAG_RETRIEVE_N = 50 → 100`
- `scripts/eval_e2e_recommendation.py:59` + `app/agents/orchestration_agent.py:1168` 양쪽 동시 적용
- 1줄 변경, 비용 ~7초 추가 (52.6s → 59.7s)

### 결과 (oracle, --no-rerank)

| 메트릭 | H5 | H7a | Δ |
|---|---|---|---|
| **Hit@1** | 65.2% | **68.3%** | **+3.1p (+12)** |
| **Hit@3** | 82.5% | **88.2%** | **+5.7p (+22)** |
| Cat Hit@3 (all) | 85.7% | 87.8% | +2.1p |
| Cat Hit@3 (labeled) | 93.7% | **96.6%** | +2.9p |
| 갭 (#1↔#3) | 17.3p | 19.9p | +2.6p |

### 진단 검증 ✅
H5 분석에서 추정한 retrieval 천장:
- Hit@3 ~88%, Hit@1 ~70%

H7a 실측:
- Hit@3 **88.2%** (예측 적중)
- Hit@1 **68.3%** (예측 -1.7p 근접)

= retrieval 단계가 진짜 cap 이었음을 확인. 1줄 변경으로 H1~H5 의 4 가설이 합쳐 짜낸 +3.9p 와 비슷한 +3.1p 추가 획득.

### 갭 변화 분석 (17.3p → 19.9p)
신규 회복 22건 (Hit@3) 중 12건만 #1 (Hit@1) 차지, 10건은 top3 진입했지만 #1 못 잡음.
즉 갭이 벌어진 건 **나쁜 일이 아님** — 완전히 빠졌던 케이스가 적어도 top3 에는 들어왔다는 뜻. 절대 메트릭 (Hit@1) 만 보면 +3.1p 명백한 win.

### 누적 성과 (베이스라인 → H7a)

| 메트릭 | 베이스라인 | H7a | Δ |
|---|---|---|---|
| Hit@1 | 61.3% | **68.3%** | **+7.0p** |
| Hit@3 | 82.5% (추정) | **88.2%** | +5.7p |
| Cat (labeled) | 94.2% (추정) | **96.6%** | +2.4p |

### 단계별 흐름 요약

```
[1] H1+H2: vector weight 절반 + intensity 강화      → Hit@1 +1.8p
[2] H3B  : medium 보상 절감                          → +0.8p
[3] H4   : high 보상 강화                            → +0p (Hit@3 만 ↑)
[4] H5   : strength dev 가중 절감                    → +1.3p
[5] (H6  : MockVector slot-derived 실험, 채택 보류)
[6] H7a  : retrieve_n 50 → 100  ← 새 카테고리 (retrieval) → +3.1p
─────────────────────────────────────────────────
누적                                                  → +7.0p
```

**인사이트:** 점수 함수만 4 라운드 튜닝한 효과 (+3.9p) 와 retrieval pool 1줄 변경 (+3.1p) 이 비슷한 임팩트.
=> **단일 단계 깊이 파는 것보다 파이프라인 전 단계 점검이 더 효율적.**

### 결정
- H7a 채택. 데모용 코드 베이스에 반영 (`exp/gemma`).
- 추가 개선 (H7b, Option 1 카테고리 매칭) 은 시간 여유 보고 결정.
- e2e + rerank ON 환경에서 진짜 demo 성능 측정 필요 (7~8분, 별도 액션).

---

## H7d 계획 — 쿼리 어휘 칵테일 형식과 정렬

### 배경: H7a 후 잔여 retrieval 약점 진단
H7a (retrieve_n 50→100) 로 +3.1p 회복했지만 retrieve_n 늘리는 건 workaround.
근본 원인 = "Qwen3-Embedding 이 negative preference 약하게 인코딩".
직접 검증 위해 임베딩 cosine similarity 측정.

### 직접 측정한 임베딩 동작 (검증)

```
같은 axis, intensity 반대:
  cos('쓴맛 강하게 선호'    ↔ '쓴맛 은은한 쪽 선호')   = 0.8455
  cos('단맛 강하게 선호'    ↔ '단맛 거의 없는 쪽 선호') = 0.8522

다른 axis, 같은 intensity (대조군):
  cos('쓴맛 강하게 선호'    ↔ '단맛 강하게 선호')      = 0.8881  ⚠️ 더 비슷!
  cos('쓴맛 강하게 선호'    ↔ '우디 강하게 선호')      = 0.7267

= 임베딩이 intensity 의미보다 "강하게 선호" surface modifier 를
  더 강한 신호로 처리. negative preference 인코딩 약함 명확히 검증.
```

### 결정적 cross-check (쿼리 vs 칵테일 형식)

칵테일 embedding_text 는 이미 "쓴맛 약간", "쓴맛 없음" 같은 descriptive 언어 사용 중.
쿼리 형식 ("은은한 쪽 선호") 과 매칭 측정:

```
LOW 케이스 (사용자가 약한 쪽 원함):
  쿼리 '쓴맛 은은한 쪽 선호' ↔ 칵테일 '쓴맛 약간'  = 0.6990  (같은 의미)
  쿼리 '쓴맛 은은한 쪽 선호' ↔ 칵테일 '쓴맛 강함'  = 0.7361  ⚠️ 정반대 의미인데 더 가까움!

HIGH 케이스 (대조):
  쿼리 '쓴맛 강하게 선호'    ↔ 칵테일 '쓴맛 강함'   = 0.9213  ✅ 잘 매칭

ZERO:
  쿼리 '쓴맛 거의 없는 쪽 선호' ↔ 칵테일 '쓴맛 없음'  = 0.7431
```

= **사용자가 low/zero 원하는데 임베딩이 정반대 칵테일을 더 가깝게 매칭.**
**이게 retrieval 미스의 직접 메커니즘 (가설 → 검증 완료).**

### 가설 (H7d)
원인이 **쿼리 어휘 ↔ 칵테일 어휘 미스매치**라면, 쿼리 어휘를 칵테일 형식과 동일화하면
임베딩 공간에서 의미 매칭이 정렬됨.

### 변경 — `_RAG_INTENSITY_KR` 매핑 4줄

| intensity | 현재 | 칵테일 텍스트 | H7d |
|---|---|---|---|
| high | `"강하게 선호"` | `"강함"` | `"강함"` |
| medium | `"적당히 선호"` | `"중간"` | `"중간"` |
| low | `"은은한 쪽 선호"` | `"약간"` | `"약간"` |
| zero | `"거의 없는 쪽 선호"` | `"없음"` | `"없음"` |

위치: `app/agents/orchestration_agent.py:340-348`

추가 변경 없음:
- 칵테일 embedding 재인코딩 불필요 (cocktail.embedding 그대로)
- score_cocktail weight 변경 없음
- 4줄 string 교체만

### 기대
- low/zero 케이스 retrieval 정렬 회복 → retrieval miss 추가 회복
- Hit@1 +1~3p 가능 (잔여 26 retrieval miss 중 일부 + 갭 좁힘)

### 리스크 (작음)
- high/medium 도 동시에 단어 변경. high 매칭은 이미 0.92 로 좋음.
- 새 형식이 다른 쿼리 부분 ("상황/무드: ...") 과 어색할 수 있음 (자연어 흐름 약간 해침)
- 임베딩이 의미 매칭에서 토큰 매칭으로 살짝 옮겨갈 수 있음 (트레이드오프)

### 결정 기준
- Hit@1 +0.5p 이상 & 다른 메트릭 -1p 이내 → 채택
- Cat Hit@3 -1p 초과 하락 → 롤백
- 효과 없으면 → 롤백, embedding 자체 fix (post-demo)

### 데모 시나리오와의 관계
- production 에서도 동일 `_RAG_INTENSITY_KR` 매핑 사용 → e2e + 실서비스 모두 동일 효과
- 슬롯 추출 정확도와 무관 (쿼리 합성 단계 변경)

---

## H7d 결과 — 효과 무효, 롤백

### 결과 (oracle, --no-rerank)

| | H7a | H7d | Δ |
|---|---|---|---|
| Hit@1 | 68.3% | 68.3% | 0 |
| Hit@3 | 88.2% | 88.2% | 0 |
| Cat (labeled) | 96.6% | 96.6% | 0 |

**382 labeled 케이스 전부 변동 0건.** retrieved set 자체가 동일.

### 분석 — 가설은 검증, fix 만 무효

```
임베딩 페어 비교 (쿼리 ↔ 칵테일):
  '쓴맛 은은한 쪽 선호' ↔ '쓴맛 약간'   = 0.6990  (같은 의미)
  '쓴맛 은은한 쪽 선호' ↔ '쓴맛 강함'   = 0.7361  (정반대 의미인데 더 가까움) ← 가설 ✓

근데 풀 쿼리 (5~6 줄, 30~50 토큰) 안에서:
  단 한 줄 안의 4~5단어 ("강하게 선호" → "강함") 만 바뀌면
  → 전체 임베딩에 미치는 영향 미미 (다른 슬롯/상황/도수 단어가 dominant)
  → retrieved cocktail set 변동 없음
```

= 단어 페어 단위 거리 (0.7361 vs 0.6990) 의미는 분명한데, 풀 쿼리 임베딩에선 dilute 됨.
**lexical alignment 만으론 부족.** 진짜 fix 는 임베딩 공간 자체 재학습 (H7c, post-demo).

### 액션
- `_RAG_INTENSITY_KR` 4줄 원복 (rollback) ✓
- 데모 베이스라인 확정: **H7a (Hit@1 68.3% / Hit@3 88.2% / Cat 96.6%)**
- post-demo: H7c (도메인 contrastive fine-tuning) — 임베딩 공간 자체 재구성 필요

---

## 데모 베이스라인 확정 (2026-04-26)

**채택된 변경 (`exp/gemma` 브랜치):**
- H1+H2: vector weight 절반 + intensity 매칭 강화
- H3B: medium 보상 절감
- H4: high 보상 강화
- H5: strength dev 가중 절감
- H7a: retrieve_n 50 → 100

**최종 oracle 측정:**
- Hit@1: 61.3% → **68.3%** (+7.0p)
- Hit@3: → **88.2%**
- Cat (labeled): → **96.6%**

**보류 작업:**
- H6 (MockVector slot-derived): `exp/gemma-h5` 브랜치 commit 으로 보존, post-demo 재출발 마커
- H7d (쿼리 어휘 정렬): 롤백, 효과 무효
- H7c (임베딩 도메인 fine-tuning): post-demo 트랙

**다음 단계:**
- 슬롯 추출 어댑터 (2B + LoRA) 통합 후 e2e 평가 (rerank ON 환경)
- 진짜 demo 성능 = e2e + adapter + rerank ON 측정값으로 확정
