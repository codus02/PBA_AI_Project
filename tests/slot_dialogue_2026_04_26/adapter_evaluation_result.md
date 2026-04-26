# 어댑터 (2B + LoRA) 검증 결과 — 최종 (평가 완료)

## 실행 정보

| 항목 | 값 |
|---|---|
| 어댑터 | `models/slot_extractor_v2_2_adapter` (LoRA r=32, α=64, 5 epochs) |
| 베이스 모델 | `google/gemma-2-2b-it` (4-bit) |
| 평가 데이터 | v2_2_500.csv (500 rows, 467 evaluable, 33 unevaluable) |
| 평가 모드 | `--mode e2e --tag adapter_v2_2_e2e_full_check --model Gemma` (rerank=on, retrieve_n=100, rerank_pool=12) |
| 시작 시각 | 2026-04-26 01:06 |
| 총 소요 | **8시간 59분** (32345.3s, 64.69s/case) |
| 종료 시각 | 2026-04-26 10:05 |
| 결과 CSV | `eval_results/cases/e2e/gemma_e2e_adapter_v2_2_e2e_full_check_20260426_1005.csv` |
| 로그 | `eval_results/check/run_adapter_v2_2_e2e.log` |

---

## 1. Sanity Check ✅

단일 inference (`analyze_user_turn`):
```
입력: "쓴맛 강한 게 좋고 우디 향 진했으면 해. 위스키는 빼줘"
출력:
{
  "taste_profile":  {"bitter": "high"},
  "aroma_profile":  {"woody": "high"},
  "disliked_bases": ["whiskey"]
}
```
→ schema 정확히 따름.

---

## 2. 최종 지표

### 2-1. Slot Extraction (467 evaluable cases)

| 지표 | 결과 |
|---|---|
| current_mood EM | **95.9%** (402/419) |
| party_purpose EM | **92.6%** (390/421) |
| strength_preference EM | 75.8% (316/417) |
| taste_profile KEY presence F1 | 99.8% (P=99.9, R=99.7) |
| **taste_profile KEY+VALUE F1** | **96.1%** (P=96.2, R=96.0) |
| aroma_profile KEY presence F1 | 99.6% (P=99.4, R=99.9) |
| **aroma_profile KEY+VALUE F1** | **95.1%** (P=94.8, R=95.3) |
| disliked_bases F1 | 93.8% (P=98.3, R=89.7) |

### 2-2. Recommendation (extracted slots → top-3 vs gold)

| 지표 | 결과 |
|---|---|
| exact gold 있는 케이스 | 382/467 = 81.8% |
| **Hit@1 (labeled)** | **57.6%** (220/382) |
| **Hit@3 (labeled)** | **79.8%** (305/382) |
| 카테고리 Hit@3 (all cases) | **85.7%** (400/467) |
| 카테고리 Hit@3 (labeled) | **94.5%** (361/382) |
| 후보 없음 | 0/467 = 0.0% |

### 2-3. unevaluable breakdown (33 cases)

- no_candidates_passed_hard_check: 21
- zero_high_impossible: 12

= 평가 자체에서 제외된 케이스. 어댑터 문제 아님 (gold 추출 불가능 케이스).

---

## 3. 슬롯 추출 비교 — Baseline 과의 격차

⚠️ 이 비교는 **슬롯 추출 메트릭 only**. e2e Hit@1 은 다음 섹션 (§3-1) 에서 별도 처리.

| 측정 | 모델 | taste KV F1 | aroma KV F1 | mood EM | strength EM | disliked F1 |
|---|---|---|---|---|---|---|
| **베이스라인 2B prompt (n18)** | gemma-2-2b-it | 16.1% | 13.1% | 38.7% | 22.1% | (n/a) |
| 9B prompt (중간 측정, Apr 24) | gemma-2-9b-it | 87.0% | 82.2% | 54.2% | 69.1% | (n/a) |
| 노트북 어댑터 held-out | 2B + LoRA | 100% | 100% | 100% | 100% | n/a |
| **실제 e2e 어댑터 (final)** | **2B + LoRA** | **96.1%** | **95.1%** | **95.9%** | **75.8%** | **93.8%** |

**vs 베이스라인 2B prompt:**
- taste KV F1: **+80.0p**
- aroma KV F1: **+82.0p**
- mood EM: **+57.2p**
- strength EM: **+53.7p**

**vs 9B prompt (중간 측정 참고):**
- taste KV F1: +9.1p
- aroma KV F1: +12.9p
- mood EM: +41.7p
- strength EM: +6.7p

= 베이스라인 큰 폭 상회. 9B prompt 중간 측정도 모든 지표에서 상회.

---

## 3-1. e2e Hit@1 57.6% 의 위치 (별도 처리)

어댑터 평가에서 부수적으로 측정된 **e2e + rerank ON** 추천 결과:
- Hit@1 (labeled): 57.6%
- Hit@3 (labeled): 79.8%
- Cat Hit@3 (all): 85.7%

⚠️ **Track 2 ranking 베이스라인 (oracle 61.3%) 또는 H7a (oracle 68.3%) 와 직접 비교 불가**:
- 모드 다름: e2e (추출 slot) vs oracle (gold slot)
- rerank 유무 다름: ON vs OFF
- 측정 목적 다름: 어댑터 sanity check vs ranking 천장

→ 이 수치는 "어댑터 슬롯이 ranking 단계까지 정상 통과한다" 의 sanity 증거. **발표 비교표에는 포함 안 함.**

추천 ranking 의 정식 베이스라인 ↔ 최종 비교는 모두 **oracle 모드** (Track 2):
- 베이스라인 oracle Hit@1: 61.3%
- H7a oracle Hit@1: 68.3% (+7.0p)

---

## 4. 중간 평가 시점 ("보류") 오판의 원인

3:45h 시점 30건 schema drop 누적 → 진행률 ~38% 가정으로 "위반율 ~17%" 추정 → "보류" 권고.

실제:
- 진행률 추정이 보수적이었음 (실제 더 진행되어 있었던 것으로 보임)
- 최종 fn 비율: taste 4.0% / aroma 4.7% / mood 4.1% 수준
- 분포 외 표현 일부 깨지지만 case 커버리지는 매우 높음 (대부분 case 가 valid key 다수 보유)

= **노트북 100% 가 분포 안 only** 라는 진단은 부분적으로 맞지만, e2e 분포 외 표현 손실은 우려보다 훨씬 작았음.

---

## 5. 채택 결정

### 권장: ✅ **어댑터 채택**

**근거:**
1. **taste/aroma KV F1 95%+** — 베이스라인 2B prompt 대비 +80p/+82p, 9B prompt 중간 측정도 상회
2. **mood/party_purpose EM 92~96%** — 사용자 발화 다양성 잘 흡수
3. **메모리 효율** — 2B + LoRA 가 9B 보다 훨씬 가벼움 (dialogue 9B 와 동시 로드 시 GPU 여유)
4. **8시간 59분 완전 평가** 안정 동작 확인 (crash, OOM 없음)
5. e2e Hit@1 57.6% (sanity 지표) — ranking 단계까지 정상 통과 확인 (Track 2 oracle 베이스라인과 직접 비교는 불가)

### 약점

- **strength_preference EM 75.8%** — 다른 슬롯보다 ~20%p 낮음. 다만 9B baseline (69.1%) 보다는 +6.7p 높음.
  - 원인 추정: 학습 데이터의 strength 라벨 분포 편향 또는 모호한 표현 다수
  - Post-demo 개선: strength 라벨 다양성 ↑ 재학습

### 채택 절차 (5분)

```bash
# .env (현재 상태)
SLOT_EXTRACTOR_BACKEND=adapter
SLOT_EXTRACTOR_MODEL=google/gemma-2-2b-it
SLOT_EXTRACTOR_ADAPTER_PATH=models/slot_extractor_v2_2_adapter
DIALOGUE_LLM_MODEL=google/gemma-2-9b-it    # dialogue 는 9B 유지
```
→ 이미 adapter 모드. .env 변경 불필요.

```bash
# uvicorn 재시작 (이미 띄워져 있다면)
pkill -f "uvicorn app.main:app" ; ./scripts/run_dev.sh
# 또는 그냥 사용 중이면 그대로 유지
```

---

## 6. Post-demo 개선 후보

| 옵션 | 비용 | 효과 |
|---|---|---|
| **strength 라벨 다양성 ↑ + 재학습** | 中 (라벨링 + GPU) | strength EM 90%+ 달성 (현재 약점 해소) |
| **분포 외 표현 데이터 추가** | 中 | 잔여 4~5% schema 위반 ↓ |
| **Constrained decoding** | 小 (라이브러리) | 디코딩 시 schema 강제 → 위반 0 |
| **9B + LoRA** (베이스 사이즈 ↑) | 中 (GPU 메모리) | 베이스 capacity 큼 → 더 안정 |

---

## 7. 결론 (한 줄)

**어댑터 e2e 평가 완료. 슬롯 추출 모든 지표에서 베이스라인 2B prompt 대비 +57~82p, 9B prompt 중간 측정도 +6~42p 상회. 중간 평가 시점 "보류" 권고는 추정 위반율 과대평가에 의한 오판. 데모용으로 어댑터 채택 권장. e2e Hit@1 57.6% 는 Track 2 oracle 과 모드 달라 직접 비교 불가 — sanity 지표로만 사용. Post-demo 에 strength 라벨 다양성 보강.**
