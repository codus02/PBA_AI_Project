# current_mood 의 추천 영향 — 코드 검증 (정정)

## 배경
어댑터가 schema 외 값 ('calm', 'neutral') 을 출력하는 케이스 발견 → validation drop → current_mood = None 처리됨. 이게 추천에 어떤 영향 주는지 점검.

## 초기 분석 (정정 전)
> "current_mood 는 score_cocktail 의 직접 weight 컴포넌트가 없으므로 drop 돼도 추천에 거의 영향 없음. 사용자 마찰 / 대화 reply 톤만 영향."

→ **틀림.** 코드 검증 결과 current_mood 가 추천 ranking 에 두 경로로 직접 영향.

## 코드 검증 (정정)

### 경로 1: synthesize_query 의 retrieval 쿼리
[orchestration_agent.py:438](../../app/agents/orchestration_agent.py#L438)
```python
mood = merged.get("current_mood")
if purpose or mood:
    parts.append(f"상황/무드: {_RAG_MOOD_KR.get(mood)}")
```
→ mood 가 retrieval 쿼리 텍스트에 들어감 → Qwen3-Embedding cosine 매칭에 영향

### 경로 2: context_embedding → score_cocktail 의 context 컴포넌트
[orchestration_agent.py:235](../../app/agents/orchestration_agent.py#L235), [:268](../../app/agents/orchestration_agent.py#L268)
```python
def _build_context_query(profile):
    mood = merged.get("current_mood")
    parts.append(f"현재 기분: {_RAG_MOOD_KR.get(mood)}")
    ...

def _score_context_similarity(cocktail, context_embedding):
    sim = _cosine_similarity(context_embedding, cocktail.embedding)
    return min(sim * 12.0, 12.0)   # ← 최대 +12점 가산
```
→ mood 가 context_query 의 일부 → 임베딩 만들어 cocktail 임베딩과 cosine 비교 → **최대 +12점** score 컴포넌트.

## Score 컴포넌트별 weight 분포 (H1~H7a 후)

| 컴포넌트 | 최대 점수 | mood 영향? |
|---|---|---|
| vector_similarity | ~44 | ❌ 무관 |
| taste_profile | ±22/축 | ❌ 무관 |
| aroma_profile | ±25/축 | ❌ 무관 |
| space_mood (이미지) | +15 | ❌ current_mood 와 별개 |
| **context** | **+12** | ✅ **directly** |
| favorite_drinks | +7 | ❌ 무관 |
| strength | ±10 | ❌ 무관 |

= context 컴포넌트는 strength (±10) 와 비슷한 weight. **절대 작지 않음.**

## current_mood = None 일 때 실제 영향

```
synthesize_query (retrieval 쿼리):
  "상황/무드: 데이트, 좋음"     ← mood 있을 때
  "상황/무드: 데이트"           ← mood None 일 때 (mood 빠짐)
       ↓
  retrieval 임베딩 변경 → 일부 mood-매칭 칵테일 누락 가능

context_embedding (score 컴포넌트):
  "현재 기분: 좋음" 포함 임베딩  ← mood 있을 때
  mood 없는 임베딩              ← mood None 일 때
       ↓
  cocktail 매칭 약화 → context 컴포넌트 0~몇 점만 (정상은 5~12)
```

= **'calm' drop → 추천 ranking 변동 발생.** mood 신호가 retrieval 단계 + score 단계 둘 다에서 손실.

## 정규화 ('calm' → 'soso') 의 실제 효과 (정정)

| 측면 | 정규화 없음 (drop) | 정규화 있음 |
|---|---|---|
| 슬롯 추출 정답률 | 같음 | 같음 |
| **retrieval 결과** | mood 신호 빠짐 | mood 신호 들어감 |
| **score context 컴포넌트** | 0 (mood 손실) | 활성 (~12점 가능) |
| 사용자 마찰 (재질문) | "또 묻네" | 회복 |
| 대화 reply 톤 | mood 무시 generic | mood 반영 |

= **정규화는 UX 만이 아니라 추천 ranking 에도 직접 영향.** 가치 큼.

## Trade-off (정규화 매핑의 한계)

```
사용자 의도: "차분한 분위기" = ?
  - 'good' (편안한 긍정)?
  - 'soso' (평이함)?

데이터 빌더의 의도 모름. 추측 기반 매핑 = 의도와 다른 추천 위험.
```

→ 안전한 매핑 (drop 보다 낫지만 진짜 정확하진 않음) vs 의미 손실 (drop) 의 trade-off.
근본 해결: schema 확장 (post-demo).

## Schema 한계의 본질

current_mood 스키마가 사용자 mood 의 다양성 못 받음:

| 사용자 표현 | 실제 의미 | 강제 매핑 |
|---|---|---|
| "기분 좋아!" | 활기찬 긍정 | good ✓ |
| "차분해" | 잔잔한 긍정 | good? soso? |
| "그저 그래" | 무덤덤 | soso ✓ |
| "지친다" | 피곤하지만 부정 아님 | bad? soso? |
| "우울" | 분명 부정 | bad ✓ |

= 양극 (good/bad) 사이의 미묘한 mood 다 'soso' 로 묶이거나 강제 분류됨. 데이터 다양성 부족 (soso 159건 = 32%) 의 원인.

## Post-demo 개선 후보

| 옵션 | 비용 | 효과 |
|---|---|---|
| Schema 확장 (mood 6~8 카테고리) | 대 (재라벨링) | 진짜 다양성 수용 |
| Rule-based 정규화 (calm→soso 등) | 소 (1시간) | 즉시 효과, 추측 매핑 위험 |
| 학습 데이터 다양성 ↑ + 재학습 | 중 | LoRA 가 더 많은 표현 학습 |
| Constrained decoding | 소 (라이브러리 추가) | schema 강제 보장 |
