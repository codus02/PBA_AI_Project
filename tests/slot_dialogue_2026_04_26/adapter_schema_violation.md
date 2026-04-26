# 어댑터 schema 위반 분석 — 베이스 모델 prior 누수

## 관찰
어댑터 e2e 평가 진행 중 schema 위반 출력 다수 발생:
```
validate_extracted_slots dropped: current_mood='calm'  (허용값: bad/good/soso)
validate_extracted_slots dropped: current_mood='neutral'
validate_extracted_slots dropped: party_purpose='quiet'  (허용값: business/celebration/date/hangout/solo)
validate_extracted_slots dropped: aroma_profile invalid: {'lemon': 'drop martini'}
```

## 데이터셋 검증
v2_2_500.csv 의 gold_current_mood 분포:
```
soso   : 159
bad    : 155
good   : 136
(null) :  50
─────────
total  : 500
```
→ **'calm', 'neutral', 'quiet' 등은 학습/평가 데이터에 0건.**

## 원인 — LoRA 의 한계

```
LoRA rank=32 → 전체 가중치의 1.5% 만 업데이트
나머지 98.5% = 사전학습 Gemma-2-2B 그대로
```

베이스 Gemma 가 일반 텍스트로 학습되면서:
- 한국어 "차분해", "평온해", "특별할 거 없어" 같은 표현 → 영어 mood 어휘 ("calm", "neutral", "quiet") 와 강하게 연결됨
- LoRA 360 샘플 × 5 epoch 으로는 이런 베이스 prior 를 **완전히 누르지 못함**
- 학습에서 본 적 있는 표현 → 정확히 soso/bad/good 출력
- 학습에서 못 본 변형 표현 → 베이스 prior 가 이겨 영어 단어 출력

## 노트북 100% 정합성

| 평가 데이터 | 결과 | 해석 |
|---|---|---|
| 노트북 held-out 100 (v2_2 분포 안) | 100% | 학습 분포 어휘 → 모두 매핑 성공 |
| e2e 467 cases (v2_2 전체) | schema 위반 다수 | 학습 분포 안인데도 schema 위반 발생 = 분포 변동성 노출 |

= **노트북 100% = "학습 분포 안" 의 결과.** 진짜 일반화 (분포 외 사용자 발화) 는 보장 안 됨.

## 시스템 안전망

```python
# preference_agent.validate_extracted_slots
allowed = ['bad', 'good', 'soso']
if value not in allowed:
    drop()                       # → 슬롯 추출 실패로 처리
```

→ schema 위반 발생해도 추천 파이프라인 망가지지 않음. 다만 그 슬롯은 손실.

## 의의

1. **어댑터는 노트북 수치만큼 generalization 강하지 않음.** 학습 분포 외 어휘에서 베이스 prior 노출.
2. **데모 시 사용자 발화 패턴이 학습 분포와 다르면 schema 위반 빈도 ↑** 가능.
3. **데모 영향:** 추천 자체는 작동 (안전망), 다만 개별 슬롯 손실 → 사용자가 같은 정보 재차 표현하는 UX 마찰. 또한 [current_mood_impact.md](current_mood_impact.md) 참조 — current_mood drop 시 ranking 영향까지 발생.

## Post-demo 개선 후보

| 옵션 | 효과 | 비용 |
|---|---|---|
| 학습 데이터 다양성 ↑ (특히 mood/party 표현) | schema 위반 ↓ | 데이터 라벨링 |
| LoRA rank ↑ (32→64) + 더 긴 학습 | 베이스 prior 약화 | GPU 시간 |
| Constrained decoding (`outlines`, `lm-format-enforcer`) | schema 강제 보장 | 코드 추가 + 추론 약간 느려짐 |
| Rule-based 정규화 ("calm/neutral" → "soso" 매핑) | 즉시 효과 | 매핑 사전 작성 1시간 |

데모 직전엔 4번 (rule-based 정규화) 이 가장 빠른 효과. 단 [current_mood_impact.md](current_mood_impact.md) 의 trade-off 검토 필요.
