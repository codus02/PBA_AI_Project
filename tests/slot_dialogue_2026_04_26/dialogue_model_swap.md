# Dialogue 모델 EXAONE Swap 검토

## 배경
사용자 보고: 현재 Gemma 9B dialogue 모델의 Pass 2 reply 가 "티키타카" 자연스러움 부족. EXAONE 으로 swap 검토.

## 과거 측정 (Apr 22, 단일 모델 시점)
3-모델 비교 (Gemma / Qwen / EXAONE), 각 모델이 슬롯+추천 둘 다 담당:

```
EXAONE-3.5-7.8B-Instruct
  피드백 정확도        : 100.0%   ⭐
  슬롯 scalar 평균     :  66.7%
  taste KV F1          :  75.8    ← Gemma 9B 87 보다 11p 낮음
  aroma KV F1          :  76.8    ← Gemma 9B 82 보다  5p 낮음
  disliked F1          :  92.6    ← Gemma 9B 94 와 비슷
  Hit@1 (단일 모델)    :   8.0%   ← Gemma 9B 40 보다 32p 낮음 ⚠️
```

→ 슬롯 추출/추천 메트릭에서 EXAONE < Gemma → **Gemma 채택** 결정.

## 현재 시점에서 다시 검토하는 이유

이전 측정은 **단일 모델 (슬롯+대화 둘 다)** 이라 EXAONE 이 슬롯 추출에서 손해 봐 떨어진 것.
현재는 **모델 분리** 됨:
- 슬롯 추출: 2B + LoRA adapter (또는 9B Gemma prompt)
- 대화 (Pass 2): DIALOGUE_LLM_MODEL

= **대화만 EXAONE 으로 swap 하면** 슬롯 추출 손실 없이 대화 품질만 변경 가능.

## 코드 호환성 — 변경 거의 0

[app/utils/model_loader.py:172](../../app/utils/model_loader.py#L172) 의 `render_chat` 가 모델-agnostic:
```python
def render_chat(tokenizer, system, user, ...):
    """표준 [system, user] 시도 → 거부되면 user 에 system 프리펜드 fallback.
    이름 기반 분기 없음. 로컬 스냅샷·alias 경로에서도 안전."""
    try:
        return _apply_template(tokenizer, [{"role":"system",...}, {"role":"user",...}])
    except:
        return _apply_template(tokenizer, [{"role":"user", "content": f"{system}\n\n{user}"}])
```

= EXAONE 의 chat template 도 자동 인식. **`.env` 한 줄만 수정하면 됨.**

```bash
DIALOGUE_LLM_MODEL=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
```

## 캐시 상태
```bash
$ ls ~/.cache/huggingface/hub/ | grep -i exaone
models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct
```
→ Apr 22 측정 때 받아둔 캐시 있음. **다운로드 불필요.**

## 평가 영향 매트릭스

EXAONE swap 시 영향:

| 측정 | DIALOGUE 의존 | EXAONE swap 영향 |
|---|---|---|
| Track 2 oracle (--no-rerank) Hit@1 68.3% | ❌ score_cocktail 만 사용 | 무관 ✓ |
| Track 2 Hit@3 88.2% / Cat 96.6% | ❌ | 무관 ✓ |
| Track 1 슬롯 추출 메트릭 (KV F1, EM) | ❌ Pass 1 = SLOT_EXTRACTOR 별개 | 무관 ✓ |
| Track 1 e2e + rerank ON 추천 메트릭 | ✅ rerank 가 사용 | ⚠️ 변동 가능 |
| 대화 reply 자연스러움 | ✅ Pass 2 가 사용 | 🤖 측정 안 함 (체감으로 판단) |
| analyze_feedback (delta 추출) | ✅ Pass 2 가 사용 | ⚠️ JSON schema 따라가기 약화 가능 |

= **80% 정량 측정 무관.** 슬롯 추출 / oracle 추천 메트릭은 그대로 유효.

## 잠재 위험

### 1. EXAONE 의 schema 따라가기 약화 가능
- 슬롯 추출은 별개 모델이라 무관
- 다만 `analyze_feedback` (피드백 → JSON delta) 에서 EXAONE 이 schema 깰 수 있음
- 현재 코드에 validation/fallback 있는지 점검 필요

### 2. _strip_non_korean_tokens 영향
- [preference_agent.py](../../app/agents/preference_agent.py) 에 한자→한글 치환 로직 있음
- EXAONE 이 영어/한자 섞어 출력하면 strip 됨 → 의미 손실 가능

### 3. Bartender 페르소나 호환성
- 현재 Gemma 용 시스템 프롬프트가 EXAONE 에서도 동일 작동할 보장 없음
- 페르소나 어조 갈리거나 무시될 가능성

## 검증 절차 (15~30분)

### 1. .env swap (5초)
```bash
DIALOGUE_LLM_MODEL=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
```

### 2. chat_llm_full.py 1~2턴 (10분)
```bash
set -a && source .env && set +a && python scripts/chat_llm_full.py
```
- Pass 2 reply 자연스러운가?
- chat template 정상 작동?
- bartender 페르소나 잘 살아있나?
- 한국어 / 영어 섞임 없나?

### 3. 피드백 발화 1턴
- "더 단 거" 같은 발화로 analyze_feedback JSON 출력 정상인지

### 4. (옵션) e2e --limit 100
- 추천 메트릭 변동 확인 (예: Hit@1 ±2~5p 변동 가능)
- 슬롯 메트릭은 변화 없음 (Pass 1 별개)

### 5. 결정
- 자연스러운가? → 채택
- 깨지면 → rollback 5초 (`.env` 다시 Gemma)

## 한 줄 결론

**코드 0줄 수정, .env 1줄만. 캐시 있음. 검증 15~30분.** 어댑터 평가 마무리 후 진행 권장.
