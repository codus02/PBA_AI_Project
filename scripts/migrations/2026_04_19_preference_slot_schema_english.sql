BEGIN;

-- 0. guest_sessions 보조 컬럼 보장
ALTER TABLE guest_sessions
    ADD COLUMN IF NOT EXISTS conversation_stage VARCHAR(30) NOT NULL DEFAULT 'INIT',
    ADD COLUMN IF NOT EXISTS feedback_round INTEGER NOT NULL DEFAULT 0;

-- 1. old flat columns 제거 (예전 스키마 흔적)
ALTER TABLE preference_slots
    DROP COLUMN IF EXISTS preferred_tastes_json,
    DROP COLUMN IF EXISTS disliked_tastes_json,
    DROP COLUMN IF EXISTS preferred_aromas_json,
    DROP COLUMN IF EXISTS disliked_aromas_json,
    DROP COLUMN IF EXISTS favorite_drinks_text;

-- 2. 신규 profile/json 컬럼 보장
ALTER TABLE preference_slots
    ADD COLUMN IF NOT EXISTS taste_profile_json JSONB,
    ADD COLUMN IF NOT EXISTS aroma_profile_json JSONB,
    ADD COLUMN IF NOT EXISTS favorite_drinks_json JSONB;

-- 3. finish_preference 제거
ALTER TABLE preference_slots
    DROP COLUMN IF EXISTS finish_preference;

-- 4. 컬럼 길이 정리
ALTER TABLE preference_slots
    ALTER COLUMN party_purpose TYPE VARCHAR(30),
    ALTER COLUMN current_mood TYPE VARCHAR(30),
    ALTER COLUMN strength_preference TYPE VARCHAR(10);

-- 5. dev 환경 가정: 기존 슬롯 데이터 초기화
--    enum 축이 많이 바뀌었으므로 안전하게 리셋
UPDATE preference_slots
SET
    party_purpose = NULL,
    current_mood = NULL,
    strength_preference = CASE
        WHEN strength_preference = 'none' THEN 'zero'
        WHEN strength_preference IN ('zero', 'light', 'medium', 'strong') THEN strength_preference
        ELSE NULL
    END,
    taste_profile_json = NULL,
    aroma_profile_json = NULL,
    disliked_bases_json = NULL,
    favorite_drinks_json = NULL,
    slot_completion_score = 0.00;

COMMIT;