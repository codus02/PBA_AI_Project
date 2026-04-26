-- Gemini API 키 로테이션을 위한 사용량 추적 테이블.
--   * 키별 분당/일간 호출 수 보존 (server restart 후에도 유지)
--   * 동시 요청 시 atomic 한 키 선택을 위해 row 단위 잠금 사용
--   * key_id = SHA256(api_key)[:16] — 실제 키는 저장하지 않음 (env 에서만 읽음)

CREATE TABLE IF NOT EXISTS gemini_key_usage (
    key_id              VARCHAR(64)  PRIMARY KEY,
    label               VARCHAR(50),                    -- "key_1", "key_2" 등 인간 식별자
    minute_window_start TIMESTAMP    WITHOUT TIME ZONE, -- 현재 분 단위 윈도우 시작 시각
    minute_count        INTEGER      NOT NULL DEFAULT 0,
    day_window_start    DATE,                           -- 현재 일 윈도우 (날짜)
    day_count           INTEGER      NOT NULL DEFAULT 0,
    last_used_at        TIMESTAMP    WITHOUT TIME ZONE,
    last_quota_at       TIMESTAMP    WITHOUT TIME ZONE, -- 가장 최근 quota 초과 발생 시각
    is_disabled         BOOLEAN      NOT NULL DEFAULT FALSE,
    updated_at          TIMESTAMP    WITHOUT TIME ZONE  NOT NULL DEFAULT NOW()
);

-- 라벨로 조회할 일이 있을 수 있어 부수 인덱스.
CREATE INDEX IF NOT EXISTS idx_gemini_key_usage_label
    ON gemini_key_usage (label);
