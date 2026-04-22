-- cocktails 테이블에 alcohol_score 추가 (칵테일 자체 도수, 0~5 스케일).
-- 기존 preference_vectors.alcohol_score(유저 선호치)와는 다른 컬럼.
BEGIN;

ALTER TABLE cocktails
    ADD COLUMN IF NOT EXISTS alcohol_score NUMERIC(3, 1);

ALTER TABLE cocktails
    DROP CONSTRAINT IF EXISTS chk_cocktails_alcohol_score;

ALTER TABLE cocktails
    ADD CONSTRAINT chk_cocktails_alcohol_score
    CHECK (alcohol_score IS NULL OR alcohol_score BETWEEN 0 AND 5);

COMMIT;
