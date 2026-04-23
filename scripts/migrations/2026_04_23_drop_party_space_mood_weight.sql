-- party_space_analysis.mood_weight 제거.
-- 가중치 사용하지 않고 mood_tags_json 만으로 추천하기로 결정.
BEGIN;

ALTER TABLE party_space_analysis
    DROP COLUMN IF EXISTS mood_weight;

COMMIT;
