-- Add conversation_stage and feedback_round to guest_sessions
-- Required by app/db/crud.py: update_guest_stage, update_feedback_round

ALTER TABLE guest_sessions
    ADD COLUMN IF NOT EXISTS conversation_stage VARCHAR(30) NOT NULL DEFAULT 'INIT',
    ADD COLUMN IF NOT EXISTS feedback_round INTEGER NOT NULL DEFAULT 0;
