import os
import shutil
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud import (
    create_party_session,
    create_guest_session,
    get_guest_session,
    upsert_initial_tags,
    save_space_analysis,
    create_dialogue_turn,
    list_dialogue_turns,
    get_preference_slot,
    update_preference_slots,
    get_final_recommendation_by_id,
    get_initial_tag_response,
    create_evaluation_log,
    update_guest_stage,
    update_feedback_round,
    count_user_turns,
)
from app.utils.config import UPLOAD_DIR
from app.agents.preference_agent import (
    run_dialogue,
    choose_next_question,
    should_move_to_recommendation,
)
from app.agents.orchestration_agent import run_recommendation, process_feedback
from app.agents.output_agent import generate_output_json


router = APIRouter()


# ============================================================
# Pydantic Schemas
# ============================================================

class PartySessionCreateRequest(BaseModel):
    session_name: Optional[str] = None
    demo_mode: bool = False


class GuestSessionCreateRequest(BaseModel):
    party_session_id: str
    guest_label: str = "게스트1"


class InitialTagRequest(BaseModel):
    familiarity: Optional[str] = Field(default=None, description="칵테일 친숙도")
    tastes: List[str] = Field(default_factory=list, description="초기 선호 맛 태그")
    strength: str = Field(..., description="초기 선호 도수/강도감 태그")
    aromas: List[str] = Field(default_factory=list, description="초기 선호 향 태그")


class DialogueRequest(BaseModel):
    message: str


class FeedbackRequest(BaseModel):
    sample_recommendation_id: str
    feedback_text: str

class EvaluationRequest(BaseModel):
    final_recommendation_id: str
    satisfaction_score: float = Field(..., ge=1.0, le=5.0, description="1~5점")
    would_reorder: bool
    review_text: Optional[str] = None


# ============================================================
# Helpers
# ============================================================

def _serialize_dialogue_turn(turn) -> dict:
    return {
        "turn_id": str(turn.turn_id),
        "guest_session_id": str(turn.guest_session_id),
        "turn_index": turn.turn_index,
        "speaker_role": turn.speaker_role,
        "utterance_text": turn.utterance_text,
        "extracted_slots_json": turn.extracted_slots_json,
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


def _slot_row_to_internal_dict(slot_row) -> dict:
    if slot_row is None:
        return {}

    return {
        "party_purpose": slot_row.party_purpose,
        "current_mood": slot_row.current_mood,
        "preferred_tastes": slot_row.preferred_tastes_json,
        "disliked_tastes": slot_row.disliked_tastes_json,
        "preferred_aromas": slot_row.preferred_aromas_json,
        "disliked_aromas": slot_row.disliked_aromas_json,
        "strength_preference": slot_row.strength_preference,
        "favorite_drinks": slot_row.favorite_drinks_text,
        "disliked_bases": slot_row.disliked_bases_json,
        "finish_preference": slot_row.finish_preference,
    }


def _serialize_preference_slot(slot_row) -> dict:
    return {
        "slot_profile_id": str(slot_row.slot_profile_id),
        "guest_session_id": str(slot_row.guest_session_id),
        "party_purpose": slot_row.party_purpose,
        "current_mood": slot_row.current_mood,
        "preferred_tastes_json": slot_row.preferred_tastes_json,
        "disliked_tastes_json": slot_row.disliked_tastes_json,
        "preferred_aromas_json": slot_row.preferred_aromas_json,
        "disliked_aromas_json": slot_row.disliked_aromas_json,
        "strength_preference": slot_row.strength_preference,
        "favorite_drinks_text": slot_row.favorite_drinks_text,
        "disliked_bases_json": slot_row.disliked_bases_json,
        "finish_preference": slot_row.finish_preference,
        "slot_completion_score": float(slot_row.slot_completion_score),
        "updated_at": slot_row.updated_at.isoformat() if slot_row.updated_at else None,
    }


# ============================================================
# Basic
# ============================================================

@router.get("/health")
def healthcheck():
    return {"status": "ok"}


# ============================================================
# 추천
# ============================================================

@router.post("/sessions/{gid}/recommend-sample")
def recommend_sample_endpoint(
    gid: str,
    force: bool = False,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    result = run_recommendation(db, gid, k=3, force=force)

    if result.get("status") == "ok":
        update_guest_stage(db, gid, "FEEDBACK_LOOP")

    return result

  


@router.post("/sessions/{gid}/feedback")
def feedback_endpoint(
    gid: str,
    req: FeedbackRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    try:
        return process_feedback(
            db=db,
            guest_session_id=gid,
            sample_recommendation_id=req.sample_recommendation_id,
            feedback_text=req.feedback_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# 1. Party / Guest Session
# ============================================================

@router.post("/sessions/party")
def create_party(req: PartySessionCreateRequest, db: Session = Depends(get_db)):
    row = create_party_session(
        db=db,
        session_name=req.session_name,
        demo_mode=req.demo_mode,
    )

    return {
        "party_session_id": str(row.party_session_id),
        "session_name": row.session_name,
        "session_status": row.session_status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "demo_mode": row.demo_mode,
    }


@router.post("/sessions/guest")
def create_guest(req: GuestSessionCreateRequest, db: Session = Depends(get_db)):
    row = create_guest_session(
        db=db,
        party_session_id=req.party_session_id,
        guest_label=req.guest_label,
    )

    return {
        "guest_session_id": str(row.guest_session_id),
        "party_session_id": str(row.party_session_id),
        "guest_label": row.guest_label,
        "session_status": row.session_status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
    }


# ============================================================
# 2. Initial Tags
# ============================================================

@router.post("/sessions/{gid}/tags")
def save_initial_tags_endpoint(
    gid: str,
    req: InitialTagRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    row = upsert_initial_tags(
        db=db,
        guest_session_id=gid,
        familiarity_tag=req.familiarity,
        taste_tags_json=req.tastes,
        strength_tag=req.strength,
        aroma_tags_json=req.aromas,
    )

    return {
        "status": "ok",
        "tag_response_id": str(row.tag_response_id),
        "guest_session_id": str(row.guest_session_id),
        "familiarity_tag": row.familiarity_tag,
        "taste_tags_json": row.taste_tags_json,
        "strength_tag": row.strength_tag,
        "aroma_tags_json": row.aroma_tags_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ============================================================
# 3. Space Image Upload
# ============================================================

@router.post("/sessions/{gid}/space-image")
async def upload_space_image(
    gid: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    filename = f"{gid}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 업로드 폴더 없으면 생성
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # TODO:
    # 나중에 분위기 파악 에이전트(Qwen2-VL 등) 연결 시
    # 아래 placeholder를 실제 분석 결과로 교체
    mood_tags_json = {
        "신나는": 0.91,
        "네온": 0.74,
    }

    analysis = save_space_analysis(
        db=db,
        party_session_id=guest.party_session_id,
        image_path=file_path,
        mood_tags_json=mood_tags_json,
        mood_weight=0.30,
    )

    return {
        "status": "ok",
        "space_analysis": {
            "space_analysis_id": str(analysis.space_analysis_id),
            "party_session_id": str(analysis.party_session_id),
            "guest_session_id": gid,
            "image_path": analysis.image_path,
            "mood_tags_json": analysis.mood_tags_json,
            "mood_weight": float(analysis.mood_weight),
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        },
    }


# ============================================================
# 4. Dialogue
# ============================================================

@router.post("/sessions/{gid}/start-dialogue")
def start_dialogue_endpoint(
    gid: str,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    slot_row = get_preference_slot(db, gid)
    current_slots = _slot_row_to_internal_dict(slot_row)

    question = choose_next_question(current_slots)

    llm_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="LLM",
        utterance_text=question,
        extracted_slots_json=None,
    )

    update_guest_stage(db, gid, "COLLECTING")

    return {
        "status": "ok",
        "stage": "COLLECTING",
        "question": question,
        "llm_turn": _serialize_dialogue_turn(llm_turn),
    }

@router.post("/sessions/{gid}/dialogue")
def dialogue_endpoint(
    gid: str,
    req: DialogueRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # 1) 사용자 턴 저장
    user_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="USER",
        utterance_text=req.message,
        extracted_slots_json=None,
    )

    # 2) 현재 슬롯 + 초기 태그 병합
    slot_row = get_preference_slot(db, gid)
    current_slots = _slot_row_to_internal_dict(slot_row)

    tag_row = get_initial_tag_response(db, gid)
    if tag_row:
        if not current_slots.get("preferred_tastes"):
            current_slots["preferred_tastes"] = tag_row.taste_tags_json or []
        if not current_slots.get("preferred_aromas"):
            current_slots["preferred_aromas"] = tag_row.aroma_tags_json or []
        if not current_slots.get("strength_preference") and tag_row.strength_tag:
            current_slots["strength_preference"] = tag_row.strength_tag

    # 3) 대화 히스토리 구성
    history_rows = list_dialogue_turns(db, gid)
    history = [
        {
            "speaker_role": row.speaker_role,
            "utterance_text": row.utterance_text,
            "extracted_slots_json": row.extracted_slots_json,
        }
        for row in history_rows
    ]

    # 4) 에이전트 실행
    agent_result = run_dialogue(
        history=history,
        slots=current_slots,
        user_msg=req.message,
    )
    extracted_slots = agent_result.get("extracted_slots", {})

    # 5) 추출된 슬롯 DB 반영
    updated_slot_row = update_preference_slots(
        db=db,
        guest_session_id=gid,
        extracted_slots=extracted_slots,
    )

    # 6) 병합 슬롯 재구성 (초기 태그 포함)
    merged_slots = _slot_row_to_internal_dict(updated_slot_row)
    if tag_row:
        merged_slots["preferred_tastes"] = list(dict.fromkeys(
            (merged_slots.get("preferred_tastes") or []) + (tag_row.taste_tags_json or [])
        ))
        merged_slots["preferred_aromas"] = list(dict.fromkeys(
            (merged_slots.get("preferred_aromas") or []) + (tag_row.aroma_tags_json or [])
        ))
        if not merged_slots.get("strength_preference") and tag_row.strength_tag:
            merged_slots["strength_preference"] = tag_row.strength_tag

    # 7) 추천 이동 판단
    user_turn_count = count_user_turns(db, gid)
    should_proceed, proceed_reason = should_move_to_recommendation(
        merged_slots=merged_slots,
        user_turn_count=user_turn_count,
        user_msg=req.message,
    )

    if should_proceed:
        update_guest_stage(db, gid, "READY_TO_RECOMMEND")
        return {
            "status": "proceed_to_recommendation",
            "guest_session_id": gid,
            "reason": proceed_reason,
            "slot_state": _serialize_preference_slot(updated_slot_row),
            "completion": float(updated_slot_row.slot_completion_score),
            "should_proceed": True,
        }

    # 8) 다음 질문 생성 + LLM 턴 저장
    question = choose_next_question(merged_slots)
    if proceed_reason == "need_required_slots":
        question = (
            "추천 전에 한 가지만 더 확인할게요. "
            + question
        )
    llm_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="LLM",
        utterance_text=question,
        extracted_slots_json=extracted_slots,
    )
    update_guest_stage(db, gid, "COLLECTING")

    return {
        "status": "ok",
        "guest_session_id": gid,
        "user_turn": _serialize_dialogue_turn(user_turn),
        "llm_turn": _serialize_dialogue_turn(llm_turn),
        "question": question,
        "extracted_slots": extracted_slots,
        "slot_state": _serialize_preference_slot(updated_slot_row) if updated_slot_row else {},
        "completion": agent_result.get("completion", 0.0),
        "should_proceed": False,
    }

@router.get("/final-output/{final_recommendation_id}")
def final_output_endpoint(
    final_recommendation_id: str,
    db: Session = Depends(get_db),
):
    final_row = get_final_recommendation_by_id(db, final_recommendation_id)
    if not final_row:
        raise HTTPException(status_code=404, detail="final_recommendation_id not found")

    return generate_output_json(
        db=db,
        cocktail_id=final_row.final_cocktail_id,
        volume_ml=90,
    )

@router.post("/sessions/{gid}/evaluation")
def save_evaluation(
    gid: str,
    req: EvaluationRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # final_recommendation 존재 확인
    final = get_final_recommendation_by_id(db, req.final_recommendation_id)
    if not final:
        raise HTTPException(status_code=404, detail="final_recommendation not found")

    if str(final.guest_session_id) != str(gid):
        raise HTTPException(status_code=403, detail="이 세션의 추천이 아닙니다")

    row = create_evaluation_log(
        db=db,
        guest_session_id=gid,
        final_satisfaction_score=req.satisfaction_score,
        would_reorder=req.would_reorder,
        review_text=req.review_text,
    )

    return {
        "status": "ok",
        "evaluation_id": str(row.evaluation_id),
        "satisfaction_score": float(row.final_satisfaction_score),
        "would_reorder": row.would_reorder,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }